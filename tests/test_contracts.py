"""Contract tests. If these fail, someone changed a shape four people depend on."""
from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from schema import api as api_mod
from schema.api import (
    AlertsResponse, BodyResponse, BudgetResponse, ChangeRequest, ChangeResponse,
    ClinicianResponse, ForecastResponse, ParseResponse, PermissionRequest,
    PermissionResponse, PlanRequest, PlanResponse, ReplayRequest, ReplayResponse,
    StateResponse, TrendsResponse,
)
from schema.events import Event, Profile
from schema.inference import get_inference
from schema.params import PatientParams, population_prior
from schema.simulate import REF_WEEK_START, get_simulator
from schema.state import FAST_IDX, N_FAST, ORGANS, DailyState, PatientState, default_fast
from engine.stub import StubSimulator

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"

# fixture file stem -> response model
FIXTURE_MODELS = {
    "state": StateResponse,
    "forecast": ForecastResponse,
    "permission": PermissionResponse,
    "plan": PlanResponse,
    "change": ChangeResponse,
    "budget": BudgetResponse,
    "alerts": AlertsResponse,
    "replay": ReplayResponse,
    "trends": TrendsResponse,
    "body": BodyResponse,
    "parse": ParseResponse,
    "clinician": ClinicianResponse,
}

DAY = datetime(2026, 9, 11)
T0 = DAY.replace(hour=7)
HORIZON_MIN = 24 * 60
DT_MIN = 5


def day_events() -> list[Event]:
    return [
        Event(type="meal", t_start=DAY.replace(hour=7, minute=30), carbs_g=60, source="photo"),
        Event(type="insulin", t_start=DAY.replace(hour=7, minute=30), units=6,
              insulin_kind="bolus", source="pump"),
        Event(type="exercise", t_start=DAY.replace(hour=12), duration_min=45, intensity=0.6,
              exercise_kind="aerobic", source="apple_health"),
        Event(type="meal", t_start=DAY.replace(hour=19), carbs_g=80, source="photo"),
        Event(type="insulin", t_start=DAY.replace(hour=19), units=8,
              insulin_kind="bolus", source="pump"),
        Event(type="alcohol", t_start=DAY.replace(hour=22, minute=30), ethanol_g=28, source="text"),
    ]


def start_state() -> PatientState:
    return PatientState(t=T0, fast=default_fast(112.0), daily=DailyState(sleep_debt_h=1.4))


# --------------------------------------------------------------------------- fixtures

@pytest.mark.parametrize("stem,model", sorted(FIXTURE_MODELS.items()))
def test_fixture_round_trips(stem: str, model: type[BaseModel]):
    path = FIXTURES / f"{stem}.json"
    assert path.exists(), f"missing fixture {path} - run `python scripts/make_fixtures.py`"
    raw = json.loads(path.read_text(encoding="utf-8"))
    obj = model.model_validate(raw)
    again = model.model_validate(obj.model_dump(mode="json"))
    assert again.model_dump(mode="json") == obj.model_dump(mode="json")


def test_request_models_round_trip():
    requests = [
        PermissionRequest(text="can I have two IPAs tonight?"),
        PlanRequest(date=DAY, calendar=day_events()[:1], wants=day_events()[2:3]),
        ChangeRequest(description="switch to morning workouts"),
        ReplayRequest(date=DAY, overrides=[]),
    ]
    for req in requests:
        cls = type(req)
        assert cls.model_validate(req.model_dump(mode="json")).model_dump(mode="json") \
            == req.model_dump(mode="json")


def test_every_api_model_is_exercised():
    """No schema.api model may be added without a fixture or an explicit sample."""
    covered = set(FIXTURE_MODELS.values()) | {
        PermissionRequest, PlanRequest, ChangeRequest, ReplayRequest,
        # component models, reached through the responses above
        api_mod.Point, api_mod.Band, api_mod.Explanation, api_mod.Safety,
        api_mod.ActiveSubstance, api_mod.Alert, api_mod.Variant, api_mod.PlanWindow,
        api_mod.Delta, api_mod.OrganView,
    }
    defined = {
        obj for _, obj in inspect.getmembers(api_mod, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == api_mod.__name__
    }
    assert defined <= covered, f"uncovered schema.api models: {sorted(c.__name__ for c in defined - covered)}"


def test_event_and_profile_round_trip():
    ev = day_events()[0]
    assert Event.model_validate(ev.model_dump(mode="json")) == ev
    p = Profile()
    assert Profile.model_validate(p.model_dump(mode="json")) == p


# --------------------------------------------------------------------------- stub simulator

def test_simulate_shapes_and_bounds():
    traj = StubSimulator().simulate(start_state(), day_events(), PatientParams(),
                                    HORIZON_MIN, DT_MIN)
    T = HORIZON_MIN // DT_MIN + 1
    assert traj.fast.shape == (T, N_FAST)
    assert len(traj.t) == T
    assert traj.energy.shape == (T,)
    assert (traj.t[1] - traj.t[0]).total_seconds() == DT_MIN * 60

    G = traj.fast[:, FAST_IDX["G"]]
    assert np.all(G >= 40) and np.all(G <= 400)
    assert np.all(np.isfinite(traj.fast))

    for key in ("tir_pct", "tbr70_pct", "tar180_pct", "min_G", "max_G", "t_min_G", "mean_G"):
        assert key in traj.summary
    assert traj.summary["min_G"] == pytest.approx(float(G.min()))
    assert traj.summary["max_G"] == pytest.approx(float(G.max()))
    assert traj.summary["mean_G"] == pytest.approx(float(G.mean()))
    assert datetime.fromisoformat(traj.summary["t_min_G"]) in traj.t
    assert traj.summary["tir_pct"] + traj.summary["tbr70_pct"] + traj.summary["tar180_pct"] \
        == pytest.approx(100.0)


def test_simulate_is_deterministic():
    a = StubSimulator().simulate(start_state(), day_events(), PatientParams(), 480, 5)
    b = StubSimulator().simulate(start_state(), day_events(), PatientParams(), 480, 5)
    assert np.allclose(a.fast, b.fast)


def test_simulate_long_shapes():
    week = [Event(type="meal", t_start=REF_WEEK_START.replace(hour=8), carbs_g=60, source="synthetic"),
            Event(type="insulin", t_start=REF_WEEK_START.replace(hour=8), units=6,
                  insulin_kind="bolus", source="synthetic")]
    lt = StubSimulator().simulate_long(start_state(), week, PatientParams(), years=10.0)
    n = len(lt.years)
    assert n == len(lt.hba1c) == len(lt.mean_G) == len(lt.hypo_per_week)
    assert sorted(lt.risk10) == sorted(ORGANS)
    assert sorted(lt.damage) == sorted(ORGANS)
    assert all(len(v) == n for v in lt.damage.values())
    assert all(0.0 <= v <= 1.0 for v in lt.risk10.values())


def test_providers_import():
    sim = get_simulator()
    assert hasattr(sim, "simulate") and hasattr(sim, "simulate_long")
    fitter, estimator, residual, drift = get_inference()
    for obj in (fitter, estimator, residual, drift):
        assert obj is not None
    assert drift.score() >= 0.0


def test_params_array_round_trip():
    p = PatientParams()
    assert PatientParams.from_array(p.to_array()) == p
    prior = population_prior(n_samples=32, seed=0)
    assert prior.samples.shape == (32, len(p.to_array()))
