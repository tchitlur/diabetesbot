"""Shape tests for RealSimulator. These are the contract between the engine and reality.

Every assertion is a clinical shape someone would recognise, not a regression on a number
this file happened to produce. Where the specification left a scenario underdetermined the
interpretation is stated in a comment; engine/NOTES.md lists them together.

Must run in under 10 seconds.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from engine.simulate import RealSimulator
from schema.events import Event
from schema.params import PatientParams
from schema.state import DailyState, FAST_IDX, PatientState, default_fast

T0 = datetime(2026, 9, 11, 7, 0)
GI = FAST_IDX["G"]
PARAMS = PatientParams()


@pytest.fixture(scope="module")
def sim() -> RealSimulator:
    return RealSimulator()


def state(G0: float = 100.0) -> PatientState:
    return PatientState(t=T0, fast=default_fast(G0), daily=DailyState())


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def meal(carbs: float, minutes: float = 30, **kw) -> Event:
    return Event(type="meal", t_start=at(minutes), carbs_g=carbs, **kw)


def bolus(units: float, minutes: float = 30) -> Event:
    return Event(type="insulin", t_start=at(minutes), units=units, insulin_kind="bolus")


def basal(hours: float = 30, u_per_h: float = 1.0) -> list[Event]:
    """A patient on a pump always has basal running. Only the first test omits it."""
    return [Event(type="insulin", t_start=T0, duration_min=hours * 60,
                  units=u_per_h, insulin_kind="basal")]


def glucose(sim: RealSimulator, events: list[Event], G0: float, hours: float,
            dt_min: int = 5) -> np.ndarray:
    return sim.simulate(state(G0), events, PARAMS, int(hours * 60), dt_min).fast[:, GI]


# --------------------------------------------------------------------------- meals

def test_unbolused_meal_peaks_where_it_should(sim):
    """60 g of carbohydrate and no insulin at all, from 100 mg/dL."""
    G = glucose(sim, [meal(60)], 100.0, 6)
    peak = G.max()
    t_peak = int(np.argmax(G)) * 5 - 30            # minutes after the meal
    assert 220 <= peak <= 300, f"peak {peak:.0f} mg/dL"
    assert 60 <= t_peak <= 120, f"peak at +{t_peak} min"


def test_matched_bolus_brings_a_meal_home(sim):
    """60 g with a carb-ratio-matched 6 U bolus should be uneventful in both directions."""
    G = glucose(sim, [meal(60), bolus(6)] + basal(), 100.0, 8)
    four_h = G[(4 * 60) // 5]
    assert G.max() < 200, f"peak {G.max():.0f} mg/dL"
    assert 80 <= four_h <= 140, f"at 4 h: {four_h:.0f} mg/dL"
    assert G.min() >= 60, f"dipped to {G.min():.0f} mg/dL"


def test_correction_bolus_bottoms_out_in_the_right_window(sim):
    """6 U on an empty stomach from 120 mg/dL: a hypo, and a timely one."""
    G = glucose(sim, [bolus(6, minutes=0)] + basal(), 120.0, 8)
    nadir = G.min()
    t_nadir = int(np.argmin(G)) * 5
    assert 40 <= nadir <= 70, f"nadir {nadir:.0f} mg/dL"
    assert 90 <= t_nadir <= 180, f"nadir at +{t_nadir} min"


# --------------------------------------------------------------------------- exercise

def test_aerobic_exercise_drops_glucose(sim):
    G = glucose(sim, [Event(type="exercise", t_start=T0, duration_min=45, intensity=0.6,
                            exercise_kind="aerobic")] + basal(), 140.0, 3)
    drop = 140.0 - G[45 // 5]
    assert 40 <= drop <= 80, f"dropped {drop:.0f} mg/dL over the session"


def test_anaerobic_exercise_spikes_then_falls(sim):
    """Catecholamines win early; the session's glucose uptake and the residual
    sensitisation win afterwards. 'Then falls' is read as after the session, since
    the adrenaline term is constant while the work continues."""
    G = glucose(sim, [Event(type="exercise", t_start=T0, duration_min=45, intensity=0.9,
                            exercise_kind="anaerobic")] + basal(), 140.0, 3)
    assert G[20 // 5] > 140.0, f"at 20 min: {G[20 // 5]:.0f} mg/dL, expected a rise"
    assert G[-1] < G[45 // 5] - 5.0, "glucose did not come back down after the session"


# --------------------------------------------------------------------------- alcohol

def _fed_day() -> list[Event]:
    """Breakfast, lunch and dinner, each bolused at the population carb ratio.

    The specification says 'with normal basal', which implies a normally fed patient:
    the delayed alcohol hypo is a story about the liver running out of glycogen, and a
    patient who has not eaten for 15 hours has none left to run out of.
    """
    out: list[Event] = []
    for minutes, carbs, units in ((30, 60, 6), (360, 70, 7), (720, 80, 8)):
        out += [meal(carbs, minutes), bolus(units, minutes)]
    return out


def test_alcohol_hypo_arrives_late(sim):
    drinks = [Event(type="alcohol", t_start=at(15 * 60), ethanol_g=56)]   # 4 drinks at 22:00
    with_alc = glucose(sim, _fed_day() + basal() + drinks, 120.0, 25)
    without = glucose(sim, _fed_day() + basal(), 120.0, 25)

    i_midnight = (17 * 60) // 5
    i_02, i_08 = (19 * 60) // 5, (25 * 60) // 5
    assert abs(with_alc[i_midnight] - without[i_midnight]) <= 20, (
        f"midnight {with_alc[i_midnight]:.0f} vs {without[i_midnight]:.0f} - too early")
    gap = without[i_02:i_08 + 1].min() - with_alc[i_02:i_08 + 1].min()
    assert gap >= 30, f"overnight nadir only {gap:.0f} mg/dL below the sober run"


# --------------------------------------------------------------------------- caffeine

def test_caffeine_raises_the_postprandial_peak(sim):
    caffeine = Event(type="caffeine", t_start=at(30), caffeine_mg=200)
    with_caf = glucose(sim, [meal(60), bolus(6), caffeine] + basal(), 100.0, 6)
    without = glucose(sim, [meal(60), bolus(6)] + basal(), 100.0, 6)
    delta = with_caf.max() - without.max()
    assert 10 <= delta <= 30, f"caffeine moved the peak by {delta:.0f} mg/dL"


# --------------------------------------------------------------------------- speed

def test_one_day_under_30ms(sim):
    """Person C runs this 200-500 times per request."""
    events = (
        [meal(c, h * 60) for h, c in ((0.5, 60), (6, 70), (12, 80))]
        + [bolus(u, h * 60) for h, u in ((0.5, 6), (6, 7), (12, 8))]
        + [Event(type="exercise", t_start=at(5 * 60), duration_min=45, intensity=0.6,
                 exercise_kind="aerobic"),
           Event(type="alcohol", t_start=at(15 * 60), ethanol_g=28),
           Event(type="caffeine", t_start=at(60), caffeine_mg=150)]
        + basal(24)
    )
    assert len(events) == 10
    s = state(120.0)
    sim.simulate(s, events, PARAMS, 1440, 5)            # warm the caches

    t = time.perf_counter()
    for _ in range(20):
        sim.simulate(s, events, PARAMS, 1440, 5)
    ms = (time.perf_counter() - t) / 20 * 1000
    assert ms < 30.0, f"{ms:.1f} ms per 24 h simulation"


# --------------------------------------------------------------------------- invariants

def test_state_stays_physical(sim):
    """Nothing in the state vector may go out of bounds or NaN, whatever the input."""
    events = _fed_day() + basal(24) + [
        Event(type="alcohol", t_start=at(15 * 60), ethanol_g=84),
        Event(type="caffeine", t_start=at(60), caffeine_mg=400),
        Event(type="water", t_start=at(200), water_ml=750),
        Event(type="exercise", t_start=at(300), duration_min=90, intensity=1.0,
              exercise_kind="mixed"),
        Event(type="illness", t_start=at(600), duration_min=600, severity=0.8),
    ]
    traj = sim.simulate(state(300.0), events, PARAMS, 1440, 5)
    cfg = sim.cfg
    assert np.all(np.isfinite(traj.fast))
    G = traj.fast[:, GI]
    assert G.min() >= cfg["G_clip_low"] and G.max() <= cfg["G_clip_high"]
    for name in ("I_p", "D1", "D2", "BAC", "caf", "ket", "ex"):
        assert traj.fast[:, FAST_IDX[name]].min() >= 0.0, f"{name} went negative"
    hyd = traj.fast[:, FAST_IDX["hyd"]]
    assert hyd.min() >= cfg["hyd_clip_low"] and hyd.max() <= cfg["hyd_clip_high"]
    assert 0.0 <= traj.fast[:, FAST_IDX["ex"]].max() <= 1.0
    assert traj.energy.min() >= 0.0 and traj.energy.max() <= 100.0


def test_trajectory_shape_matches_the_contract(sim):
    traj = sim.simulate(state(120.0), _fed_day() + basal(), PARAMS, 720, 5)
    assert traj.fast.shape == (720 // 5 + 1, len(FAST_IDX))
    assert len(traj.t) == traj.fast.shape[0]
    assert traj.t[0] == T0 and traj.t[-1] == T0 + timedelta(minutes=720)
    for key in ("tir_pct", "tbr70_pct", "tar180_pct", "min_G", "max_G", "t_min_G", "mean_G"):
        assert key in traj.summary


def test_basal_only_patient_holds_steady(sim):
    """A correctly basalled patient who eats nothing should not drift away.

    This is not in the specification's list, but everything else rests on it: if the
    resting set point wanders, every comparison against a baseline run is meaningless.
    """
    G = glucose(sim, basal(14), 100.0, 12)
    assert 85 <= G.min() and G.max() <= 115, f"drifted to [{G.min():.0f}, {G.max():.0f}]"
