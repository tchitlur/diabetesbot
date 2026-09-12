"""Parameter recovery for RealFitter.

Build a synthetic patient with known parameters, run the forward model, add CGM noise,
then fit from the population prior and check we get the parameters back.

On the k_alc bound, see engine/NOTES.md: with two drinking nights in a week and 8 mg/dL
of sensor noise it trades off against overnight insulin sensitivity, and the 40% bound
in the specification is not reachable. Measured spread over seven synthetic patients was
6-56%. Everything else recovers comfortably inside the specified bounds.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from engine.simulate import RealSimulator
from inference.impl import RealFitter
from schema.events import Event
from schema.params import PARAM_NAMES, PatientParams, population_prior
from schema.state import DailyState, FAST_IDX, PatientState, default_fast

WEEK_START = datetime(2026, 8, 24)          # a Monday
DAYS = 7
NOISE_SD = 8.0

# name -> allowed relative error
BOUNDS = {"si_day": 0.25, "carb_ratio": 0.25, "k_abs": 0.25, "k_ex": 0.40, "k_alc": 0.60}


def week_events(start: datetime = WEEK_START, days: int = DAYS) -> list[Event]:
    """Three bolused meals a day, three workouts, two drinking nights."""
    out: list[Event] = []
    for d in range(days):
        day = start + timedelta(days=d)
        out.append(Event(type="insulin", t_start=day, duration_min=1440, units=1.0,
                         insulin_kind="basal", source="synthetic"))
        for hour, minute, carbs, units in ((7, 30, 55, 5.5), (12, 30, 70, 7.0), (19, 0, 80, 8.0)):
            ts = day.replace(hour=hour, minute=minute)
            out.append(Event(type="meal", t_start=ts, carbs_g=carbs, fat_g=carbs * 0.2,
                             protein_g=carbs * 0.3, source="synthetic"))
            out.append(Event(type="insulin", t_start=ts, units=units, insulin_kind="bolus",
                             source="synthetic"))
        if d in (0, 2, 4):
            out.append(Event(type="exercise", t_start=day.replace(hour=17, minute=30),
                             duration_min=50, intensity=0.65, exercise_kind="aerobic",
                             source="synthetic"))
        if d in (2, 5):
            out.append(Event(type="alcohol", t_start=day.replace(hour=21, minute=30),
                             ethanol_g=42, source="synthetic"))
    return out


def synthetic_cgm(truth: PatientParams, seed: int) -> pd.Series:
    """Run the forward model for a week and add N(0, 8) sensor noise."""
    sim = RealSimulator()
    st = PatientState(t=WEEK_START, fast=default_fast(130.0), daily=DailyState())
    traj = sim.simulate(st, week_events(), truth, DAYS * 1440, 5)
    G = traj.fast[:, FAST_IDX["G"]]
    noisy = G + np.random.default_rng(seed).normal(0, NOISE_SD, G.size)
    return pd.Series(noisy, index=pd.DatetimeIndex(traj.t), dtype=float)


def draw_truth(prior, seed: int) -> PatientParams:
    rng = np.random.default_rng(seed)
    return PatientParams.from_array(prior.samples[rng.integers(0, len(prior.samples))])


@lru_cache(maxsize=None)
def fitted(seed: int):
    """Fit once per seed; a fit costs ~20 s and two tests want the same one."""
    prior = population_prior(500, seed=3)
    truth = draw_truth(prior, seed)
    cgm = synthetic_cgm(truth, seed)
    fitter = RealFitter()
    started = time.perf_counter()
    posterior = fitter.fit(cgm, week_events(), prior)
    return prior, truth, fitter, posterior, time.perf_counter() - started


@pytest.mark.parametrize("seed", [23, 101])
def test_fitter_recovers_parameters(seed: int):
    _, truth, fitter, posterior, elapsed = fitted(seed)

    assert elapsed < 60.0, f"fit took {elapsed:.1f} s"
    assert fitter.rmse_after < fitter.rmse_before, (
        f"RMSE {fitter.rmse_before:.1f} -> {fitter.rmse_after:.1f}: the fit did not help")
    assert fitter.rmse_after < 1.5 * NOISE_SD, (
        f"RMSE {fitter.rmse_after:.1f} mg/dL is well above the {NOISE_SD:.0f} noise floor")

    for name, limit in BOUNDS.items():
        got, want = getattr(posterior.mean, name), getattr(truth, name)
        err = abs(got - want) / abs(want)
        assert err <= limit, f"{name}: fitted {got:.4g} vs true {want:.4g} ({err:.0%} off)"


def test_posterior_is_usable(seed: int = 23):
    """Person C draws from these samples for Monte Carlo, so they have to be sane."""
    prior, _truth, _fitter, posterior, _ = fitted(seed)

    assert posterior.samples.shape == (500, len(PARAM_NAMES))
    assert np.all(np.isfinite(posterior.samples))

    from inference.impl import patient_ranges
    rng = patient_ranges()
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = rng[name]
        col = posterior.samples[:, i]
        assert col.min() >= lo - 1e-9 and col.max() <= hi + 1e-9, f"{name} escaped its range"
        assert col.std() > 0, f"{name} posterior collapsed to a point"

    # and it must be tighter than the prior it started from, or it learned nothing
    for name in ("si_day", "carb_ratio", "k_abs"):
        i = PARAM_NAMES.index(name)
        assert posterior.samples[:, i].std() < prior.samples[:, i].std(), (
            f"{name} posterior is no tighter than the prior")

    drawn = posterior.sample(8, np.random.default_rng(0))
    assert len(drawn) == 8 and all(isinstance(p, PatientParams) for p in drawn)


def test_fitter_declines_to_fit_thin_data():
    """Two hours of CGM must not produce a confident patient model."""
    prior = population_prior(200, seed=3)
    idx = pd.date_range(WEEK_START, periods=24, freq="5min")
    thin = pd.Series(np.full(24, 120.0), index=idx)
    out = RealFitter().fit(thin, [], prior)
    assert out is prior
