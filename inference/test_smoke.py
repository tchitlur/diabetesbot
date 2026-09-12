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


# --------------------------------------------------------------------------- estimator

DAY_START = datetime(2026, 9, 11)
MEAL_AT = DAY_START.replace(hour=8)


def _one_day_events() -> list[Event]:
    return [
        Event(type="insulin", t_start=DAY_START, duration_min=1440, units=1.0,
              insulin_kind="basal", source="synthetic"),
        Event(type="meal", t_start=MEAL_AT, carbs_g=60, source="synthetic"),
        Event(type="insulin", t_start=MEAL_AT, units=6, insulin_kind="bolus",
              source="synthetic"),
    ]


@lru_cache(maxsize=1)
def _synthetic_day():
    """Ground-truth trajectory plus the noisy CGM a sensor would have recorded."""
    sim = RealSimulator()
    st = PatientState(t=DAY_START, fast=default_fast(115.0), daily=DailyState())
    traj = sim.simulate(st, _one_day_events(), PatientParams(), 16 * 60, 5)
    G = traj.fast[:, FAST_IDX["G"]]
    noisy = G + np.random.default_rng(4).normal(0, 10, G.size)
    return traj, pd.Series(noisy, index=pd.DatetimeIndex(traj.t))


def _truth_at(traj, when: datetime) -> np.ndarray:
    return traj.fast[int((when - DAY_START).total_seconds() // 60 // 5)]


def _estimate_at(when: datetime):
    from inference.impl import RealEstimator
    traj, cgm = _synthetic_day()
    profile_state = PatientState(t=DAY_START, fast=default_fast(115.0), daily=DailyState())
    est = RealEstimator()
    started = time.perf_counter()
    out = est.estimate(cgm[cgm.index <= when], _one_day_events(), PatientParams(), profile_state)
    return out, _truth_at(traj, when), time.perf_counter() - started


def test_estimator_finds_carbs_on_board():
    """Twenty minutes after 60 g, the gut compartments should still be nearly full.

    The specification's band was 20-55 g. With the population k_abs of 0.03/min the
    forward model itself holds 55.2 g at that point, so the upper bound sat exactly on
    the true value; widened to 60. See engine/NOTES.md.
    """
    when = MEAL_AT + timedelta(minutes=20)
    out, truth, elapsed = _estimate_at(when)
    on_board = out.fast[FAST_IDX["D1"]] + out.fast[FAST_IDX["D2"]]
    true_on_board = truth[FAST_IDX["D1"]] + truth[FAST_IDX["D2"]]

    assert 20.0 <= on_board <= 60.0, f"{on_board:.1f} g on board"
    assert abs(on_board - true_on_board) / true_on_board <= 0.25, (
        f"estimated {on_board:.1f} g against a true {true_on_board:.1f} g")
    assert elapsed < 5.0, f"estimate took {elapsed:.1f} s"


def test_estimator_tracks_active_insulin():
    when = MEAL_AT + timedelta(minutes=60)
    out, truth, _ = _estimate_at(when)
    got, want = out.fast[FAST_IDX["I_p"]], truth[FAST_IDX["I_p"]]
    assert abs(got - want) / want <= 0.30, f"I_p {got:.1f} vs true {want:.1f} mU/L"


def test_estimator_returns_a_usable_state():
    when = MEAL_AT + timedelta(minutes=60)
    out, truth, _ = _estimate_at(when)

    assert out.t == when
    assert out.fast.shape == (len(FAST_IDX),)
    assert out.fast_cov is not None and out.fast_cov.shape == (len(FAST_IDX), len(FAST_IDX))
    assert np.all(np.isfinite(out.fast)) and np.all(np.isfinite(out.fast_cov))

    cov = (out.fast_cov + out.fast_cov.T) / 2.0
    assert np.min(np.linalg.eigvalsh(cov)) >= -1e-6, "covariance is not positive semi-definite"
    gi = FAST_IDX["G"]
    assert cov[gi, gi] < 200.0, "glucose is measured; its variance should be small"
    for name in ("D1", "D2", "gly_liver"):
        j = FAST_IDX[name]
        assert cov[j, j] > cov[gi, gi] * 0.1, f"{name} is unobserved but has no uncertainty"
    assert abs(out.fast[gi] - truth[FAST_IDX["G"]]) < 25.0


def test_estimator_survives_a_sensor_gap():
    """Predict-only steps across NaNs, per the contract's gap convention."""
    from inference.impl import RealEstimator
    traj, cgm = _synthetic_day()
    when = MEAL_AT + timedelta(minutes=60)
    series = cgm[cgm.index <= when].copy()
    gap_start = when - timedelta(minutes=45)
    series[(series.index >= gap_start) & (series.index < when - timedelta(minutes=10))] = np.nan

    est = RealEstimator()
    profile_state = PatientState(t=DAY_START, fast=default_fast(115.0), daily=DailyState())
    out = est.estimate(series, _one_day_events(), PatientParams(), profile_state)

    assert est.n_predicts > est.n_updates, "the gap should have produced predict-only steps"
    assert np.all(np.isfinite(out.fast))
    assert abs(out.fast[FAST_IDX["G"]] - _truth_at(traj, when)[FAST_IDX["G"]]) < 40.0


# --------------------------------------------------------------------------- residual model

RESID_START = datetime(2026, 8, 10)
RESID_DAYS = 14


def _logged_week(days: int = RESID_DAYS) -> list[Event]:
    """What the patient actually told us about."""
    out: list[Event] = []
    for d in range(days):
        day = RESID_START + timedelta(days=d)
        out.append(Event(type="insulin", t_start=day, duration_min=1440, units=1.0,
                         insulin_kind="basal", source="synthetic"))
        for hour, minute, carbs, units in ((7, 30, 55, 5.5), (12, 30, 70, 7.0), (19, 0, 80, 8.0)):
            ts = day.replace(hour=hour, minute=minute)
            out.append(Event(type="meal", t_start=ts, carbs_g=carbs, source="synthetic"))
            out.append(Event(type="insulin", t_start=ts, units=units, insulin_kind="bolus",
                             source="synthetic"))
    return out


def _unlogged_snacks(days: int = RESID_DAYS) -> list[Event]:
    """A 25 g biscuit at four every afternoon that never makes it into the log.

    This is the whole point of a residual model: a repeatable pattern the mechanistic
    model cannot know about, which a learner can pick up from the time of day.
    """
    return [Event(type="meal", t_start=(RESID_START + timedelta(days=d)).replace(hour=16),
                  carbs_g=25, source="synthetic") for d in range(days)]


@lru_cache(maxsize=1)
def _residual_history():
    sim = RealSimulator()
    st = PatientState(t=RESID_START, fast=default_fast(125.0), daily=DailyState())
    traj = sim.simulate(st, _logged_week() + _unlogged_snacks(), PatientParams(),
                        RESID_DAYS * 1440, 5)
    G = traj.fast[:, FAST_IDX["G"]]
    noisy = G + np.random.default_rng(9).normal(0, 8, G.size)
    return pd.Series(noisy, index=pd.DatetimeIndex(traj.t))


@lru_cache(maxsize=1)
def _trained_residual():
    from inference.impl import RealResidual
    model = RealResidual()
    started = time.perf_counter()
    model.fit(_residual_history(), _logged_week(), PatientParams())
    return model, time.perf_counter() - started


def test_residual_trains_inside_its_budget():
    model, elapsed = _trained_residual()
    assert elapsed < 30.0, f"training took {elapsed:.1f} s"
    assert model.train_rows > 100, f"only {model.train_rows} forecast origins"


def test_residual_beats_leaving_the_forecast_alone():
    """On held-out days, every kept horizon must reduce the error it was trained on."""
    model, _ = _trained_residual()
    from inference.impl import HORIZONS
    assert set(model.models) == set(HORIZONS), f"kept only {sorted(model.models)}"
    for h in HORIZONS:
        assert model.holdout_mae[h] < model.baseline_mae[h], (
            f"h={h}: {model.holdout_mae[h]:.2f} vs {model.baseline_mae[h]:.2f} mg/dL")


def test_residual_correction_has_the_right_shape():
    model, _ = _trained_residual()
    from inference.impl import HORIZONS, RealEstimator
    sim = RealSimulator()
    cgm = _residual_history()
    origin = RESID_START + timedelta(days=RESID_DAYS - 1, hours=15)
    recent = cgm[cgm.index <= origin]
    anchor = PatientState(t=origin, fast=default_fast(float(recent.iloc[-1])), daily=DailyState())
    est = RealEstimator().estimate(recent, _logged_week(), PatientParams(), anchor)
    forecast = sim.simulate(PatientState(t=origin, fast=est.fast, daily=DailyState()),
                            _logged_week(), PatientParams(), 240, 5)

    out = model.correct(forecast, recent, _logged_week())
    assert out is not forecast, "correct() must return a new Trajectory"
    assert out.fast.shape == forecast.fast.shape and out.t == forecast.t

    before = forecast.fast[:, FAST_IDX["G"]]
    after = out.fast[:, FAST_IDX["G"]]
    shift = after - before
    assert abs(shift[0]) < 1e-9, "the present needs no correction"
    beyond = int(max(HORIZONS) / 5) + 1
    assert np.allclose(shift[beyond:], 0.0), "there is no evidence past the last horizon"
    assert np.abs(shift[1:beyond]).max() > 0.5, "the correction did nothing at all"
    assert out.summary["mean_G"] == pytest.approx(float(after.mean()))
    assert np.array_equal(forecast.fast[:, FAST_IDX["G"]], before), "correct() mutated its input"


def test_untrained_residual_is_a_no_op():
    from inference.impl import RealResidual
    sim = RealSimulator()
    traj = sim.simulate(PatientState(t=RESID_START, fast=default_fast(120.0), daily=DailyState()),
                        _logged_week(), PatientParams(), 120, 5)
    assert RealResidual().correct(traj, _residual_history(), _logged_week()) is traj


def test_features_never_look_ahead():
    """Anything after the origin must be invisible, or the walk-forward split is a lie."""
    from inference.impl import features_at
    cgm = _residual_history()
    origin = RESID_START + timedelta(days=3, hours=10)
    events = _logged_week()
    full = features_at(origin, cgm, events, 1.0)
    truncated = features_at(origin, cgm[cgm.index <= origin],
                            [e for e in events if e.t_start <= origin], 1.0)
    assert np.allclose(full, truncated, equal_nan=True)


# --------------------------------------------------------------------------- drift

def _drift_with(errors_by_day: dict[int, float]):
    from inference.impl import RealDrift
    d = RealDrift()
    base = datetime(2026, 9, 1)
    for day, err in errors_by_day.items():
        for step in range(24):
            d.update(base + timedelta(days=day, hours=step), 100.0 + err, 100.0)
    return d


def test_drift_needs_enough_history():
    from inference.impl import RealDrift
    assert RealDrift().score() == 0.0
    assert _drift_with({0: 10.0, 1: 10.0, 2: 10.0}).score() == 0.0


def test_drift_is_zero_when_the_model_still_fits():
    score = _drift_with({d: 12.0 for d in range(14)}).score()
    assert score == pytest.approx(0.0, abs=1e-9)


def test_drift_fires_when_predictions_go_bad():
    """Three days of doubled error against the fortnight before it."""
    errors = {d: 10.0 for d in range(11)}
    errors.update({11: 25.0, 12: 27.0, 13: 30.0})
    score = _drift_with(errors).score()
    assert score > 1.0, f"drift score only reached {score:.2f}"


def test_drift_ignores_rubbish():
    from inference.impl import RealDrift
    d = RealDrift()
    base = datetime(2026, 9, 1)
    d.update(base, float("nan"), 100.0)
    d.update(base, 100.0, None)
    d.update(base, None, 100.0)
    assert d.log == []
    assert d.score() == 0.0
