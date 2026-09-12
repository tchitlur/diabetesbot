"""Real inference: fitting, state estimation, residual correction, drift monitoring.

schema.inference.get_inference() picks these up automatically once all four import
cleanly. Nothing here changes a signature; everything satisfies the Protocols in
schema/inference.py.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from engine.simulate import RealSimulator, load_params
from schema.events import Event, Profile
from schema.params import PARAM_NAMES, ParamPosterior, PatientParams
from schema.simulate import Trajectory
from schema.state import FAST_IDX, N_FAST, DailyState, PatientState, default_fast

log = logging.getLogger(__name__)

_G = FAST_IDX["G"]

# Parameters fitted in log space. The rest (bounded or signed) are fitted linearly.
LOG_PARAMS = ("si_night", "si_day", "si_evening", "carb_ratio", "k_abs", "k_ex", "k_alc")
LINEAR_PARAMS = ("basal_drift", "k_sleep")

MAX_FIT_DAYS = 21
FIT_DT_MIN = 0           # 0 = choose from how much history there is (see _choose_dt)
FIT_DT_FINE = 5          # the native CGM grid
FIT_DT_COARSE = 15       # "subsample the grid to 15 minutes if needed", per spec
FIT_DT_SWITCH_DAYS = 10  # above this much history, coarsen to stay inside the budget
N_POSTERIOR = 500


def patient_ranges() -> dict[str, list[float]]:
    return load_params()["patient_ranges"]


# --------------------------------------------------------------------------- transforms

def to_x(p: PatientParams) -> np.ndarray:
    """PatientParams -> unconstrained optimiser vector."""
    out = np.empty(len(PARAM_NAMES))
    for i, name in enumerate(PARAM_NAMES):
        v = getattr(p, name)
        out[i] = math.log(max(v, 1e-6)) if name in LOG_PARAMS else v
    return out


def from_x(x: np.ndarray) -> PatientParams:
    """Optimiser vector -> PatientParams."""
    kw = {}
    for i, name in enumerate(PARAM_NAMES):
        kw[name] = float(np.exp(x[i])) if name in LOG_PARAMS else float(x[i])
    return PatientParams(**kw)


def _prior_moments(prior: ParamPosterior) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of the prior in the transformed space."""
    s = np.asarray(prior.samples, dtype=float)
    xs = np.empty_like(s)
    for i, name in enumerate(PARAM_NAMES):
        col = s[:, i]
        xs[:, i] = np.log(np.clip(col, 1e-6, None)) if name in LOG_PARAMS else col
    sd = xs.std(axis=0)
    sd[sd < 1e-6] = 1e-6
    return xs.mean(axis=0), sd


def _choose_dt(n_days: int) -> int:
    """Five-minute resolution while the history is short enough to afford it."""
    return FIT_DT_FINE if n_days <= FIT_DT_SWITCH_DAYS else FIT_DT_COARSE


def _autocorr_inflation(residual: np.ndarray, cap: float = 60.0) -> float:
    """(1 + rho) / (1 - rho): the effective-sample-size correction for AR(1) errors."""
    r = np.asarray(residual, dtype=float)
    if r.size < 10:
        return 1.0
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom <= 0.0:
        return 1.0
    rho = float(np.dot(r[:-1], r[1:]) / denom)
    rho = min(max(rho, 0.0), 0.99)
    return float(min((1.0 + rho) / (1.0 - rho), cap))


def _clip_to_ranges(arr: np.ndarray) -> np.ndarray:
    rng = patient_ranges()
    out = np.array(arr, dtype=float, copy=True)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = rng[name]
        np.clip(out[..., i], lo, hi, out=out[..., i])
    return out


# --------------------------------------------------------------------------- day slicing

def _split_days(glucose: pd.Series, events: list[Event]) -> list[tuple]:
    """Split observations and events into (day_start, observed_series, day_events)."""
    if glucose is None or len(glucose) == 0:
        return []
    idx = pd.DatetimeIndex(glucose.index)
    days = sorted({d.date() for d in idx})[-MAX_FIT_DAYS:]
    evs = sorted(events, key=lambda e: e.t_start)
    out = []
    for d in days:
        start = datetime(d.year, d.month, d.day)
        end = start + timedelta(days=1)
        obs = glucose[(idx >= start) & (idx < end)].dropna()
        if len(obs) < 2:
            continue
        # An event that is still acting at midnight has to come along, so reach back
        # far enough to cover a basal block or a long workout from the previous day.
        day_events = [e for e in evs
                      if start - timedelta(hours=8) <= e.t_start < end]
        out.append((start, obs, day_events))
    return out


class _DayFit:
    """Pre-sliced observations, so the residual loop does no pandas work."""

    __slots__ = ("start", "events", "offsets", "observed", "first")

    def __init__(self, start: datetime, obs: pd.Series, events: list[Event], dt_min: int):
        rel = np.array([(t - start).total_seconds() / 60.0 for t in obs.index])
        keep = (rel >= 0) & (rel <= 1440) & (np.abs(rel % dt_min) < 1e-6)
        if not keep.any():                       # grid does not line up; take the nearest
            keep = (rel >= 0) & (rel <= 1440)
        self.start = start
        self.events = events
        self.offsets = np.round(rel[keep] / dt_min).astype(int)
        self.observed = obs.to_numpy(dtype=float)[keep]
        self.first = float(self.observed[0]) if self.observed.size else float("nan")


# --------------------------------------------------------------------------- fitter

class RealFitter:
    """Maximum a posteriori fit of PatientParams, with a Laplace posterior.

    Satisfies schema.inference.Fitter.
    """

    def __init__(self, simulator: RealSimulator | None = None, profile: Profile | None = None,
                 dt_min: int = FIT_DT_MIN, max_nfev: int = 40, time_budget_s: float = 45.0):
        self.sim = simulator or RealSimulator(profile=profile)
        if profile is not None:
            self.sim.profile = profile
        self.dt_min = int(dt_min)
        self.max_nfev = int(max_nfev)
        self.time_budget_s = float(time_budget_s)
        self.rmse_before: float | None = None
        self.rmse_after: float | None = None
        self.n_points: int = 0

    # -- forward model ---------------------------------------------------
    def _simulate_days(self, days: list[_DayFit], params: PatientParams,
                       daily: DailyState, dt_min: int) -> np.ndarray:
        """Simulated glucose at every observed point, days chained through the state."""
        carried: np.ndarray | None = None
        out: list[np.ndarray] = []
        for day in days:
            y = default_fast(day.first) if carried is None else carried.copy()
            # Per spec: glucose is re-anchored to the day's first reading, the rest of
            # the state carries over from yesterday.
            y[_G] = day.first
            st = PatientState(t=day.start, fast=y, daily=daily)
            traj = self.sim.simulate(st, day.events, params, 1440, dt_min)
            g = traj.fast[:, _G]
            idx = np.clip(day.offsets, 0, g.size - 1)
            out.append(g[idx])
            carried = traj.fast[-1]
        return np.concatenate(out) if out else np.zeros(0)

    def fit(self, glucose: pd.Series, events: list[Event], prior: ParamPosterior) -> ParamPosterior:
        started = time.perf_counter()
        sliced = _split_days(glucose, events)
        dt = self.dt_min or _choose_dt(len(sliced))
        days = [_DayFit(s, o, e, dt) for s, o, e in sliced]
        days = [d for d in days if d.observed.size >= 2]
        observed = np.concatenate([d.observed for d in days]) if days else np.zeros(0)
        self.n_points = int(observed.size)

        if observed.size < 4 * len(PARAM_NAMES):
            log.info("RealFitter: only %d usable points, returning the prior unchanged",
                     observed.size)
            return prior

        daily = DailyState()
        mu, sd = _prior_moments(prior)
        x0 = to_x(prior.mean)

        def residual(x: np.ndarray) -> np.ndarray:
            sim_g = self._simulate_days(days, from_x(x), daily, dt)
            if sim_g.size != observed.size:      # a day failed to integrate
                return np.full(observed.size + len(PARAM_NAMES), 1e3)
            return np.concatenate([sim_g - observed, (x - mu) / sd])

        t_eval = time.perf_counter()
        r0 = residual(x0)
        per_eval = max(time.perf_counter() - t_eval, 1e-3)
        self.rmse_before = float(np.sqrt(np.mean(r0[:observed.size] ** 2)))

        lo, hi = self._bounds()
        # Each trust-region iteration costs one residual plus a finite-difference
        # Jacobian, so len(PARAM_NAMES) + 1 evaluations. Size the iteration cap from
        # what actually fits in the budget rather than hoping a fixed number does.
        remaining = self.time_budget_s - (time.perf_counter() - started)
        affordable = int(remaining / (per_eval * (len(PARAM_NAMES) + 1)))
        budget = int(np.clip(affordable, 8, self.max_nfev))
        try:
            res = least_squares(residual, np.clip(x0, lo + 1e-9, hi - 1e-9),
                                bounds=(lo, hi), method="trf", x_scale="jac",
                                max_nfev=budget, ftol=1e-6, xtol=1e-8)
        except Exception:                        # pragma: no cover - solver blow-up
            log.exception("RealFitter: least_squares failed, returning the prior")
            return prior

        r1 = residual(res.x)
        self.rmse_after = float(np.sqrt(np.mean(r1[:observed.size] ** 2)))
        elapsed = time.perf_counter() - started
        log.info("RealFitter: %d points over %d days at dt=%d min, %d iterations, "
                 "RMSE %.1f -> %.1f mg/dL in %.1f s",
                 observed.size, len(days), dt, budget, self.rmse_before, self.rmse_after,
                 elapsed)
        if self.rmse_after > self.rmse_before:   # the optimiser made things worse
            log.warning("RealFitter: fit did not improve on the prior, keeping the prior")
            return prior

        mean = from_x(res.x)
        samples = self._laplace_samples(res, r1[:observed.size])
        return ParamPosterior(mean=mean, samples=samples)

    # -- posterior --------------------------------------------------------
    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        rng = patient_ranges()
        lo = np.empty(len(PARAM_NAMES))
        hi = np.empty(len(PARAM_NAMES))
        for i, name in enumerate(PARAM_NAMES):
            a, b = rng[name]
            if name in LOG_PARAMS:
                lo[i], hi[i] = math.log(max(a, 1e-6)), math.log(b)
            else:
                lo[i], hi[i] = a, b
        return lo, hi

    def _laplace_samples(self, res, data_residual: np.ndarray,
                         n: int = N_POSTERIOR) -> np.ndarray:
        """Gaussian around the MAP estimate with the inverse Hessian as covariance.

        Scaled up by the residual autocorrelation. The textbook estimator assumes
        independent residuals; CGM residuals at five-minute spacing are nothing of
        the sort (lag-1 rho is typically above 0.9), so treating every point as
        independent understates the covariance by more than an order of magnitude and
        hands Person C a posterior far too confident to do Monte Carlo with.
        """
        rng = np.random.default_rng(0)
        k = len(PARAM_NAMES)
        J = np.asarray(res.jac, dtype=float)
        n_obs = int(data_residual.size)
        dof = max(n_obs - k, 1)
        s2 = 2.0 * float(res.cost) / dof
        s2 *= _autocorr_inflation(data_residual)
        try:
            JTJ = J.T @ J + np.eye(k) * 1e-9
            cov = np.linalg.inv(JTJ) * s2
            cov = (cov + cov.T) / 2.0
            # A Laplace covariance from a bounded fit is not always positive definite.
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-12, None)
            L = vecs @ np.diag(np.sqrt(vals))
        except np.linalg.LinAlgError:            # pragma: no cover
            L = np.eye(k) * 1e-3
        draws = res.x[None, :] + rng.standard_normal((n, k)) @ L.T
        out = np.empty((n, k))
        for i, name in enumerate(PARAM_NAMES):
            out[:, i] = np.exp(draws[:, i]) if name in LOG_PARAMS else draws[:, i]
        return _clip_to_ranges(out)


# --------------------------------------------------------------------------- estimator

STATE_BOUNDS = {
    "I_p": (0.0, None), "D1": (0.0, None), "D2": (0.0, None),
    "BAC": (0.0, None), "caf": (0.0, None), "ket": (0.0, None), "ex": (0.0, 1.0),
}


def _robust_sqrt(P: np.ndarray) -> np.ndarray:
    """Matrix square root that cannot fail.

    The sigma-point factorisation is the fragile part of a UKF on a bounded,
    clipped state: one non-positive-definite covariance and Cholesky raises. An
    eigendecomposition with the spectrum floored at zero always returns something
    usable, at the cost of being a little slower than Cholesky.
    """
    P = (np.asarray(P, dtype=float) + np.asarray(P, dtype=float).T) / 2.0
    try:
        return np.linalg.cholesky(P).T
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(P)
        vals = np.clip(vals, 0.0, None)
        return (vecs @ np.diag(np.sqrt(vals))).T


class RealEstimator:
    """Unscented Kalman filter over the fast state, with glucose as the only measurement.

    Satisfies schema.inference.Estimator. Everything except G is hidden: the filter's
    real job is to say how much carbohydrate is still in the gut, how much insulin is
    still active, and how much alcohol is still on board.
    """

    def __init__(self, simulator: RealSimulator | None = None, profile: Profile | None = None,
                 cfg: dict | None = None):
        self.sim = simulator or RealSimulator(profile=profile)
        if profile is not None:
            self.sim.profile = profile
        self.cfg = cfg or self.sim.cfg
        self.n_updates = 0
        self.n_predicts = 0

    def _process_noise(self) -> np.ndarray:
        from schema.state import FAST_NAMES
        return np.diag([float(self.cfg[f"q_{n}"]) for n in FAST_NAMES])

    def estimate(self, glucose: pd.Series, events: list[Event], params: PatientParams,
                 profile_state: PatientState) -> PatientState:
        from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

        cfg = self.cfg
        dt = 5.0
        base = np.asarray(profile_state.fast, dtype=float).copy()

        if glucose is None or len(glucose) == 0:
            return replace_state(profile_state, base, np.diag(np.diag(self._process_noise())))

        idx = pd.DatetimeIndex(glucose.index)
        t_end = idx[-1]
        t_start = t_end - timedelta(hours=float(cfg["ukf_window_h"]))
        window = glucose[idx >= t_start]
        if len(window) == 0:
            window = glucose.iloc[-1:]
        w_idx = pd.DatetimeIndex(window.index)
        t0 = w_idx[0].to_pydatetime()
        horizon = max((t_end - w_idx[0]).total_seconds() / 60.0, dt)

        first = window.dropna()
        base[_G] = float(first.iloc[0]) if len(first) else base[_G]

        step = self.sim.make_stepper(t0, events, params, profile_state.daily, horizon + dt)
        clock = {"t": 0.0}

        def fx(x, dt_min):
            return step(x, clock["t"], dt_min)

        def hx(x):
            return np.array([x[_G]])

        Q = self._process_noise()
        pts = MerweScaledSigmaPoints(n=N_FAST, alpha=float(cfg["ukf_alpha"]),
                                     beta=float(cfg["ukf_beta"]), kappa=float(cfg["ukf_kappa"]),
                                     sqrt_method=_robust_sqrt)
        ukf = UnscentedKalmanFilter(dim_x=N_FAST, dim_z=1, dt=dt, fx=fx, hx=hx, points=pts)
        ukf.x = base
        ukf.P = Q * float(cfg["ukf_p0_mult"])
        ukf.Q = Q
        ukf.R = np.array([[float(cfg["ukf_meas_sd"]) ** 2]])

        jitter = float(cfg["ukf_jitter"])
        n_steps = int(round(horizon / dt))
        obs = window.reindex(pd.date_range(w_idx[0], periods=n_steps + 1, freq=f"{int(dt)}min"))
        self.n_updates = self.n_predicts = 0

        for k in range(1, n_steps + 1):
            clock["t"] = (k - 1) * dt
            try:
                ukf.predict()
                self.n_predicts += 1
                z = obs.iloc[k] if k < len(obs) else np.nan
                if z is not None and np.isfinite(z):
                    ukf.update(np.array([float(z)]))
                    self.n_updates += 1
            except Exception:                       # pragma: no cover - filter blow-up
                log.exception("RealEstimator: UKF step %d failed, holding the last state", k)
                break
            ukf.x = self._clip(ukf.x)
            ukf.P = (ukf.P + ukf.P.T) / 2.0 + np.eye(N_FAST) * jitter

        log.info("RealEstimator: %d predicts, %d updates over %.1f h",
                 self.n_predicts, self.n_updates, horizon / 60.0)
        return replace_state(profile_state, self._clip(ukf.x), np.asarray(ukf.P, dtype=float),
                             t=t_end.to_pydatetime())

    def _clip(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        y = np.asarray(x, dtype=float).copy()
        y[_G] = min(max(y[_G], cfg["G_clip_low"]), cfg["G_clip_high"])
        y[FAST_IDX["hyd"]] = min(max(y[FAST_IDX["hyd"]], cfg["hyd_clip_low"]), cfg["hyd_clip_high"])
        y[FAST_IDX["gly_liver"]] = min(max(y[FAST_IDX["gly_liver"]], 0.0), cfg["gly_liver_full"])
        y[FAST_IDX["gly_muscle"]] = min(max(y[FAST_IDX["gly_muscle"]], 0.0), cfg["gly_muscle_full"])
        for name, (lo, hi) in STATE_BOUNDS.items():
            j = FAST_IDX[name]
            if lo is not None and y[j] < lo:
                y[j] = lo
            if hi is not None and y[j] > hi:
                y[j] = hi
        return y


def replace_state(profile_state: PatientState, fast: np.ndarray, cov: np.ndarray,
                  t: datetime | None = None) -> PatientState:
    return PatientState(t=t or profile_state.t, fast=np.asarray(fast, dtype=float),
                        daily=replace(profile_state.daily), slow=replace(profile_state.slow),
                        fast_cov=np.asarray(cov, dtype=float))


# --------------------------------------------------------------------------- placeholders
# Replaced in Deliverable 5. Until then get_inference() returns a real fitter and a real
# estimator alongside these, which is strictly better than falling back across the board.

from inference.naive import NaiveDrift, NaiveResidual  # noqa: E402


class RealResidual(NaiveResidual):
    """Deliverable 5 replaces this with the LightGBM residual model."""


class RealDrift(NaiveDrift):
    """Deliverable 5 replaces this with the rolling prediction-error monitor."""
