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


# --------------------------------------------------------------------------- residual model

HORIZONS = (30, 60, 120)          # minutes ahead that get their own regressor
ORIGIN_STRIDE_MIN = 30            # how often to take a forecast origin when training
FEATURE_NAMES = (
    "hour_sin", "hour_cos", "dow", "slope_30", "G_now",
    "carbs_3h", "fat_3h", "protein_3h", "insulin_4h", "ethanol_8h", "caffeine_6h",
    "exercise_min_12h", "sleep_h", "si_mult",
)


# LightGBM's native training API, not its sklearn wrapper: lightgbm.sklearn needs
# scikit-learn, which is not in requirements.txt, and requirements.txt is not ours to
# edit. See engine/NOTES.md.
LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "verbose": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
}
LGB_ROUNDS = 200


def _window_sum(events: list[Event], t: datetime, hours: float, field: str,
                types: tuple[str, ...]) -> float:
    lo = t - timedelta(hours=hours)
    return float(sum(getattr(e, field) for e in events
                     if e.type in types and lo <= e.t_start <= t))


def _sleep_hours(events: list[Event], t: datetime) -> float:
    """Hours slept in the 24 hours up to the forecast origin."""
    lo = t - timedelta(hours=24)
    return float(sum(e.duration_min for e in events
                     if e.type == "sleep" and lo <= e.t_start <= t) / 60.0)


def features_at(t: datetime, glucose: pd.Series, events: list[Event],
                si_mult: float) -> np.ndarray:
    """One feature row. Reads only data at or before t, so training cannot leak."""
    past = glucose[glucose.index <= t].dropna()
    G_now = float(past.iloc[-1]) if len(past) else float("nan")
    slope = 0.0
    if len(past) >= 2:
        window = past[past.index >= t - timedelta(minutes=30)]
        if len(window) >= 2:
            span = (window.index[-1] - window.index[0]).total_seconds() / 60.0
            if span > 0:
                slope = float((window.iloc[-1] - window.iloc[0]) / span)
    hour = t.hour + t.minute / 60.0
    return np.array([
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        float(t.weekday()),
        slope,
        G_now,
        _window_sum(events, t, 3, "carbs_g", ("meal", "alcohol")),
        _window_sum(events, t, 3, "fat_g", ("meal",)),
        _window_sum(events, t, 3, "protein_g", ("meal",)),
        _window_sum(events, t, 4, "units", ("insulin",)),
        _window_sum(events, t, 8, "ethanol_g", ("alcohol",)),
        _window_sum(events, t, 6, "caffeine_mg", ("caffeine",)),
        _window_sum(events, t, 12, "duration_min", ("exercise",)),
        _sleep_hours(events, t),
        float(si_mult),
    ], dtype=float)


class RealResidual:
    """Gradient-boosted correction on top of the mechanistic forecast.

    Satisfies schema.inference.Residual. One regressor per horizon; correct() applies
    them at 30, 60 and 120 minutes and interpolates linearly in between, tapering to
    zero beyond the last horizon because there is no evidence out there.
    """

    def __init__(self, simulator: RealSimulator | None = None, profile: Profile | None = None,
                 time_budget_s: float = 25.0):
        self.sim = simulator or RealSimulator(profile=profile)
        if profile is not None:
            self.sim.profile = profile
        self.time_budget_s = float(time_budget_s)
        self.models: dict[int, object] = {}
        self.train_rows = 0
        self.train_mae: dict[int, float] = {}
        self.holdout_mae: dict[int, float] = {}
        self.baseline_mae: dict[int, float] = {}

    # -- training ----------------------------------------------------------
    def _origins(self, glucose: pd.Series) -> list[datetime]:
        idx = pd.DatetimeIndex(glucose.dropna().index)
        if len(idx) == 0:
            return []
        stride = timedelta(minutes=ORIGIN_STRIDE_MIN)
        last_usable = idx[-1] - timedelta(minutes=max(HORIZONS))
        out = []
        cursor = idx[0] + timedelta(hours=12)        # leave room for the lookback windows
        while cursor <= last_usable:
            out.append(cursor.to_pydatetime())
            cursor = cursor + stride
        return out

    def _history_states(self, obs: pd.Series, events: list[Event], params: PatientParams,
                        daily: DailyState) -> dict[datetime, np.ndarray]:
        """Replay the history day by day, keeping the full state at every grid point.

        Glucose is re-anchored to each day's first reading and the hidden states carry
        over, the same convention RealFitter uses, so the two agree about what the
        patient's insulin and carbohydrate on board were at any moment.
        """
        out: dict[datetime, np.ndarray] = {}
        carried: np.ndarray | None = None
        for start, day_obs, day_events in _split_days(obs, events):
            first = float(day_obs.iloc[0])
            y = default_fast(first) if carried is None else carried.copy()
            y[_G] = first
            traj = self.sim.simulate(PatientState(t=start, fast=y, daily=daily),
                                     day_events, params, 1440, 5)
            for when, row in zip(traj.t, traj.fast):
                out[when] = row
            carried = traj.fast[-1]
        return out

    def fit(self, glucose: pd.Series, events: list[Event], params: PatientParams) -> None:
        started = time.perf_counter()
        self.models = {}
        if glucose is None or len(glucose.dropna()) < 200:
            log.info("RealResidual: not enough history to train on")
            return

        obs = glucose.dropna()
        lookup = {t.to_pydatetime(): float(v) for t, v in obs.items()}
        daily = DailyState()
        evs = sorted(events, key=lambda e: e.t_start)
        history = self._history_states(obs, evs, params, daily)

        X: list[np.ndarray] = []
        Y: dict[int, list[float]] = {h: [] for h in HORIZONS}
        for origin in self._origins(glucose):
            if time.perf_counter() - started > self.time_budget_s:
                log.warning("RealResidual: ran out of time building features")
                break
            truth = {h: lookup.get(origin + timedelta(minutes=h)) for h in HORIZONS}
            if any(v is None for v in truth.values()):
                continue
            row = features_at(origin, obs, evs, daily.si_mult)
            G_now = row[FEATURE_NAMES.index("G_now")]
            if not np.isfinite(G_now):
                continue
            fast = history.get(origin)
            if fast is None:
                continue
            # Anchor glucose on the reading and keep everything else the model believes,
            # which is exactly what a forecast off the state estimator looks like. Starting
            # from default_fast instead would throw away the insulin and carbohydrate on
            # board, and the regressors would spend their capacity re-learning those rather
            # than the model error they exist to capture.
            fast = fast.copy()
            fast[_G] = G_now
            st = PatientState(t=origin, fast=fast, daily=daily)
            recent = [e for e in evs
                      if origin - timedelta(hours=12) <= e.t_start
                      <= origin + timedelta(minutes=max(HORIZONS))]
            traj = self.sim.simulate(st, recent, params, max(HORIZONS), 30)
            g = traj.fast[:, _G]
            X.append(row)
            for h in HORIZONS:
                Y[h].append(truth[h] - float(g[h // 30]))

        self.train_rows = len(X)
        if self.train_rows < 50:
            log.info("RealResidual: only %d usable origins, staying out of the way",
                     self.train_rows)
            return

        import lightgbm as lgb

        Xa = np.vstack(X)
        split = int(self.train_rows * 0.8)          # walk-forward: fit the past, score the future
        for h in HORIZONS:
            ya = np.asarray(Y[h], dtype=float)
            train = lgb.Dataset(Xa[:split], label=ya[:split],
                                feature_name=list(FEATURE_NAMES), free_raw_data=False)
            model = lgb.train(LGB_PARAMS, train, num_boost_round=LGB_ROUNDS)
            self.train_mae[h] = float(np.mean(np.abs(model.predict(Xa[:split]) - ya[:split])))
            if split < self.train_rows:
                self.holdout_mae[h] = float(np.mean(np.abs(model.predict(Xa[split:]) - ya[split:])))
                self.baseline_mae[h] = float(np.mean(np.abs(ya[split:])))
                if self.holdout_mae[h] >= self.baseline_mae[h]:
                    log.info("RealResidual: h=%d min does not beat leaving it alone "
                             "(%.1f vs %.1f mg/dL), dropping it",
                             h, self.holdout_mae[h], self.baseline_mae[h])
                    continue
            self.models[h] = model

        log.info("RealResidual: %d origins, %d of %d horizons kept, %.1f s",
                 self.train_rows, len(self.models), len(HORIZONS),
                 time.perf_counter() - started)

    # -- application -------------------------------------------------------
    def correct(self, traj: Trajectory, glucose_recent: pd.Series,
                events_recent: list[Event]) -> Trajectory:
        if not self.models or not traj.t:
            return traj
        origin = traj.t[0]
        obs = glucose_recent.dropna() if glucose_recent is not None else pd.Series(dtype=float)
        if len(obs) == 0:
            return traj
        row = features_at(origin, obs, sorted(events_recent, key=lambda e: e.t_start),
                          traj.daily_end.si_mult).reshape(1, -1)
        if not np.isfinite(row).all():
            return traj

        knots = [0.0]                                # the present needs no correction
        values = [0.0]
        for h in HORIZONS:
            model = self.models.get(h)
            if model is None:
                continue
            knots.append(float(h))
            values.append(float(model.predict(row)[0]))
        if len(knots) == 1:
            return traj

        minutes = np.array([(t - origin).total_seconds() / 60.0 for t in traj.t])
        shift = np.interp(minutes, knots, values, left=0.0, right=0.0)
        shift[minutes > knots[-1]] = 0.0            # no evidence past the last horizon

        cfg = self.sim.cfg
        fast = traj.fast.copy()
        fast[:, _G] = np.clip(fast[:, _G] + shift, cfg["G_clip_low"], cfg["G_clip_high"])
        from schema.state import energy_score
        energy = np.array([energy_score(r, traj.daily_end) for r in fast])
        return Trajectory(t=list(traj.t), fast=fast, energy=energy,
                          daily_end=replace(traj.daily_end), slow_end=replace(traj.slow_end),
                          summary=self.sim.summary(fast[:, _G], list(traj.t)))


# --------------------------------------------------------------------------- drift

class RealDrift:
    """Is the model still describing this patient?

    A rolling log of one-hour prediction errors. The score compares the last three days
    against the ten before them: 0 means in sync, above 1 means the error has more than
    doubled, which is the signature of an illness, a failed infusion site, or an insulin
    change nobody logged.
    """

    RECENT_DAYS = 3
    BASELINE_DAYS = 14
    MIN_DAYS = 4
    MAX_LOG = 20000

    def __init__(self):
        self.log: list[tuple[datetime, float]] = []

    def update(self, t, predicted_G: float, actual_G: float) -> None:
        if predicted_G is None or actual_G is None:
            return
        if not (np.isfinite(predicted_G) and np.isfinite(actual_G)):
            return
        self.log.append((t, abs(float(predicted_G) - float(actual_G))))
        if len(self.log) > self.MAX_LOG:
            self.log = self.log[-self.MAX_LOG:]

    def score(self) -> float:
        if not self.log:
            return 0.0
        now = max(t for t, _ in self.log)
        span_days = (now - min(t for t, _ in self.log)).total_seconds() / 86400.0
        if span_days < self.MIN_DAYS:
            return 0.0
        recent_from = now - timedelta(days=self.RECENT_DAYS)
        base_from = now - timedelta(days=self.BASELINE_DAYS)
        recent = [e for t, e in self.log if t > recent_from]
        baseline = [e for t, e in self.log if base_from <= t <= recent_from]
        if not recent or not baseline:
            return 0.0
        base_mean = float(np.mean(baseline))
        if base_mean <= 1e-9:
            return 0.0
        return float(max(0.0, float(np.mean(recent)) / base_mean - 1.0))
