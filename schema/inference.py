from __future__ import annotations
from typing import Protocol
import pandas as pd
from .state import PatientState
from .events import Event
from .params import PatientParams, ParamPosterior
from .simulate import Trajectory

# glucose series convention everywhere: pd.Series, float mg/dL, DatetimeIndex on a 5-minute grid, NaN for gaps.


class Fitter(Protocol):
    def fit(self, glucose: pd.Series, events: list[Event], prior: ParamPosterior) -> ParamPosterior: ...


class Estimator(Protocol):
    def estimate(self, glucose: pd.Series, events: list[Event], params: PatientParams,
                 profile_state: PatientState) -> PatientState: ...
    # returns state at glucose.index[-1] with fast_cov filled


class Residual(Protocol):
    def fit(self, glucose: pd.Series, events: list[Event], params: PatientParams) -> None: ...
    def correct(self, traj: Trajectory, glucose_recent: pd.Series, events_recent: list[Event]) -> Trajectory: ...
    # returns a new Trajectory with fast[:, G] adjusted at 30/60/120 min ahead (interpolated between)


class DriftMonitor(Protocol):
    def update(self, t, predicted_G: float, actual_G: float) -> None: ...
    def score(self) -> float: ...     # 0 = in sync; >1 = out of sync (3+ days of rising error)


def get_inference():
    """Returns (fitter, estimator, residual, drift). Real ones if inference package has them, else naive ones."""
    try:
        from inference.impl import RealFitter, RealEstimator, RealResidual, RealDrift
        return RealFitter(), RealEstimator(), RealResidual(), RealDrift()
    except ImportError:
        from inference.naive import NaiveFitter, NaiveEstimator, NaiveResidual, NaiveDrift
        return NaiveFitter(), NaiveEstimator(), NaiveResidual(), NaiveDrift()
