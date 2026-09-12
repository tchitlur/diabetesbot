"""Do-nothing inference. Lets the rest of the system run before inference/impl.py exists."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from schema.events import Event
from schema.params import ParamPosterior, PatientParams
from schema.simulate import Trajectory
from schema.state import FAST_IDX, N_FAST, PatientState

SMALL_VAR = 1.0   # diagonal of the returned covariance; "we know nothing, but pretend it's tight"


class NaiveFitter:
    def fit(self, glucose: pd.Series, events: list[Event], prior: ParamPosterior) -> ParamPosterior:
        return prior


class NaiveEstimator:
    def estimate(self, glucose: pd.Series, events: list[Event], params: PatientParams,
                 profile_state: PatientState) -> PatientState:
        fast = np.asarray(profile_state.fast, dtype=float).copy()
        t = profile_state.t
        if glucose is not None and len(glucose) > 0:
            valid = glucose.dropna()
            if len(valid) > 0:
                fast[FAST_IDX["G"]] = float(valid.iloc[-1])
            t = glucose.index[-1]
        return PatientState(
            t=t,
            fast=fast,
            daily=replace(profile_state.daily),
            slow=replace(profile_state.slow),
            fast_cov=np.eye(N_FAST) * SMALL_VAR,
        )


class NaiveResidual:
    def fit(self, glucose: pd.Series, events: list[Event], params: PatientParams) -> None:
        return None

    def correct(self, traj: Trajectory, glucose_recent: pd.Series, events_recent: list[Event]) -> Trajectory:
        return traj


class NaiveDrift:
    def update(self, t, predicted_G: float, actual_G: float) -> None:
        return None

    def score(self) -> float:
        return 0.0
