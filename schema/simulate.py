from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
import numpy as np
from .state import PatientState, DailyState, SlowState
from .events import Event
from .params import PatientParams


@dataclass
class Trajectory:
    t: list[datetime]           # length T, spaced dt_min
    fast: np.ndarray            # (T, N_FAST)
    energy: np.ndarray          # (T,) from schema.state.energy_score
    daily_end: DailyState
    slow_end: SlowState
    summary: dict               # keys: tir_pct, tbr70_pct, tar180_pct, min_G, max_G, t_min_G (iso str), mean_G


@dataclass
class LongTrajectory:
    years: np.ndarray           # e.g. 0..10 step 0.25
    hba1c: np.ndarray
    mean_G: np.ndarray
    hypo_per_week: np.ndarray
    damage: dict[str, np.ndarray]   # organ -> damage accumulator over years
    risk10: dict[str, float]        # organ -> 10-year probability of the named complication, 0..1


class Simulator(Protocol):
    def simulate(self, state: PatientState, events: list[Event], params: PatientParams,
                 horizon_min: int, dt_min: int = 5) -> Trajectory: ...
    def simulate_long(self, state: PatientState, weekly_template: list[Event], params: PatientParams,
                      years: float = 10.0) -> LongTrajectory: ...


# weekly_template convention: events whose t_start falls in the reference week
# Monday 2000-01-03 00:00 through Sunday 2000-01-09 23:59. simulate_long tiles this week.
REF_WEEK_START = datetime(2000, 1, 3)


def get_simulator() -> Simulator:
    """Returns the real engine if present, else the stub. Everyone calls this; nobody imports engine directly."""
    try:
        from engine.simulate import RealSimulator
        return RealSimulator()
    except ImportError:
        from engine.stub import StubSimulator
        return StubSimulator()
