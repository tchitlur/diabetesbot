from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Fast-layer state vector. ORDER IS THE CONTRACT. Everyone indexes with FAST_IDX.
FAST_NAMES = [
    "G",           # plasma glucose, mg/dL
    "I_p",         # plasma insulin, mU/L
    "D1",          # gut carbohydrate compartment 1, g
    "D2",          # gut carbohydrate compartment 2, g
    "gly_liver",   # liver glycogen, g (nominal full ~100)
    "gly_muscle",  # muscle glycogen, g (nominal full ~400)
    "BAC",         # blood alcohol, g/dL (0.08 = US legal limit)
    "caf",         # caffeine in body, mg
    "ket",         # blood ketones, mmol/L
    "hyd",         # hydration, fraction of euhydrated (1.0 = normal)
    "ex",          # residual exercise effect, dimensionless 0..1, decays after activity
]
FAST_IDX = {n: i for i, n in enumerate(FAST_NAMES)}
N_FAST = len(FAST_NAMES)


@dataclass
class DailyState:
    sleep_debt_h: float = 0.0     # cumulative hours below sleep target; decays
    si_mult: float = 1.0          # insulin sensitivity multiplier from sleep/training/illness/cycle
    training_load: float = 0.0    # exercise_min * intensity, decays with ~7 day time constant
    fat_mass_kg: float = 15.0
    lean_mass_kg: float = 55.0
    hba1c_pct: float = 7.0


@dataclass
class SlowState:
    # organ damage accumulators, dimensionless, 0 = at baseline for this patient
    eye: float = 0.0
    kidney: float = 0.0
    nerve: float = 0.0
    cardio: float = 0.0
    liver: float = 0.0


ORGANS = ["eye", "kidney", "nerve", "cardio", "liver"]


@dataclass
class PatientState:
    t: datetime
    fast: np.ndarray                          # shape (N_FAST,)
    daily: DailyState = field(default_factory=DailyState)
    slow: SlowState = field(default_factory=SlowState)
    fast_cov: np.ndarray | None = None        # (N_FAST, N_FAST) from the state estimator; None if unknown


def default_fast(G: float = 120.0) -> np.ndarray:
    x = np.zeros(N_FAST)
    x[FAST_IDX["G"]] = G
    x[FAST_IDX["I_p"]] = 10.0
    x[FAST_IDX["gly_liver"]] = 80.0
    x[FAST_IDX["gly_muscle"]] = 350.0
    x[FAST_IDX["hyd"]] = 1.0
    return x


def energy_score(fast: np.ndarray, daily: DailyState) -> float:
    """Readiness 0..100. Deterministic function of state; everyone uses this one.
    100 - 0.4*|G-100| - 8*sleep_debt_h + 5*min(caf/100,1) - 40*(1-hyd), clipped to [0,100]."""
    G = fast[FAST_IDX["G"]]; caf = fast[FAST_IDX["caf"]]; hyd = fast[FAST_IDX["hyd"]]
    s = 100 - 0.4*abs(G-100) - 8*daily.sleep_debt_h + 5*min(caf/100, 1) - 40*(1-hyd)
    return float(np.clip(s, 0, 100))
