from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

# Per-patient FITTED parameters. Everything else (population constants) lives in engine/params.yaml.
PARAM_NAMES = ["si_night","si_day","si_evening","carb_ratio","k_abs","k_ex","k_alc","basal_drift","k_sleep"]


@dataclass
class PatientParams:
    si_night: float = 50.0     # mg/dL drop per unit insulin, 00:00-08:00
    si_day: float = 40.0       # 08:00-16:00
    si_evening: float = 45.0   # 16:00-24:00
    carb_ratio: float = 10.0   # g carbs covered per unit
    k_abs: float = 0.03        # 1/min gut absorption rate constant
    k_ex: float = 1.0          # scale on exercise-driven sensitivity/uptake (1 = population)
    k_alc: float = 1.0         # scale on alcohol suppression of hepatic glucose output (1 = population)
    basal_drift: float = 0.0   # U/h added to logged basal (captures under/over-basaled patients)
    k_sleep: float = 0.20      # fractional insulin-sensitivity drop per hour of sleep debt, capped at 0.5 total

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=float)
    @classmethod
    def from_array(cls, a) -> "PatientParams":
        return cls(**{n: float(v) for n, v in zip(PARAM_NAMES, a)})
    def to_dict(self) -> dict: return asdict(self)


@dataclass
class ParamPosterior:
    mean: PatientParams
    samples: np.ndarray      # shape (n_samples, len(PARAM_NAMES))
    def sample(self, n: int, rng: np.random.Generator) -> list[PatientParams]:
        idx = rng.integers(0, len(self.samples), size=n)
        return [PatientParams.from_array(self.samples[i]) for i in idx]


def population_prior(n_samples: int = 500, seed: int = 0) -> ParamPosterior:
    """Wide prior. Lognormal around defaults, 30% CV, k_sleep/basal_drift normal."""
    rng = np.random.default_rng(seed)
    mean = PatientParams()
    m = mean.to_array()
    s = np.empty((n_samples, len(m)))
    for j, name in enumerate(PARAM_NAMES):
        if name == "basal_drift": s[:, j] = rng.normal(0, 0.2, n_samples)
        elif name == "k_sleep":   s[:, j] = np.clip(rng.normal(0.2, 0.08, n_samples), 0, 0.5)
        else:                     s[:, j] = m[j] * np.exp(rng.normal(0, 0.3, n_samples))
    return ParamPosterior(mean=mean, samples=s)
