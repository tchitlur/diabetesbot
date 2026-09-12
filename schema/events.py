from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel

EventType = Literal["meal","insulin","alcohol","caffeine","water","exercise","sleep","illness","stress"]
Source = Literal["cgm_export","pump","apple_health","photo","text","tap","synthetic","proposed","profile"]


class Event(BaseModel):
    """One discrete thing the patient did or proposes to do. Emitted by adapters, parsers, taps,
    the synthetic generator, and the decision engine. Consumed by simulate()."""
    type: EventType
    t_start: datetime
    duration_min: float = 0.0            # >0 for exercise, sleep, basal insulin, illness, stress
    source: Source = "tap"
    uncertainty: float = 0.10            # fractional 1-sigma on the primary amount (carbs_g, units, ethanol_g, ...)
    label: str = ""                      # display only, e.g. "2 IPAs", "Chipotle bowl"
    # meal (also alcohol drinks' carbs)
    carbs_g: float = 0.0
    fat_g: float = 0.0
    protein_g: float = 0.0
    fiber_g: float = 0.0
    # insulin
    units: float = 0.0                   # bolus: total units at t_start. basal: units PER HOUR over duration_min
    insulin_kind: Optional[Literal["bolus","basal"]] = None
    # alcohol
    ethanol_g: float = 0.0               # one US standard drink = 14 g
    # caffeine
    caffeine_mg: float = 0.0
    # water
    water_ml: float = 0.0
    # exercise
    intensity: float = 0.0               # 0..1 (0.3 walk, 0.6 moderate, 0.9 hard)
    exercise_kind: Optional[Literal["aerobic","anaerobic","mixed"]] = None
    # sleep
    sleep_quality: float = 1.0           # 0..1
    # illness / stress
    severity: float = 0.0                # 0..1


class Profile(BaseModel):
    age: int = 30
    sex: Literal["M","F"] = "M"
    height_cm: float = 175
    weight_kg: float = 70
    body_fat_pct: float = 20
    diabetes_years: float = 10
    hba1c_pct: float = 7.0
    egfr: float = 100
    systolic_bp: float = 120
    carb_ratio: float = 10          # initial guess, g/U
    correction_factor: float = 40   # initial guess, mg/dL per U
    basal_u_per_h: float = 1.0
    target_low: float = 70
    target_high: float = 180
    sleep_target_h: float = 8
    existing_complications: list[str] = []
    meds: list[str] = []            # e.g. ["sglt2", "beta_blocker", "steroid"]
