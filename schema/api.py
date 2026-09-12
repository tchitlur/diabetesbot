from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel
from .events import Event, Profile


class Point(BaseModel):
    t: datetime; y: float


class Band(BaseModel):
    t: list[datetime]; p05: list[float]; p50: list[float]; p95: list[float]


class Explanation(BaseModel):
    confidence: float                 # 0..1
    data_used: list[str]              # e.g. ["CGM (last 15 min)", "pump log", "sleep"]
    data_missing: list[str]           # e.g. ["insulin bolus for lunch"]
    top_factors: list[str]            # e.g. ["28 g ethanol at 22:30 suppressing liver output"]


class Safety(BaseModel):
    status: Literal["ok","need_data","escalate"]
    reason: str = ""
    needed: list[str] = []            # what to provide when status == need_data
    rule_id: Optional[str] = None     # which clinician rule fired


class ActiveSubstance(BaseModel):
    name: Literal["insulin","alcohol","caffeine","carbs","exercise"]
    remaining_pct: float              # 0..100
    ends_at: datetime


class Alert(BaseModel):
    t_start: datetime; t_end: datetime
    kind: Literal["low","high","ketone","drift"]
    probability: float
    cause: str


class Variant(BaseModel):
    label: str                        # "proposed", "baseline", "delay 1h", "half amount", "+15 g carbs", "pre-workout snack"
    events: list[Event]
    band: Band
    p_hypo: float; p_hyper: float; tir_pct: float; energy_avg: float
    delta_risk10: dict[str, float]    # organ -> change in 10y probability vs baseline (from simulate_long, 0 if unavailable)
    passes: bool


class StateResponse(BaseModel):
    t: datetime
    glucose_now: Optional[float]; glucose_age_min: Optional[float]
    estimated: dict[str, float]       # FAST_NAMES -> value
    estimated_sd: dict[str, float]
    energy_score: float
    active: list[ActiveSubstance]
    daily: dict[str, float]           # DailyState fields
    safety: Safety


class ForecastResponse(BaseModel):
    band: Band; p_hypo_8h: float; p_hyper_8h: float; tir_pct: float
    alerts: list[Alert]; explanation: Explanation; safety: Safety


class PermissionRequest(BaseModel):
    text: Optional[str] = None        # free text; server parses
    event: Optional[Event] = None     # or a structured proposed event from a tap
    patient_id: str = "default"


class PermissionResponse(BaseModel):
    verdict: Literal["approved","approved_with_condition","denied"]
    condition: str                    # "" if none
    parsed_event: Optional[Event]
    proposed: Variant; baseline: Variant; best: Variant
    alternatives: list[Variant]
    explanation: Explanation; safety: Safety


class PlanRequest(BaseModel):
    date: datetime; calendar: list[Event]     # fixed commitments as events with source="proposed"
    wants: list[Event] = []                   # flexible items to place (workout, drinks)
    patient_id: str = "default"


class PlanWindow(BaseModel):
    start: datetime; end: datetime; status: Literal["green","amber","red"]; label: str


class PlanResponse(BaseModel):
    windows: list[PlanWindow]; scheduled: list[Event]; energy: list[Point]
    tir_pct: float; p_hypo: float; explanation: Explanation; safety: Safety


class ChangeRequest(BaseModel):
    description: str                          # "switch to morning workouts"
    weekly_template: Optional[list[Event]] = None   # if the client already built it; else server parses description
    patient_id: str = "default"


class Delta(BaseModel):
    name: str; baseline: float; proposed: float; unit: str


class ChangeResponse(BaseModel):
    verdict: Literal["merged","merged_with_changes","rejected"]
    changes_requested: str
    deltas_30d: list[Delta]; deltas_10y: list[Delta]
    risk10_baseline: dict[str, float]; risk10_proposed: dict[str, float]
    explanation: Explanation; safety: Safety


class BudgetResponse(BaseModel):
    weekly: dict[str, float]                  # {"alcohol_drinks": 6, "caffeine_mg": 1400, "offplan_meals": 3}
    risk_tolerance: float                     # allowed increase in 10y risk, e.g. 0.01
    explanation: Explanation


class AlertsResponse(BaseModel):
    alerts: list[Alert]; safety: Safety


class ReplayRequest(BaseModel):
    date: datetime; overrides: list[Event]    # replaces same-type events on that day; empty = as recorded
    patient_id: str = "default"


class ReplayResponse(BaseModel):
    actual: list[Point]; simulated: Band; events: list[Event]


class TrendsResponse(BaseModel):
    hba1c_proj: list[Point]; si_drift: list[Point]; hypo_per_week: list[Point]
    sleep_vs_si: list[dict]                   # [{"sleep_h": 6.2, "si": 0.83}, ...]


class OrganView(BaseModel):
    damage_now: float; risk10_baseline: float; risk10_proposed: Optional[float]; trajectory: list[Point]


class BodyResponse(BaseModel):
    organs: dict[str, OrganView]              # keys = schema.state.ORGANS


class ParseResponse(BaseModel):
    event: Optional[Event]; confidence: float
    question_type: Literal["permission","plan","change","budget","why","other"]
    raw: str


class ClinicianResponse(BaseModel):
    params_history: list[dict]                # [{"t": iso, **PatientParams.to_dict()}]
    residual_log: list[Point]
    escalations: list[Safety]
    drift_score: float


# ROUTES (all under /api, JSON bodies as above, patient_id query param defaults to "default"):
# GET  /state                 -> StateResponse
# GET  /forecast?hours=8      -> ForecastResponse
# POST /permission            -> PermissionResponse
# POST /plan                  -> PlanResponse
# POST /change                -> ChangeResponse
# GET  /budget                -> BudgetResponse
# GET  /alerts                -> AlertsResponse
# POST /replay                -> ReplayResponse
# GET  /trends                -> TrendsResponse
# GET  /body?change_id=       -> BodyResponse
# POST /parse  {text}         -> ParseResponse
# POST /parse_photo (multipart image) -> Event (type=meal or alcohol) with uncertainty
# POST /events (Event)        -> 204   (tap-logged intake)
# POST /upload (multipart: kind=cgm|pump|health, file) -> {"events": n, "glucose_points": n}
# GET  /clinician             -> ClinicianResponse
# GET  /health                -> {"ok": true, "simulator": "real"|"stub", "inference": "real"|"naive"}
