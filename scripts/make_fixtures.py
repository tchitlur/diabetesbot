"""Generate data/fixtures/*.json: one file per schema.api response model.

Everything here is fake and deterministic. The frontend builds against these files
before the API exists, and tests/test_contracts.py round-trips them through the
pydantic models so a schema change that breaks the UI fails CI instead of the demo.

Run: python scripts/make_fixtures.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.stub import StubSimulator                                  # noqa: E402
from schema.api import (                                               # noqa: E402
    ActiveSubstance, Alert, AlertsResponse, Band, BodyResponse, BudgetResponse,
    ChangeResponse, ClinicianResponse, Delta, Explanation, ForecastResponse,
    OrganView, ParseResponse, PermissionResponse, PlanResponse, PlanWindow, Point,
    ReplayResponse, Safety, StateResponse, TrendsResponse, Variant,
)
from schema.events import Event                                        # noqa: E402
from schema.params import PatientParams                                # noqa: E402
from schema.simulate import REF_WEEK_START, Trajectory                 # noqa: E402
from schema.state import (                                             # noqa: E402
    DailyState, FAST_IDX, FAST_NAMES, ORGANS, PatientState, default_fast,
)

OUT = ROOT / "data" / "fixtures"

DAY = datetime(2026, 9, 11)
T0 = DAY.replace(hour=7)              # the simulated day starts here
NOW = DAY.replace(hour=22, minute=45)  # "now" for the state/forecast fixtures
DT_MIN = 5
HORIZON_MIN = 24 * 60

SIM = StubSimulator()
PARAMS = PatientParams()
RNG = np.random.default_rng(7)


def at(h: int, m: int = 0, day: datetime = DAY) -> datetime:
    return day.replace(hour=h, minute=m)


def day_events() -> list[Event]:
    """One day as recorded: breakfast + bolus, midday workout, dinner + bolus, two drinks."""
    return [
        Event(type="meal", t_start=at(7, 30), carbs_g=60, fat_g=12, protein_g=18, fiber_g=6,
              source="photo", uncertainty=0.15, label="Oatmeal, banana, coffee"),
        Event(type="insulin", t_start=at(7, 30), units=6, insulin_kind="bolus",
              source="pump", uncertainty=0.02, label="Breakfast bolus"),
        Event(type="exercise", t_start=at(12, 0), duration_min=45, intensity=0.6,
              exercise_kind="aerobic", source="apple_health", uncertainty=0.10,
              label="45 min moderate run"),
        Event(type="meal", t_start=at(19, 0), carbs_g=80, fat_g=28, protein_g=35, fiber_g=9,
              source="photo", uncertainty=0.20, label="Chipotle bowl"),
        Event(type="insulin", t_start=at(19, 0), units=8, insulin_kind="bolus",
              source="pump", uncertainty=0.02, label="Dinner bolus"),
        Event(type="alcohol", t_start=at(22, 30), ethanol_g=28, source="text",
              uncertainty=0.25, label="2 IPAs"),
    ]


def start_state() -> PatientState:
    return PatientState(
        t=T0,
        fast=default_fast(112.0),
        daily=DailyState(sleep_debt_h=1.4, si_mult=0.92, training_load=180.0,
                         fat_mass_kg=15.5, lean_mass_kg=56.0, hba1c_pct=7.1),
    )


def run(events: list[Event]) -> Trajectory:
    return SIM.simulate(start_state(), events, PARAMS, HORIZON_MIN, DT_MIN)


def idx_of(t: datetime) -> int:
    return int((t - T0).total_seconds() // 60 // DT_MIN)


def band(traj: Trajectory, i0: int, i1: int, every: int = 1, sd0: float = 5.0,
         growth: float = 0.35) -> Band:
    """p50 = simulated glucose; p05/p95 widen with horizon."""
    sel = range(i0, i1 + 1, every)
    ts, p05, p50, p95 = [], [], [], []
    for k, i in enumerate(sel):
        g = float(traj.fast[i, FAST_IDX["G"]])
        sd = sd0 + growth * k * every
        ts.append(traj.t[i])
        p50.append(round(g, 1))
        p05.append(round(float(np.clip(g - 1.645 * sd, 40, 400)), 1))
        p95.append(round(float(np.clip(g + 1.645 * sd, 40, 400)), 1))
    return Band(t=ts, p05=p05, p50=p50, p95=p95)


def window_summary(traj: Trajectory, i0: int, i1: int) -> dict:
    g = traj.fast[i0:i1 + 1, FAST_IDX["G"]]
    return {
        "tir_pct": round(float(100 * np.mean((g >= 70) & (g <= 180))), 1),
        "min_G": round(float(g.min()), 1),
        "max_G": round(float(g.max()), 1),
        "mean_G": round(float(g.mean()), 1),
        "energy_avg": round(float(traj.energy[i0:i1 + 1].mean()), 1),
    }


def weekly_template(workout_days: tuple[int, ...] = (0, 2, 4), workout_hour: int = 12) -> list[Event]:
    """The patient's routine tiled across the reference week, with Fri/Sat drinks.

    Boluses are deliberately a touch short of the 10 g/U carb ratio: this is a patient
    running an HbA1c near 7%, not a textbook one.
    """
    out: list[Event] = []
    for d in range(7):
        day = REF_WEEK_START + timedelta(days=d)
        out += [
            Event(type="meal", t_start=day.replace(hour=7, minute=30), carbs_g=60,
                  source="synthetic", label="breakfast"),
            Event(type="insulin", t_start=day.replace(hour=7, minute=30), units=4.0,
                  insulin_kind="bolus", source="synthetic", label="breakfast bolus"),
            Event(type="meal", t_start=day.replace(hour=13, minute=0), carbs_g=70,
                  source="synthetic", label="lunch"),
            Event(type="insulin", t_start=day.replace(hour=13, minute=0), units=4.0,
                  insulin_kind="bolus", source="synthetic", label="lunch bolus"),
            Event(type="meal", t_start=day.replace(hour=19, minute=0), carbs_g=80,
                  source="synthetic", label="dinner"),
            Event(type="insulin", t_start=day.replace(hour=19, minute=0), units=5.0,
                  insulin_kind="bolus", source="synthetic", label="dinner bolus"),
        ]
        if d in workout_days:
            out.append(Event(type="exercise", t_start=day.replace(hour=workout_hour, minute=30),
                             duration_min=45, intensity=0.6, exercise_kind="aerobic",
                             source="synthetic", label="run"))
        if d in (4, 5):
            out.append(Event(type="alcohol", t_start=day.replace(hour=22, minute=30),
                             ethanol_g=28, source="synthetic", label="2 drinks"))
    return out


def morning_template() -> list[Event]:
    """The 'change' being evaluated: workouts at 06:30, and five of them instead of three."""
    return weekly_template(workout_days=(0, 1, 2, 3, 4), workout_hour=6)


def explanation(confidence: float, used: list[str], missing: list[str], factors: list[str]) -> Explanation:
    return Explanation(confidence=confidence, data_used=used, data_missing=missing, top_factors=factors)


OK = Safety(status="ok", reason="", needed=[], rule_id=None)


def write(name: str, model) -> None:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(model.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    events = day_events()
    traj = run(events)
    i_now = idx_of(NOW)
    i_end8 = i_now + (8 * 60) // DT_MIN

    no_alcohol = [e for e in events if e.type != "alcohol"]
    snack = Event(type="meal", t_start=at(23, 30), carbs_g=20, source="proposed",
                  uncertainty=0.10, label="20 g bedtime snack")
    half_drinks = [e.model_copy(update={"ethanol_g": 14.0, "label": "1 IPA"})
                   if e.type == "alcohol" else e for e in events]
    early_drinks = [e.model_copy(update={"t_start": at(20, 30)})
                    if e.type == "alcohol" else e for e in events]

    traj_baseline = run(no_alcohol)
    traj_best = run(events + [snack])
    traj_half = run(half_drinks)
    traj_early = run(early_drinks)

    long_base = SIM.simulate_long(start_state(), weekly_template(), PARAMS, years=10.0)
    long_prop = SIM.simulate_long(start_state(), morning_template(), PARAMS, years=10.0)

    # ---------------- state ----------------
    fast_now = traj.fast[i_now]
    write("state", StateResponse(
        t=NOW,
        glucose_now=round(float(fast_now[FAST_IDX["G"]]), 1),
        glucose_age_min=4.0,
        estimated={n: round(float(fast_now[FAST_IDX[n]]), 3) for n in FAST_NAMES},
        estimated_sd={n: v for n, v in zip(FAST_NAMES, [8.0, 2.5, 4.0, 4.0, 6.0, 20.0,
                                                        0.004, 12.0, 0.05, 0.03, 0.08])},
        energy_score=round(float(traj.energy[i_now]), 1),
        active=[
            ActiveSubstance(name="insulin", remaining_pct=22.0, ends_at=at(0, 0, DAY + timedelta(days=1))),
            ActiveSubstance(name="carbs", remaining_pct=14.0, ends_at=at(23, 30)),
            ActiveSubstance(name="alcohol", remaining_pct=88.0,
                            ends_at=at(8, 0, DAY + timedelta(days=1))),
        ],
        daily={"sleep_debt_h": 1.4, "si_mult": 0.92, "training_load": 180.0,
               "fat_mass_kg": 15.5, "lean_mass_kg": 56.0, "hba1c_pct": 7.1},
        safety=OK,
    ))

    # ---------------- forecast ----------------
    w = window_summary(traj, i_now, i_end8)
    write("forecast", ForecastResponse(
        band=band(traj, i_now, i_end8),
        p_hypo_8h=0.38, p_hyper_8h=0.11, tir_pct=w["tir_pct"],
        alerts=[
            Alert(t_start=at(4, 0, DAY + timedelta(days=1)),
                  t_end=at(7, 30, DAY + timedelta(days=1)),
                  kind="low", probability=0.38,
                  cause="28 g ethanol at 22:30 suppressing overnight liver glucose output"),
        ],
        explanation=explanation(
            0.72,
            ["CGM (last 15 min)", "pump log", "meal photos", "Apple Health workout"],
            ["carb estimate for dinner is a photo guess (±20%)"],
            ["28 g ethanol at 22:30 suppressing liver output",
             "45 min run at 12:00 still raising insulin sensitivity",
             "1.4 h sleep debt lowering sensitivity slightly"],
        ),
        safety=OK,
    ))

    # ---------------- permission ----------------
    zero_risk = {o: 0.0 for o in ORGANS}
    drink_delta = {o: round(0.002 + 0.001 * i, 4) for i, o in enumerate(ORGANS)}

    def variant(label: str, evs: list[Event], tr: Trajectory, p_hypo: float, p_hyper: float,
                delta: dict[str, float], passes: bool) -> Variant:
        s = window_summary(tr, i_now, i_end8)
        return Variant(label=label, events=evs, band=band(tr, i_now, i_end8, every=3),
                       p_hypo=p_hypo, p_hyper=p_hyper, tir_pct=s["tir_pct"],
                       energy_avg=s["energy_avg"], delta_risk10=delta, passes=passes)

    proposed_ev = Event(type="alcohol", t_start=at(22, 30), ethanol_g=28, source="text",
                        uncertainty=0.25, label="2 IPAs")
    write("permission", PermissionResponse(
        verdict="approved_with_condition",
        condition="Eat 20 g carbs before bed",
        parsed_event=proposed_ev,
        proposed=variant("proposed", [proposed_ev], traj, 0.38, 0.10, drink_delta, True),
        baseline=variant("baseline", [], traj_baseline, 0.06, 0.14, zero_risk, True),
        best=variant("+20 g carbs at bedtime", [proposed_ev, snack], traj_best, 0.09, 0.16,
                     drink_delta, True),
        alternatives=[
            variant("half amount", [proposed_ev.model_copy(update={"ethanol_g": 14.0, "label": "1 IPA"})],
                    traj_half, 0.17, 0.12, {o: v / 2 for o, v in drink_delta.items()}, True),
            variant("delay 2h earlier", [proposed_ev.model_copy(update={"t_start": at(20, 30)})],
                    traj_early, 0.21, 0.13, drink_delta, True),
        ],
        explanation=explanation(
            0.68,
            ["CGM (last 15 min)", "pump log", "dinner photo"],
            ["exact ABV of the beers"],
            ["Alcohol peaks 5-9 h after the last drink, right through your sleep window",
             "Dinner bolus of 8 U is still ~22% active",
             "Your last two drinking nights both ended below 70 mg/dL"],
        ),
        safety=OK,
    ))

    # ---------------- plan ----------------
    write("plan", PlanResponse(
        windows=[
            PlanWindow(start=at(7, 0), end=at(11, 0), status="green", label="Stable, good for focus work"),
            PlanWindow(start=at(11, 0), end=at(13, 0), status="amber", label="Pre-workout, watch the drop"),
            PlanWindow(start=at(13, 0), end=at(18, 0), status="green", label="Post-workout, high sensitivity"),
            PlanWindow(start=at(18, 0), end=at(22, 0), status="green", label="Dinner covered"),
            PlanWindow(start=at(22, 0), end=at(7, 30, DAY + timedelta(days=1)), status="red",
                       label="Alcohol + overnight hypo risk"),
        ],
        scheduled=[
            Event(type="exercise", t_start=at(12, 0), duration_min=45, intensity=0.6,
                  exercise_kind="aerobic", source="proposed", label="45 min run"),
            snack,
        ],
        energy=[Point(t=traj.t[i], y=round(float(traj.energy[i]), 1))
                for i in range(0, len(traj.t), 12)],
        tir_pct=traj.summary["tir_pct"],
        p_hypo=0.31,
        explanation=explanation(
            0.70,
            ["calendar", "CGM history", "pump log", "sleep"],
            ["whether dinner is out or at home"],
            ["Midday run lands in the flattest window",
             "Drinks after 22:00 push hypo risk into your sleep"],
        ),
        safety=OK,
    ))

    # ---------------- change ----------------
    write("change", ChangeResponse(
        verdict="merged_with_changes",
        changes_requested="Keep Friday's session in the evening; move Mon/Wed to 06:30 only.",
        deltas_30d=[
            Delta(name="Time in range", baseline=68.0, proposed=74.0, unit="%"),
            Delta(name="Hypos per week", baseline=2.8, proposed=1.6, unit="count"),
            Delta(name="Mean glucose", baseline=round(float(long_base.mean_G[0]), 1),
                  proposed=round(float(long_prop.mean_G[0]), 1), unit="mg/dL"),
            Delta(name="Energy score", baseline=71.0, proposed=76.0, unit="0-100"),
        ],
        deltas_10y=[
            Delta(name="HbA1c", baseline=round(float(long_base.hba1c[0]), 2),
                  proposed=round(float(long_prop.hba1c[0]), 2), unit="%"),
            Delta(name="Retinopathy risk", baseline=float(round(long_base.risk10["eye"], 4)),
                  proposed=float(round(long_prop.risk10["eye"], 4)), unit="probability"),
            Delta(name="Nephropathy risk", baseline=float(round(long_base.risk10["kidney"], 4)),
                  proposed=float(round(long_prop.risk10["kidney"], 4)), unit="probability"),
        ],
        risk10_baseline={o: round(float(v), 4) for o, v in long_base.risk10.items()},
        risk10_proposed={o: round(float(v), 4) for o, v in long_prop.risk10.items()},
        explanation=explanation(
            0.61,
            ["12 weeks of CGM", "pump log", "workout history", "sleep"],
            ["how well you actually sleep when you get up at 05:45"],
            ["Morning workouts remove the post-lunch insulin stack",
             "Fridays stay in the evening because that is when you drink"],
        ),
        safety=OK,
    ))

    # ---------------- budget ----------------
    write("budget", BudgetResponse(
        weekly={"alcohol_drinks": 6.0, "caffeine_mg": 1400.0, "offplan_meals": 3.0},
        risk_tolerance=0.01,
        explanation=explanation(
            0.55,
            ["10 y simulation", "12 weeks of CGM"],
            ["liver panel"],
            ["6 drinks/week keeps the 10 y liver and cardio increase under 1 point",
             "Caffeine after 14:00 costs you more in sleep debt than it returns in energy"],
        ),
    ))

    # ---------------- alerts ----------------
    write("alerts", AlertsResponse(
        alerts=[
            Alert(t_start=at(4, 0, DAY + timedelta(days=1)),
                  t_end=at(7, 30, DAY + timedelta(days=1)), kind="low", probability=0.38,
                  cause="Alcohol at 22:30 suppressing overnight liver glucose output"),
            Alert(t_start=at(20, 0), t_end=at(21, 30), kind="high", probability=0.22,
                  cause="High-fat dinner absorbing later than the bolus covers"),
            Alert(t_start=at(7, 0), t_end=at(9, 0, DAY + timedelta(days=1)), kind="drift",
                  probability=0.12, cause="Morning predictions running 18 mg/dL low for 2 days"),
        ],
        safety=OK,
    ))

    # ---------------- replay ----------------
    actual_idx = range(0, len(traj.t), 3)
    write("replay", ReplayResponse(
        actual=[Point(t=traj.t[i],
                      y=round(float(np.clip(traj.fast[i, FAST_IDX["G"]] + RNG.normal(0, 6), 40, 400)), 1))
                for i in actual_idx],
        simulated=band(traj, 0, len(traj.t) - 1, every=3),
        events=events,
    ))

    # ---------------- trends ----------------
    week0 = DAY - timedelta(weeks=11)
    write("trends", TrendsResponse(
        hba1c_proj=[Point(t=DAY + timedelta(days=int(round(y * 365))), y=round(float(h), 2))
                    for y, h in zip(long_base.years[::4], long_base.hba1c[::4])],
        si_drift=[Point(t=week0 + timedelta(weeks=i), y=round(0.78 + 0.012 * i, 3)) for i in range(12)],
        hypo_per_week=[Point(t=week0 + timedelta(weeks=i), y=float(v))
                       for i, v in enumerate([4, 3, 3, 5, 2, 3, 2, 2, 3, 1, 2, 2])],
        sleep_vs_si=[{"sleep_h": round(s, 1), "si": round(si, 2)} for s, si in
                     [(5.1, 0.74), (5.8, 0.79), (6.2, 0.83), (6.6, 0.86), (7.0, 0.91),
                      (7.4, 0.95), (7.9, 0.99), (8.3, 1.02)]],
    ))

    # ---------------- body ----------------
    write("body", BodyResponse(organs={
        o: OrganView(
            damage_now=round(float(long_base.damage[o][4]), 4),
            risk10_baseline=round(float(long_base.risk10[o]), 4),
            risk10_proposed=round(float(long_prop.risk10[o]), 4),
            trajectory=[Point(t=DAY + timedelta(days=int(round(y * 365))), y=round(float(d), 4))
                        for y, d in zip(long_base.years[::4], long_base.damage[o][::4])],
        ) for o in ORGANS
    }))

    # ---------------- parse ----------------
    write("parse", ParseResponse(
        event=proposed_ev,
        confidence=0.81,
        question_type="permission",
        raw="can I have two IPAs tonight?",
    ))

    # ---------------- clinician ----------------
    write("clinician", ClinicianResponse(
        params_history=[
            {"t": (DAY - timedelta(days=d)).isoformat(),
             **{k: round(v * (1 + 0.01 * d), 4) for k, v in PARAMS.to_dict().items()}}
            for d in (28, 21, 14, 7, 0)
        ],
        residual_log=[Point(t=traj.t[i], y=round(float(RNG.normal(0, 9)), 1))
                      for i in range(0, len(traj.t), 12)],
        escalations=[
            Safety(status="escalate", reason="Three or more severe lows this week. This needs clinical review.",
                   needed=[], rule_id="escalate_repeated_severe_lows"),
            Safety(status="need_data", reason="Need a CGM reading from the last 20 minutes.",
                   needed=["CGM reading"], rule_id="stale_cgm"),
        ],
        drift_score=0.42,
    ))


if __name__ == "__main__":
    main()
