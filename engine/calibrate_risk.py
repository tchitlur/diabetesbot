"""Derive the organ-damage and hazard constants in params.yaml from the DCCT targets.

The slow layer has three sets of constants per organ: a damage rate `k_<organ>`, a
baseline hazard, and a `beta`. They are not measurable individually - only the risks
they produce are. This script pins them:

1. Build a reference patient at HbA1c 7.0 and the same patient at HbA1c 9.0, by
   bisecting total insulin until the simulated mean glucose hits 154 and 222 mg/dL.
2. Run the ten-year projection for both with every `k_<organ>` set to 1, giving the raw
   damage each organ accumulates.
3. Normalise `k_<organ>` so the reference patient accumulates exactly 1.0 of damage.
   "Damage 1.0" then means "what ten years at HbA1c 7 does to this organ".
4. Solve the two-point hazard exactly: the reference patient must land on the DCCT
   ten-year risk, and the HbA1c 9.0 patient on that risk times the target ratio.

    risk = 1 - exp(-h * exp(beta * damage))
    => beta = ln(B/A) / (D9 - D7),  h = A * exp(-beta * D7)
       with A = -ln(1 - risk7), B = -ln(1 - risk9)

Run `python engine/calibrate_risk.py` to see the numbers, `--write` to patch params.yaml.
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.simulate import RealSimulator, defaults, load_params  # noqa: E402
from schema.events import Event  # noqa: E402
from schema.params import PatientParams  # noqa: E402
from schema.simulate import REF_WEEK_START  # noqa: E402
from schema.state import DailyState, ORGANS, PatientState, default_fast  # noqa: E402

PARAMS_PATH = ROOT / "engine" / "params.yaml"


def reference_template(ins_scale: float, weekly_ethanol_g: float) -> list[Event]:
    """The calibration patient's week: three bolused meals a day, three workouts,
    three drinks, and half an hour less sleep than they need."""
    out: list[Event] = []
    per_drink = 14.0
    n_drinks = int(round(weekly_ethanol_g / per_drink))
    drink_days = [4, 5, 3, 6, 2, 1, 0]
    for d in range(7):
        day = REF_WEEK_START + timedelta(days=d)
        out.append(Event(type="insulin", t_start=day, duration_min=1440,
                         units=1.0 * ins_scale, insulin_kind="basal", source="synthetic"))
        out.append(Event(type="sleep", t_start=day.replace(hour=23), duration_min=7.5 * 60,
                         source="synthetic"))
        for hour, minute, carbs, units in ((7, 30, 60, 6.0), (13, 0, 70, 7.0), (19, 0, 80, 8.0)):
            ts = day.replace(hour=hour, minute=minute)
            out.append(Event(type="meal", t_start=ts, carbs_g=carbs, fat_g=carbs * 0.25,
                             protein_g=carbs * 0.30, source="synthetic"))
            out.append(Event(type="insulin", t_start=ts, units=units * ins_scale,
                             insulin_kind="bolus", source="synthetic"))
        if d in (0, 2, 4):
            out.append(Event(type="exercise", t_start=day.replace(hour=17, minute=30),
                             duration_min=45, intensity=0.6, exercise_kind="aerobic",
                             source="synthetic"))
    for i in range(n_drinks):
        day = REF_WEEK_START + timedelta(days=drink_days[i % 7])
        out.append(Event(type="alcohol", t_start=day.replace(hour=21, minute=30),
                         ethanol_g=per_drink, source="synthetic"))
    return out


def start_state() -> PatientState:
    return PatientState(t=REF_WEEK_START, fast=default_fast(140.0), daily=DailyState())


def week_mean_G(sim: RealSimulator, template: list[Event], params: PatientParams) -> float:
    from engine.longrun import _run_week
    return _run_week(sim, start_state(), template, params, DailyState()).mean_G


def find_insulin_scale(sim, params, target_mean_G: float, weekly_ethanol_g: float,
                       lo: float = 0.02, hi: float = 1.2, tol: float = 0.5) -> float:
    """Bisect total insulin until the simulated week hits the target mean glucose."""
    for _ in range(40):
        mid = (lo + hi) / 2.0
        got = week_mean_G(sim, reference_template(mid, weekly_ethanol_g), params)
        if abs(got - target_mean_G) < tol:
            return mid
        if got > target_mean_G:      # too high: needs more insulin
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def raw_damage(cfg: dict, params: PatientParams, template: list[Event]) -> dict[str, float]:
    """Ten-year damage with every k_<organ> set to 1."""
    unit = dict(cfg)
    for organ in ORGANS:
        unit[f"k_{organ}"] = 1.0
    sim = RealSimulator(cfg=unit)
    lt = sim.simulate_long(start_state(), template, params, 10.0)
    return {organ: float(lt.damage[organ][-1]) for organ in ORGANS}


def solve_hazard(risk7: float, risk9: float, d7: float, d9: float) -> tuple[float, float]:
    A = -math.log(1.0 - risk7)
    B = -math.log(1.0 - risk9)
    if abs(d9 - d7) < 1e-9:
        return A * math.exp(-1.0 * d7), 1.0
    beta = math.log(B / A) / (d9 - d7)
    return A * math.exp(-beta * d7), beta


def main(write: bool) -> None:
    doc = load_params()
    cfg = dict(doc["defaults"])
    targets = doc["risk_targets"]
    params = PatientParams()
    sim = RealSimulator(cfg=cfg)

    ethanol = float(targets["weekly_ethanol_g"])
    scale7 = find_insulin_scale(sim, params, float(targets["mean_G_reference"]), ethanol)
    scale9 = find_insulin_scale(sim, params, float(targets["mean_G_high"]), ethanol)
    t7 = reference_template(scale7, ethanol)
    t9 = reference_template(scale9, ethanol)
    print(f"reference patient : insulin x{scale7:.4f} -> mean_G {week_mean_G(sim, t7, params):.1f}")
    print(f"high-A1c patient  : insulin x{scale9:.4f} -> mean_G {week_mean_G(sim, t9, params):.1f}")

    raw7 = raw_damage(cfg, params, t7)
    raw9 = raw_damage(cfg, params, t9)

    out: dict[str, float] = {}
    print(f"\n{'organ':8s} {'k':>12s} {'D7':>6s} {'D9':>7s} {'hazard':>9s} {'beta':>7s} "
          f"{'risk7':>7s} {'risk9':>7s}")
    for organ in ORGANS:
        r7 = float(targets["risk10"][organ])
        r9 = min(r7 * float(targets["high_risk_ratio"][organ]), 0.95)
        if raw7[organ] <= 0.0:
            raise SystemExit(f"{organ}: reference patient accumulates no damage; "
                             "the calibration scenario cannot normalise it")
        k = 1.0 / raw7[organ]
        d7, d9 = 1.0, raw9[organ] * k
        h, beta = solve_hazard(r7, r9, d7, d9)
        out[f"k_{organ}"] = k
        out[f"baseline_hazard_{organ}"] = h
        out[f"beta_{organ}"] = beta
        check7 = 1 - math.exp(-h * math.exp(beta * d7))
        check9 = 1 - math.exp(-h * math.exp(beta * d9))
        print(f"{organ:8s} {k:12.6g} {d7:6.2f} {d9:7.3f} {h:9.5f} {beta:7.3f} "
              f"{check7:7.4f} {check9:7.4f}")

    if not write:
        print("\n(dry run - pass --write to patch engine/params.yaml)")
        return

    text = PARAMS_PATH.read_text(encoding="utf-8")
    for name, value in out.items():
        import re
        pattern = re.compile(rf"^(  {re.escape(name)}: )([-\d.eE+]+)(.*)$", re.MULTILINE)
        if not pattern.search(text):
            raise SystemExit(f"{name} not found in params.yaml")
        text = pattern.sub(lambda m: f"{m.group(1)}{value:.6g}{m.group(3)}", text, count=1)
    PARAMS_PATH.write_text(text, encoding="utf-8")
    print(f"\nwrote {len(out)} constants to {PARAMS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="patch engine/params.yaml in place")
    main(ap.parse_args().write)
