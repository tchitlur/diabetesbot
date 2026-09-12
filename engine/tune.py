"""Sweep population constants within their declared ranges and report what still holds.

Every constant in engine/params.yaml has a `ranges` entry saying how far it could
plausibly move. This walks one constant at a time across its range and reports which
values keep all of the clinical shape targets satisfied - the same targets
engine/test_smoke.py asserts, restated here so the tuner does not depend on pytest.

    python engine/tune.py                    # every constant that has a range
    python engine/tune.py --param V_g        # just one
    python engine/tune.py --points 9         # finer sweep

A constant whose whole range passes is one the shapes do not constrain; a constant with
a narrow passing window is one to be careful with. Both are useful to know before
changing anything.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.simulate import RealSimulator, load_params  # noqa: E402
from schema.events import Event  # noqa: E402
from schema.params import PatientParams  # noqa: E402
from schema.state import DailyState, FAST_IDX, PatientState, default_fast  # noqa: E402

T0 = datetime(2026, 9, 11, 7, 0)
GI = FAST_IDX["G"]
PARAMS = PatientParams()

# name -> (inclusive lower bound, inclusive upper bound)
TARGETS = {
    "meal_peak": (220, 300),        # 60 g unbolused, mg/dL
    "meal_peak_min": (60, 120),     # minutes to that peak
    "bolused_peak": (0, 200),
    "bolused_4h": (80, 140),
    "bolused_min": (60, 1e9),
    "correction_nadir": (40, 70),
    "correction_min": (90, 180),
    "aerobic_drop": (40, 80),
    "anaerobic_rise_20m": (0.5, 1e9),
    "anaerobic_fall": (5, 1e9),
    "alcohol_midnight_gap": (0, 20),
    "alcohol_overnight_gap": (30, 1e9),
    "caffeine_peak_shift": (10, 30),
    "resting_drift": (0, 15),       # mg/dL of wander on basal alone
}


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _basal(hours: float = 30) -> list[Event]:
    return [Event(type="insulin", t_start=T0, duration_min=hours * 60, units=1.0,
                  insulin_kind="basal")]


def _meal(carbs: float, minutes: float = 30) -> Event:
    return Event(type="meal", t_start=_at(minutes), carbs_g=carbs)


def _bolus(units: float, minutes: float = 30) -> Event:
    return Event(type="insulin", t_start=_at(minutes), units=units, insulin_kind="bolus")


def _fed_day() -> list[Event]:
    out: list[Event] = []
    for minutes, carbs, units in ((30, 60, 6), (360, 70, 7), (720, 80, 8)):
        out += [_meal(carbs, minutes), _bolus(units, minutes)]
    return out


def measure(cfg: dict) -> dict[str, float]:
    """Run every scenario once and return the numbers the targets are stated in."""
    sim = RealSimulator(cfg=cfg)

    def G(events, G0, hours, dt=5):
        st = PatientState(t=T0, fast=default_fast(G0), daily=DailyState())
        return sim.simulate(st, events, PARAMS, int(hours * 60), dt).fast[:, GI]

    out: dict[str, float] = {}
    g = G([_meal(60)], 100.0, 6)
    out["meal_peak"] = float(g.max())
    out["meal_peak_min"] = float(int(np.argmax(g)) * 5 - 30)

    g = G([_meal(60), _bolus(6)] + _basal(), 100.0, 8)
    out["bolused_peak"] = float(g.max())
    out["bolused_4h"] = float(g[(4 * 60) // 5])
    out["bolused_min"] = float(g.min())

    g = G([_bolus(6, 0)] + _basal(), 120.0, 8)
    out["correction_nadir"] = float(g.min())
    out["correction_min"] = float(int(np.argmin(g)) * 5)

    g = G([Event(type="exercise", t_start=T0, duration_min=45, intensity=0.6,
                 exercise_kind="aerobic")] + _basal(), 140.0, 3)
    out["aerobic_drop"] = float(140.0 - g[45 // 5])

    g = G([Event(type="exercise", t_start=T0, duration_min=45, intensity=0.9,
                 exercise_kind="anaerobic")] + _basal(), 140.0, 3)
    out["anaerobic_rise_20m"] = float(g[20 // 5] - 140.0)
    out["anaerobic_fall"] = float(g[45 // 5] - g[45 // 5:].min())

    drinks = [Event(type="alcohol", t_start=_at(15 * 60), ethanol_g=56)]
    wet = G(_fed_day() + _basal() + drinks, 120.0, 25)
    dry = G(_fed_day() + _basal(), 120.0, 25)
    i_mid, i_02, i_08 = (17 * 60) // 5, (19 * 60) // 5, (25 * 60) // 5
    out["alcohol_midnight_gap"] = float(abs(wet[i_mid] - dry[i_mid]))
    out["alcohol_overnight_gap"] = float(dry[i_02:i_08 + 1].min() - wet[i_02:i_08 + 1].min())

    caf = Event(type="caffeine", t_start=_at(30), caffeine_mg=200)
    out["caffeine_peak_shift"] = float(
        G([_meal(60), _bolus(6), caf] + _basal(), 100.0, 6).max()
        - G([_meal(60), _bolus(6)] + _basal(), 100.0, 6).max())

    g = G(_basal(14), 100.0, 12)
    out["resting_drift"] = float(max(abs(g.max() - 100.0), abs(g.min() - 100.0)))
    return out


def failures(values: dict[str, float]) -> list[str]:
    bad = []
    for name, got in values.items():
        lo, hi = TARGETS[name]
        if not (lo <= got <= hi):
            bad.append(name)
    return bad


def sweep(name: str, points: int) -> None:
    doc = load_params()
    base = dict(doc["defaults"])
    lo, hi = doc["ranges"][name]
    grid = np.linspace(float(lo), float(hi), points)
    current = base[name]
    print(f"\n{name}  range [{lo}, {hi}]  currently {current}")
    passing = []
    for value in grid:
        cfg = dict(base)
        cfg[name] = float(value)
        if name == "k_ins":
            # k_u2mUL and k_si are derived from k_ins; sweeping it alone would break the
            # calibration rather than test it. See engine/NOTES.md.
            cfg["k_u2mUL"] = 600.0 * float(value)
            cfg["k_si"] = 1.0 / 600.0
        try:
            bad = failures(measure(cfg))
        except Exception as exc:                    # a wild value can break the integrator
            print(f"   {value:12.6g}  ERROR {type(exc).__name__}")
            continue
        if bad:
            print(f"   {value:12.6g}  fails: {', '.join(bad)}")
        else:
            print(f"   {value:12.6g}  ok")
            passing.append(float(value))
    if not passing:
        print("   -> nothing in this range satisfies every target")
    elif len(passing) == points:
        print("   -> the whole range passes; the shape targets do not constrain this one")
    else:
        print(f"   -> passes on [{min(passing):.6g}, {max(passing):.6g}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", help="sweep only this constant")
    ap.add_argument("--points", type=int, default=7, help="grid points per constant")
    args = ap.parse_args()

    doc = load_params()
    baseline = measure(dict(doc["defaults"]))
    bad = failures(baseline)
    print("current defaults:")
    for name, got in baseline.items():
        lo, hi = TARGETS[name]
        flag = " " if lo <= got <= hi else "X"
        print(f"  {flag} {name:24s} {got:9.2f}   target [{lo}, {hi}]")
    if bad:
        print(f"\nWARNING: the shipped defaults already fail: {', '.join(bad)}")

    names = [args.param] if args.param else sorted(doc["ranges"])
    for name in names:
        if name not in doc["ranges"]:
            raise SystemExit(f"{name} has no range in params.yaml")
        sweep(name, args.points)


if __name__ == "__main__":
    main()
