"""Placeholder simulator. Plausible-looking, deterministic, instant.

No real physiology lives here: it exists only so that the API, the decision engine,
the frontend, and the fixtures can be built before engine/simulate.py lands.
schema.simulate.get_simulator() prefers engine.simulate.RealSimulator and falls back
to StubSimulator, so deleting this file's callers is never necessary.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from schema.events import Event
from schema.params import PatientParams
from schema.simulate import LongTrajectory, REF_WEEK_START, Trajectory
from schema.state import FAST_IDX, N_FAST, ORGANS, PatientState, energy_score

# Shape constants for the stub's made-up curves.
MEAL_GAIN = 3.0            # mg/dL per gram of carbohydrate at the peak of the bump
MEAL_TAU_MIN = 90.0        # time to peak of the carb bump
BOLUS_TAU_MIN = 120.0      # time to peak (nadir) of the insulin bump
# DEVIATION FROM THE SPEC, and the only one. The contract sheet prescribed the insulin term
# as si_day*units*bump(dt/180). With the population defaults (si_day=40, carb_ratio=10) a
# perfectly carb-matched bolus then removes 240 mg/dL against a 180 mg/dL meal rise, on a
# slower curve, so the specified fixture day collapses onto the 40 mg/dL clip and stays
# there: 85% below range, a flat line for the frontend to render. Damping and shortening the
# insulin curve makes a matched bolus roughly cancel its meal, which is what "plausible-
# looking" needs. Set BOLUS_EFFECT_SCALE = 1.0 and BOLUS_TAU_MIN = 180.0 for the literal
# formula. Nothing outside this file depends on either number.
BOLUS_EFFECT_SCALE = 0.5
EXERCISE_DROP = 25.0       # mg/dL at full effect
EXERCISE_DECAY_MIN = 240.0 # post-exercise decay time constant
ALCOHOL_DROP = 30.0        # mg/dL per standard drink at the peak of the delayed dip
ALCOHOL_DELAY_MIN = 300.0  # alcohol dip starts ~5 h after the drink
ALCOHOL_TAU_MIN = 240.0    # time from delay to peak of the dip
STD_DRINK_G = 14.0         # one US standard drink, grams of ethanol
INSULIN_HALFLIFE_MIN = 60.0
INSULIN_PER_UNIT = 10.0    # mU/L added to plasma insulin per unit bolused
CAFFEINE_HALFLIFE_MIN = 300.0
BAC_PER_DRINK = 0.02       # g/dL per standard drink
BAC_CLEAR_PER_H = 0.015    # zero-order clearance
G_MIN, G_MAX = 40.0, 400.0
TARGET_LOW, TARGET_HIGH = 70.0, 180.0
KETONES_FIXED = 0.1
HYDRATION_FIXED = 1.0

BASE_RISK10 = {"eye": 0.12, "kidney": 0.08, "nerve": 0.10, "cardio": 0.09, "liver": 0.03}
REFERENCE_MEAN_G = 154.0   # mean glucose corresponding to the baseline risks above


def bump(x: float | np.ndarray) -> np.ndarray:
    """x*exp(1-x) for x>0, else 0. Peaks at exactly 1.0 when x == 1."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x * np.exp(1.0 - x), 0.0)


def _minutes_since(grid: np.ndarray, t0: datetime, t_start: datetime) -> np.ndarray:
    """Minutes from t_start to each grid point (grid is minutes since t0)."""
    return grid - (t_start - t0).total_seconds() / 60.0


def _is_bolus(ev: Event) -> bool:
    return ev.type == "insulin" and ev.insulin_kind != "basal"


def _exercise_factor(dt: np.ndarray, duration_min: float) -> np.ndarray:
    """1.0 while exercising, then exponential decay over EXERCISE_DECAY_MIN. 0 before the event."""
    after = np.clip(dt - max(duration_min, 0.0), 0.0, None)
    f = np.where(dt >= 0.0, np.exp(-after / EXERCISE_DECAY_MIN), 0.0)
    return np.clip(f, 0.0, 1.0)


class StubSimulator:
    """Satisfies schema.simulate.Simulator."""

    def simulate(self, state: PatientState, events: list[Event], params: PatientParams,
                 horizon_min: int, dt_min: int = 5) -> Trajectory:
        T = int(horizon_min // dt_min) + 1
        grid = np.arange(T, dtype=float) * dt_min           # minutes since state.t
        t0 = state.t
        times = [t0 + timedelta(minutes=float(m)) for m in grid]

        x0 = np.asarray(state.fast, dtype=float)
        fast = np.zeros((T, N_FAST))

        # --- glucose -------------------------------------------------------
        # Carb-bearing events (meals, and drinks that carry carbs) push G up;
        # boluses, exercise and alcohol pull it down.
        G = np.full(T, x0[FAST_IDX["G"]])
        D1 = np.zeros(T)
        D2 = np.zeros(T)
        I_p = x0[FAST_IDX["I_p"]] * 0.5 ** (grid / INSULIN_HALFLIFE_MIN)
        caf = x0[FAST_IDX["caf"]] * 0.5 ** (grid / CAFFEINE_HALFLIFE_MIN)
        ex = np.zeros(T)
        k_abs = max(float(params.k_abs), 1e-6)

        # gut contents already on board at t0 decay through the same two compartments
        D1 += x0[FAST_IDX["D1"]] * np.exp(-k_abs * grid)
        D2 += (x0[FAST_IDX["D2"]] * np.exp(-k_abs * grid)
               + x0[FAST_IDX["D1"]] * k_abs * grid * np.exp(-k_abs * grid))

        for ev in events:
            dt = _minutes_since(grid, t0, ev.t_start)
            past = dt >= 0.0

            if ev.carbs_g > 0:
                G = G + MEAL_GAIN * ev.carbs_g * bump(dt / MEAL_TAU_MIN)
                tau = np.clip(dt, 0.0, None)
                D1 = D1 + np.where(past, ev.carbs_g * np.exp(-k_abs * tau), 0.0)
                D2 = D2 + np.where(past, ev.carbs_g * k_abs * tau * np.exp(-k_abs * tau), 0.0)

            if _is_bolus(ev) and ev.units > 0:
                G = G - BOLUS_EFFECT_SCALE * params.si_day * ev.units * bump(dt / BOLUS_TAU_MIN)
                I_p = I_p + np.where(
                    past, INSULIN_PER_UNIT * ev.units * 0.5 ** (np.clip(dt, 0.0, None) / INSULIN_HALFLIFE_MIN), 0.0
                )

            if ev.type == "exercise":
                f = _exercise_factor(dt, ev.duration_min)
                G = G - EXERCISE_DROP * f
                ex = np.maximum(ex, f)

            if ev.type == "alcohol" and ev.ethanol_g > 0:
                G = G - ALCOHOL_DROP * (ev.ethanol_g / STD_DRINK_G) * bump(
                    (dt - ALCOHOL_DELAY_MIN) / ALCOHOL_TAU_MIN
                )

            if ev.type == "caffeine" and ev.caffeine_mg > 0:
                caf = caf + np.where(
                    past, ev.caffeine_mg * 0.5 ** (np.clip(dt, 0.0, None) / CAFFEINE_HALFLIFE_MIN), 0.0
                )

        G = np.clip(G, G_MIN, G_MAX)

        # --- blood alcohol: zero-order clearance, so step it on the grid ----
        BAC = np.zeros(T)
        dt_h = dt_min / 60.0
        drinks = [(ev.t_start, BAC_PER_DRINK * ev.ethanol_g / STD_DRINK_G)
                  for ev in events if ev.type == "alcohol" and ev.ethanol_g > 0]
        bac = float(x0[FAST_IDX["BAC"]]) + sum(a for ts, a in drinks if ts <= times[0])
        BAC[0] = max(bac, 0.0)
        for i in range(1, T):
            bac = max(bac - BAC_CLEAR_PER_H * dt_h, 0.0)
            bac += sum(a for ts, a in drinks if times[i - 1] < ts <= times[i])
            BAC[i] = bac

        fast[:, FAST_IDX["G"]] = G
        fast[:, FAST_IDX["I_p"]] = I_p
        fast[:, FAST_IDX["D1"]] = D1
        fast[:, FAST_IDX["D2"]] = D2
        fast[:, FAST_IDX["gly_liver"]] = x0[FAST_IDX["gly_liver"]]
        fast[:, FAST_IDX["gly_muscle"]] = x0[FAST_IDX["gly_muscle"]]
        fast[:, FAST_IDX["BAC"]] = BAC
        fast[:, FAST_IDX["caf"]] = caf
        fast[:, FAST_IDX["ket"]] = KETONES_FIXED
        fast[:, FAST_IDX["hyd"]] = HYDRATION_FIXED
        fast[:, FAST_IDX["ex"]] = ex

        energy = np.array([energy_score(fast[i], state.daily) for i in range(T)])

        return Trajectory(
            t=times,
            fast=fast,
            energy=energy,
            daily_end=replace(state.daily),
            slow_end=replace(state.slow),
            summary=_summary(G, times),
        )

    def simulate_long(self, state: PatientState, weekly_template: list[Event], params: PatientParams,
                      years: float = 10.0) -> LongTrajectory:
        week = self._reference_week(state, weekly_template, params)
        Gw = week.fast[:, FAST_IDX["G"]]
        mean_G = float(np.mean(Gw))

        n = int(round(years / 0.25)) + 1
        yrs = np.arange(n, dtype=float) * 0.25

        hba1c = np.full(n, 0.0296 * mean_G + 2.42)
        mean_G_arr = np.full(n, mean_G)
        hypo_per_week = np.full(n, float(_hypo_episodes(Gw)))

        scale = mean_G / REFERENCE_MEAN_G
        risk10 = {organ: float(np.clip(BASE_RISK10[organ] * scale, 0.0, 1.0)) for organ in ORGANS}
        damage = {organ: risk10[organ] * (yrs / 10.0) for organ in ORGANS}

        return LongTrajectory(
            years=yrs,
            hba1c=hba1c,
            mean_G=mean_G_arr,
            hypo_per_week=hypo_per_week,
            damage=damage,
            risk10=risk10,
        )

    def _reference_week(self, state: PatientState, weekly_template: list[Event],
                        params: PatientParams) -> Trajectory:
        """Run the weekly template once over the reference week at 30-minute resolution."""
        ref_state = PatientState(
            t=REF_WEEK_START,
            fast=np.asarray(state.fast, dtype=float).copy(),
            daily=replace(state.daily),
            slow=replace(state.slow),
        )
        return self.simulate(ref_state, list(weekly_template), params, horizon_min=7 * 24 * 60, dt_min=30)


def _summary(G: np.ndarray, times: list[datetime]) -> dict:
    n = len(G)
    i_min = int(np.argmin(G))
    return {
        "tir_pct": float(100.0 * np.mean((G >= TARGET_LOW) & (G <= TARGET_HIGH))) if n else 0.0,
        "tbr70_pct": float(100.0 * np.mean(G < TARGET_LOW)) if n else 0.0,
        "tar180_pct": float(100.0 * np.mean(G > TARGET_HIGH)) if n else 0.0,
        "min_G": float(np.min(G)) if n else 0.0,
        "max_G": float(np.max(G)) if n else 0.0,
        "t_min_G": times[i_min].isoformat() if n else "",
        "mean_G": float(np.mean(G)) if n else 0.0,
    }


def _hypo_episodes(G: np.ndarray) -> int:
    """Number of distinct excursions below TARGET_LOW (down-crossings)."""
    low = G < TARGET_LOW
    if not low.any():
        return 0
    return int(low[0]) + int(np.sum(low[1:] & ~low[:-1]))
