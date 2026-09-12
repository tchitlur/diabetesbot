"""Ten-year projection: medium and slow layers stepped weekly.

Simulating 520 weeks at five-minute resolution would take minutes. Instead one
representative week is simulated, its aggregates drive the weekly medium/slow steps,
and the week is only re-simulated when insulin sensitivity or body composition has
actually moved. That bounds a call to ~20 fast simulations.

Organ damage accumulates weekly; the ten-year risks come from a Cox-style hazard
calibrated to the DCCT targets in engine/params.yaml. engine/calibrate_risk.py derives
those constants and writes them back.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np

from schema.events import Event
from schema.params import PatientParams
from schema.simulate import LongTrajectory, REF_WEEK_START
from schema.state import FAST_IDX, ORGANS, DailyState, PatientState, SlowState

_G = FAST_IDX["G"]
_HYD = FAST_IDX["hyd"]

WEEKS_PER_YEAR = 52.0
QUARTER = 0.25


class WeekStats:
    """Aggregates of one simulated reference week."""

    __slots__ = ("mean_G", "sd_G", "tbr_pct", "hypo_per_week", "mean_hyd")

    def __init__(self, mean_G, sd_G, tbr_pct, hypo_per_week, mean_hyd):
        self.mean_G = mean_G
        self.sd_G = sd_G
        self.tbr_pct = tbr_pct
        self.hypo_per_week = hypo_per_week
        self.mean_hyd = mean_hyd


def _weekday_events(template: list[Event]) -> list[list[Event]]:
    """Split the reference week into seven day buckets for the daily update."""
    buckets: list[list[Event]] = [[] for _ in range(7)]
    for ev in template:
        d = (ev.t_start - REF_WEEK_START).days
        if 0 <= d < 7:
            buckets[d].append(ev)
    return buckets


def _hypo_episodes(G: np.ndarray, threshold: float, min_minutes: float, dt_min: float) -> float:
    """Runs below threshold lasting at least min_minutes, per the consensus definition."""
    need = max(1, int(round(min_minutes / dt_min)))
    low = G < threshold
    count = 0
    run = 0
    for flag in low:
        if flag:
            run += 1
            if run == need:
                count += 1
        else:
            run = 0
    return float(count)


def _run_week(sim, state: PatientState, template: list[Event], params: PatientParams,
              daily: DailyState) -> WeekStats:
    cfg = sim.cfg
    dt = int(cfg["long_week_dt_min"])
    ref = PatientState(t=REF_WEEK_START, fast=np.asarray(state.fast, dtype=float).copy(),
                       daily=replace(daily), slow=replace(state.slow))
    traj = sim.simulate(ref, template, params, 7 * 1440, dt)
    G = traj.fast[:, _G]
    return WeekStats(
        mean_G=float(G.mean()),
        sd_G=float(G.std()),
        tbr_pct=float(100.0 * np.mean(G < cfg["hypo_threshold"])),
        hypo_per_week=_hypo_episodes(G, cfg["hypo_threshold"], cfg["hypo_episode_min"], dt),
        mean_hyd=float(traj.fast[:, _HYD].mean()),
    )


def _weekly_damage(cfg: dict, profile, stats: WeekStats, daily: DailyState,
                   weekly_ethanol_g: float) -> dict[str, float]:
    """One week's increment to each organ accumulator."""
    weight = max(daily.fat_mass_kg + daily.lean_mass_kg, 1.0)
    hyper = max(0.0, stats.mean_G - cfg["organ_G_threshold"])
    adiposity = max(0.0, daily.fat_mass_kg / weight - cfg["body_fat_ref"])
    return {
        "eye": cfg["k_eye"] * (hyper + cfg["var_gain"] * stats.sd_G),
        "kidney": cfg["k_kidney"] * (hyper
                                     + cfg["bp_gain"] * max(0.0, profile.systolic_bp - 130.0)
                                     + cfg["dehyd_gain"] * max(0.0, 1.0 - stats.mean_hyd)),
        "nerve": cfg["k_nerve"] * (hyper + cfg["alc_nerve_gain"] * weekly_ethanol_g),
        "cardio": cfg["k_cardio"] * (hyper
                                     + cfg["fat_gain"] * adiposity
                                     + cfg["alc_cardio_gain"] * weekly_ethanol_g
                                     + cfg["sleep_gain"] * daily.sleep_debt_h),
        "liver": cfg["k_liver"] * (cfg["alc_liver_gain"] * weekly_ethanol_g
                                   + cfg["fat_liver_gain"] * adiposity),
    }


def risk10(cfg: dict, damage_10y: dict[str, float]) -> dict[str, float]:
    """1 - exp(-baseline_hazard * exp(beta * damage)), per organ."""
    out = {}
    for organ in ORGANS:
        h = float(cfg[f"baseline_hazard_{organ}"])
        beta = float(cfg[f"beta_{organ}"])
        d = float(damage_10y.get(organ, 0.0))
        # Exponent capped so an absurd damage figure saturates at certainty instead of
        # overflowing. engine/calibrate_risk.py deliberately runs with k_organ = 1, which
        # makes the raw damage enormous.
        expo = min(beta * d, 50.0)
        out[organ] = float(np.clip(1.0 - np.exp(-h * np.exp(expo)), 0.0, 1.0))
    return out


def simulate_long(sim, state: PatientState, weekly_template: list[Event],
                  params: PatientParams, years: float = 10.0) -> LongTrajectory:
    cfg = sim.cfg
    profile = sim.profile
    template = list(weekly_template)
    buckets = _weekday_events(template)
    weekly_ethanol = sum(e.ethanol_g for e in template if e.type == "alcohol")

    n_weeks = max(1, int(round(years * WEEKS_PER_YEAR)))
    tol = float(cfg["long_resim_tol"])
    max_resim = int(cfg["long_max_resim"])

    daily = replace(state.daily)
    slow = replace(state.slow)
    damage = {organ: float(getattr(slow, organ)) for organ in ORGANS}

    stats = _run_week(sim, state, template, params, daily)
    n_resim = 1
    anchor_si = daily.si_mult
    anchor_mass = daily.fat_mass_kg + daily.lean_mass_kg

    # quarterly output grid
    n_out = int(round(years / QUARTER)) + 1
    yrs = np.arange(n_out, dtype=float) * QUARTER
    out_week = np.rint(yrs * WEEKS_PER_YEAR).astype(int)
    hba1c = np.empty(n_out)
    mean_G = np.empty(n_out)
    hypo = np.empty(n_out)
    damage_hist = {organ: np.empty(n_out) for organ in ORGANS}

    cursor = 0

    def record(week: int) -> None:
        nonlocal cursor
        while cursor < n_out and out_week[cursor] <= week:
            hba1c[cursor] = daily.hba1c_pct
            mean_G[cursor] = stats.mean_G
            hypo[cursor] = stats.hypo_per_week
            for organ in ORGANS:
                damage_hist[organ][cursor] = damage[organ]
            cursor += 1

    record(0)
    for week in range(1, n_weeks + 1):
        # re-simulate only when the patient has actually changed
        mass = daily.fat_mass_kg + daily.lean_mass_kg
        if n_resim < max_resim and (
                abs(daily.si_mult - anchor_si) > tol * max(abs(anchor_si), 1e-6)
                or abs(mass - anchor_mass) > tol * max(anchor_mass, 1e-6)):
            stats = _run_week(sim, state, template, params, daily)
            n_resim += 1
            anchor_si, anchor_mass = daily.si_mult, mass

        inc = _weekly_damage(cfg, profile, stats, daily, weekly_ethanol)
        for organ in ORGANS:
            damage[organ] += inc[organ]

        for d in range(7):
            daily = sim.daily_update(daily, buckets[d], params, stats.mean_G,
                                     stats.mean_hyd, REF_WEEK_START + timedelta(days=d))
        record(week)

    while cursor < n_out:                      # years landed exactly on the last week
        record(n_weeks)

    # The hazard is calibrated on ten years of accumulation. A shorter horizon is scaled
    # up to that basis so risk10 keeps meaning "probability within ten years".
    scale = 10.0 / years if years > 0 else 1.0
    damage_10y = {organ: (damage[organ] - getattr(state.slow, organ)) * scale
                  + getattr(state.slow, organ) for organ in ORGANS}

    return LongTrajectory(
        years=yrs,
        hba1c=hba1c,
        mean_G=mean_G,
        hypo_per_week=hypo,
        damage=damage_hist,
        risk10=risk10(cfg, damage_10y),
    )


def final_slow(state: PatientState, damage: dict[str, float]) -> SlowState:
    return SlowState(**{organ: float(damage[organ]) for organ in ORGANS})
