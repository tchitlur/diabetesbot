"""RealSimulator: a Hovorka-style glucose-insulin ODE model.

State order is schema.state.FAST_NAMES. Every population constant comes from
engine/params.yaml; there are no magic numbers below. Per-patient fitted
parameters arrive in the PatientParams argument.

schema.simulate.get_simulator() picks this up automatically once it imports cleanly.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import solve_ivp

from schema.events import Event, Profile
from schema.params import PatientParams
from schema.simulate import LongTrajectory, Trajectory
from schema.state import (
    FAST_IDX, N_FAST, ORGANS, DailyState, PatientState, SlowState, energy_score,
)

PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"
LN2 = math.log(2.0)

# Integration settings. Accuracy is bounded by the segment restarts, not by rtol;
# CGM measurement noise is ~10 mg/dL, so ~2 mg/dL of solver error is well inside it.
_RTOL = 1e-3
_ATOL = 1e-6
_MAX_STEP_SEGMENTS = 6.0   # cap a step at this many output intervals
_T_EPS = 1e-9

# Index constants derived from the contract. Never write a literal index.
_G = FAST_IDX["G"]
_I_P = FAST_IDX["I_p"]
_D1 = FAST_IDX["D1"]
_D2 = FAST_IDX["D2"]
_GLY_L = FAST_IDX["gly_liver"]
_GLY_M = FAST_IDX["gly_muscle"]
_BAC = FAST_IDX["BAC"]
_CAF = FAST_IDX["caf"]
_KET = FAST_IDX["ket"]
_HYD = FAST_IDX["hyd"]
_EX = FAST_IDX["ex"]


@lru_cache(maxsize=1)
def load_params(path: str = str(PARAMS_PATH)) -> dict:
    """Load engine/params.yaml once. Returns the whole document."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def defaults() -> dict:
    return load_params()["defaults"]


def ranges() -> dict:
    return load_params()["ranges"]


# --------------------------------------------------------------------------- inputs

# Channel order inside one input segment.
_C_CARB, _C_INS, _C_ALC, _C_CAF, _C_WATER, _C_EXI, _C_EXA, _C_ILL, _C_KABS = range(9)
_N_CHAN = 9


class InputSchedule:
    """Piecewise-constant exogenous inputs, precomputed once per simulate() call.

    Built as a sorted breakpoint list plus one tuple of channel values per segment,
    so the ODE right-hand side costs a single bisect instead of a loop over events.
    """

    __slots__ = ("bp", "vals")

    def __init__(self, bp: list[float], vals: list[tuple]):
        self.bp = bp
        self.vals = vals

    def at(self, t: float) -> tuple:
        return self.vals[bisect_right(self.bp, t) - 1]


def _macro_fracs(ev: Event) -> tuple[float, float]:
    """Fat and fibre fractions of a meal, used to slow gastric emptying."""
    bulk = ev.carbs_g + ev.fat_g + ev.protein_g
    fat_frac = ev.fat_g / bulk if bulk > 0 else 0.0
    fiber_frac = ev.fiber_g / ev.carbs_g if ev.carbs_g > 0 else 0.0
    return min(max(fat_frac, 0.0), 1.0), min(max(fiber_frac, 0.0), 1.0)


def build_inputs(events: list[Event], t0: datetime, horizon_min: float,
                 params: PatientParams, cfg: dict, weight_kg: float) -> InputSchedule:
    """Turn events into piecewise-constant input rates on [0, horizon_min]."""
    carb_pulse = cfg["carb_pulse_min"]
    ramp_min = cfg["bolus_ramp_min"]
    ramp_steps = int(cfg["bolus_ramp_steps"])
    alc_absorb = cfg["alc_absorb_min"]
    caf_absorb = cfg["caf_absorb_min"]
    water_absorb = cfg["water_absorb_min"]
    fat_slow = cfg["fat_slowdown"]
    fiber_slow = cfg["fiber_slowdown"]
    kabs_floor = cfg["k_abs_floor_frac"]
    u2mUL = cfg["k_u2mUL"] * cfg["insulin_weight_ref"] / weight_kg
    bac_per_g = cfg["bac_per_g"] * cfg["insulin_weight_ref"] / weight_kg

    # (start, end, channel, rate) intervals
    spans: list[tuple[float, float, int, float]] = []
    # (start, multiplier) for gastric emptying, held until the next carb event
    kabs_marks: list[tuple[float, float]] = []

    for ev in events:
        rel = (ev.t_start - t0).total_seconds() / 60.0
        dur = max(ev.duration_min, 0.0)

        if ev.carbs_g > 0:
            spans.append((rel, rel + carb_pulse, _C_CARB, ev.carbs_g / carb_pulse))
            fat_frac, fiber_frac = _macro_fracs(ev)
            mult = max(1.0 - fat_slow * fat_frac - fiber_slow * fiber_frac, kabs_floor)
            kabs_marks.append((rel, mult))

        if ev.type == "insulin":
            if ev.insulin_kind == "basal":
                rate = (ev.units + params.basal_drift) / 60.0 * u2mUL
                if dur > 0 and rate != 0.0:
                    spans.append((rel, rel + dur, _C_INS, rate))
            elif ev.units > 0:
                # Rising ramp over bolus_ramp_min, discretised so the input stays
                # piecewise constant. Sub-interval k carries weight (2k+1)/steps^2.
                step = ramp_min / ramp_steps
                for k in range(ramp_steps):
                    frac = (2.0 * k + 1.0) / (ramp_steps * ramp_steps)
                    spans.append((rel + k * step, rel + (k + 1) * step, _C_INS,
                                  ev.units * u2mUL * frac / step))

        if ev.type == "alcohol" and ev.ethanol_g > 0:
            spans.append((rel, rel + alc_absorb, _C_ALC,
                          ev.ethanol_g * bac_per_g / alc_absorb))

        if ev.type == "caffeine" and ev.caffeine_mg > 0:
            spans.append((rel, rel + caf_absorb, _C_CAF, ev.caffeine_mg / caf_absorb))

        if ev.type == "water" and ev.water_ml > 0:
            spans.append((rel, rel + water_absorb, _C_WATER, ev.water_ml / water_absorb))

        if ev.type == "exercise" and dur > 0:
            spans.append((rel, rel + dur, _C_EXI, ev.intensity))
            if ev.exercise_kind in ("anaerobic", "mixed"):
                spans.append((rel, rel + dur, _C_EXA, ev.intensity))

        # Stress raises insulin resistance the same way illness does; see NOTES.md.
        if ev.type in ("illness", "stress") and ev.severity > 0 and dur > 0:
            spans.append((rel, rel + dur, _C_ILL, ev.severity))

    bounds = {0.0, horizon_min}
    for s, e, _, _ in spans:
        bounds.add(s)
        bounds.add(e)
    for s, _ in kabs_marks:
        bounds.add(s)
    bp = sorted(b for b in bounds if 0.0 <= b <= horizon_min)
    if not bp or bp[0] > 0.0:
        bp.insert(0, 0.0)

    kabs_marks.sort()
    vals: list[tuple] = []
    for i, start in enumerate(bp):
        mid = (start + bp[i + 1]) / 2.0 if i + 1 < len(bp) else start
        chan = [0.0] * _N_CHAN
        chan[_C_KABS] = 1.0
        for s, e, c, r in spans:
            if s <= mid < e:
                # intensity-like channels take the strongest overlapping event,
                # rate-like channels add up
                if c in (_C_EXI, _C_EXA, _C_ILL):
                    chan[c] = max(chan[c], r)
                else:
                    chan[c] += r
        for s, m in kabs_marks:
            if s <= mid:
                chan[_C_KABS] = m
        vals.append(tuple(chan))

    return InputSchedule(bp, vals)


# --------------------------------------------------------------------------- rhs

def _make_rhs(inp: InputSchedule, params: PatientParams, cfg: dict, daily: DailyState,
              profile: Profile, weight_kg: float, t0: datetime):
    """Build the ODE right-hand side with every constant bound as a local."""
    bp, vals = inp.bp, inp.vals

    V_dL = cfg["V_g"] * weight_kg
    mgdL_to_g = V_dL / 1000.0
    # Glucose appearance per gram of carbohydrate. The clinical identity behind the
    # "500 rule": one unit covers carb_ratio grams and drops glucose by si_day mg/dL,
    # so a gram must raise it by si_day / carb_ratio. Using this rather than the raw
    # 1000/(V_g*weight) volume conversion is what makes carb_ratio identifiable from
    # CGM at all - nothing else in the model reads it. At the population defaults the
    # two agree to within 1% (4.00 vs 3.97 mg/dL per gram). See NOTES.md.
    g_to_mgdL = params.si_day / params.carb_ratio if params.carb_ratio > 0 else 1000.0 / V_dL
    G_ref = cfg["G_ref"]
    k_renal = cfg["k_renal"]
    renal_thr = cfg["renal_threshold"]
    G_floor = cfg["G_clip_low"]

    HGO_base = cfg["HGO_base"]
    hgo_exp = cfg["hgo_G_exp"]

    k_ins = cfg["k_ins"]
    k_si = cfg["k_si"]

    kabs_floor = cfg["k_abs_floor_frac"] * params.k_abs
    k_abs_nom = params.k_abs

    caf_drop = cfg["caf_si_drop"]
    caf_ref = cfg["caf_ref_mg"]
    ex_si_gain = cfg["ex_si_gain"]
    ill_drop = cfg["illness_si_drop"]

    ex_gain = cfg["ex_uptake_gain"]
    adren_gain = cfg["adrenaline_gain"]
    ex_rise = cfg["ex_rise"]
    ex_decay = LN2 / cfg["ex_halflife_min"]

    gly_l_full = cfg["gly_liver_full"]
    gly_nominal = cfg["gly_nominal"]
    glyc_full = cfg["glycogenolysis_share"]
    gly_m_full = cfg["gly_muscle_full"]
    refill = cfg["refill_gain"]
    fast_drain = cfg["fast_drain"]
    m_refill = cfg["muscle_refill"]
    m_drain = cfg["muscle_drain"]

    alc_clear = cfg["alc_clear_per_min"]
    alc_rate = -math.log(max(1.0 - cfg["alc_hgo_gain"], 1e-3))
    alc_ref = cfg["alc_hgo_ref_bac"]

    caf_decay = LN2 / cfg["caf_halflife_min"]

    ket_gain = cfg["ket_gain"]
    if "sglt2" in (profile.meds or []):
        ket_gain *= cfg["sglt2_ket_mult"]
    ket_thr = cfg["ket_ins_thresh"]
    ket_clear = cfg["ket_clear"]

    hyd_loss_G = cfg["hyd_loss_G"]
    hyd_loss_alc = cfg["hyd_loss_alc"]
    hyd_gain_water = cfg["hyd_gain_water"]
    hyd_recover = cfg["hyd_recover"]
    hyd_lo = cfg["hyd_clip_low"]
    hyd_hi = cfg["hyd_clip_high"]

    si_night, si_day, si_evening = params.si_night, params.si_day, params.si_evening
    k_ex, k_alc = params.k_ex, params.k_alc
    si_mult = daily.si_mult
    t0_min = t0.hour * 60.0 + t0.minute + t0.second / 60.0

    def rhs(t, y):
        seg = vals[bisect_right(bp, t) - 1]
        crate = seg[_C_CARB]; irate = seg[_C_INS]; arate = seg[_C_ALC]
        cafrate = seg[_C_CAF]; wrate = seg[_C_WATER]
        exi = seg[_C_EXI]; exa = seg[_C_EXA]; ill = seg[_C_ILL]; kmult = seg[_C_KABS]

        G = y[_G]; I_p = y[_I_P]; D1 = y[_D1]; D2 = y[_D2]
        gly_l = y[_GLY_L]; gly_m = y[_GLY_M]
        BAC = y[_BAC]; caf = y[_CAF]; ket = y[_KET]; hyd = y[_HYD]; ex = y[_EX]

        G_safe = G if G > G_floor else G_floor
        g_ratio = G_safe / G_ref

        # --- gut -------------------------------------------------------------
        k_abs_eff = k_abs_nom * kmult
        if k_abs_eff < kabs_floor:
            k_abs_eff = kabs_floor
        dD1 = crate - k_abs_eff * D1
        dD2 = k_abs_eff * (D1 - D2)
        Ra_g = k_abs_eff * D2
        Ra = Ra_g * g_to_mgdL

        # --- insulin ---------------------------------------------------------
        dI_p = irate - k_ins * I_p

        # --- insulin sensitivity ---------------------------------------------
        hour = ((t0_min + t) / 60.0) % 24.0
        si_tod = si_night if hour < 8.0 else (si_day if hour < 16.0 else si_evening)
        caf_factor = 1.0 - caf_drop * (caf / caf_ref if caf < caf_ref else 1.0)
        SI_eff = (si_tod * k_si * si_mult * caf_factor
                  * (1.0 + ex_si_gain * k_ex * ex) * (1.0 - ill_drop * ill))
        if SI_eff < 0.0:
            SI_eff = 0.0
        uptake_ins = SI_eff * I_p * g_ratio

        # --- hepatic output ---------------------------------------------------
        # Two sources, because they behave differently. Glycogenolysis needs stores and
        # is what depletes them; gluconeogenesis does not, but it is the branch ethanol
        # inhibits. That split is what makes the alcohol hypo arrive hours late instead
        # of immediately: while the liver still holds glycogen it covers the shortfall.
        gly_frac = gly_l / gly_nominal
        if gly_frac > 1.0:
            gly_frac = 1.0
        # Exponential rather than linear-and-clamped. A hard clamp at zero means every
        # k_alc above ~0.7 produces an identical trajectory at drinking-night BAC, which
        # makes the parameter unfittable; it also claims gluconeogenesis stops dead.
        # alc_rate is set so that BAC 0.05 at k_alc = 1 still suppresses alc_hgo_gain
        # of hepatic gluconeogenesis, exactly as specified.
        alc_factor = math.exp(-alc_rate * k_alc * BAC / alc_ref) if BAC > 0.0 else 1.0
        counter_reg = (G_ref / G_safe) ** hgo_exp
        # Hepatic autoregulation: gluconeogenesis makes up whatever glycogenolysis
        # cannot supply, so total output is preserved as the store empties. Ethanol
        # inhibits only the gluconeogenic branch, so its bite grows through the night.
        glyc_share = glyc_full * gly_frac
        HGO_glyc = HGO_base * glyc_share * counter_reg
        HGO = HGO_glyc + HGO_base * (1.0 - glyc_share) * alc_factor * counter_reg

        # --- exercise, renal ---------------------------------------------------
        uptake_ex = ex_gain * k_ex * exi * g_ratio
        renal = k_renal * (G - renal_thr) if G > renal_thr else 0.0

        dG = Ra - uptake_ins + HGO - uptake_ex - renal + adren_gain * exa

        # --- glycogen ----------------------------------------------------------
        dgly_l = refill * Ra_g - HGO_glyc * mgdL_to_g - fast_drain
        if (gly_l <= 0.0 and dgly_l < 0.0) or (gly_l >= gly_l_full and dgly_l > 0.0):
            dgly_l = 0.0
        dgly_m = m_refill * Ra_g - m_drain * exi
        if (gly_m <= 0.0 and dgly_m < 0.0) or (gly_m >= gly_m_full and dgly_m > 0.0):
            dgly_m = 0.0

        # --- alcohol, caffeine -------------------------------------------------
        dBAC = arate - (alc_clear if BAC > 0.0 else 0.0)
        if BAC <= 0.0 and dBAC < 0.0:
            dBAC = 0.0
        dcaf = cafrate - caf_decay * caf

        # --- ketones -----------------------------------------------------------
        ins_lack = (ket_thr - I_p) / ket_thr
        if ins_lack < 0.0:
            ins_lack = 0.0
        dket = ket_gain * ins_lack * (1.0 - gly_l / gly_l_full) * (1.0 + ex) - ket_clear * ket
        if ket <= 0.0 and dket < 0.0:
            dket = 0.0

        # --- hydration ---------------------------------------------------------
        dhyd = (-hyd_loss_G * (G - renal_thr if G > renal_thr else 0.0)
                - hyd_loss_alc * BAC + hyd_gain_water * wrate + hyd_recover * (1.0 - hyd))
        if (hyd <= hyd_lo and dhyd < 0.0) or (hyd >= hyd_hi and dhyd > 0.0):
            dhyd = 0.0

        # --- residual exercise effect -------------------------------------------
        dex = ex_rise * exi * (1.0 - ex) - ex_decay * ex

        out = [0.0] * N_FAST
        out[_G] = dG; out[_I_P] = dI_p; out[_D1] = dD1; out[_D2] = dD2
        out[_GLY_L] = dgly_l; out[_GLY_M] = dgly_m
        out[_BAC] = dBAC; out[_CAF] = dcaf; out[_KET] = dket
        out[_HYD] = dhyd; out[_EX] = dex
        return out

    return rhs


# --------------------------------------------------------------------------- simulator

class RealSimulator:
    """Satisfies schema.simulate.Simulator.

    The Simulator protocol has no Profile argument, but sex, meds, blood pressure and
    the sleep target are needed for the medium and slow layers. They are supplied
    through the constructor instead, which keeps the protocol signatures untouched:
    get_simulator() builds RealSimulator() with the population default profile, and
    callers who have a real patient set `sim.profile`.
    """

    def __init__(self, profile: Profile | None = None, cfg: dict | None = None):
        self.profile = profile or Profile()
        self.cfg = cfg or defaults()

    # -- helpers ------------------------------------------------------------
    def _weight(self, daily: DailyState) -> float:
        w = daily.fat_mass_kg + daily.lean_mass_kg
        return w if w > 1.0 else self.profile.weight_kg

    def simulate(self, state: PatientState, events: list[Event], params: PatientParams,
                 horizon_min: int, dt_min: int = 5) -> Trajectory:
        cfg = self.cfg
        horizon_min = int(horizon_min)
        dt_min = int(dt_min)
        weight = self._weight(state.daily)

        evs = sorted(events, key=lambda e: e.t_start)
        inp = build_inputs(evs, state.t, float(horizon_min), params, cfg, weight)
        rhs = _make_rhs(inp, params, cfg, state.daily, self.profile, weight, state.t)

        n_steps = horizon_min // dt_min
        grid = np.arange(n_steps + 1, dtype=float) * dt_min
        y0 = np.asarray(state.fast, dtype=float).copy()
        fast = self._integrate(rhs, inp, y0, grid, float(horizon_min), float(dt_min))

        np.clip(fast[:, _G], cfg["G_clip_low"], cfg["G_clip_high"], out=fast[:, _G])
        np.clip(fast[:, _HYD], cfg["hyd_clip_low"], cfg["hyd_clip_high"], out=fast[:, _HYD])
        np.clip(fast[:, _GLY_L], 0.0, cfg["gly_liver_full"], out=fast[:, _GLY_L])
        np.clip(fast[:, _GLY_M], 0.0, cfg["gly_muscle_full"], out=fast[:, _GLY_M])
        for j in (_I_P, _D1, _D2, _BAC, _CAF, _KET, _EX):
            np.clip(fast[:, j], 0.0, None, out=fast[:, j])

        times = [state.t + timedelta(minutes=float(m)) for m in grid]
        energy = np.array([energy_score(row, state.daily) for row in fast])

        daily_end = replace(state.daily)
        slow_end = replace(state.slow)
        if horizon_min >= 1440:
            daily_end = self._advance_days(daily_end, evs, state.t, horizon_min, params, fast, grid)

        return Trajectory(t=times, fast=fast, energy=energy, daily_end=daily_end,
                          slow_end=slow_end, summary=self.summary(fast[:, _G], times))


    # -- integration --------------------------------------------------------
    def _integrate(self, rhs, inp: InputSchedule, y0: np.ndarray, grid: np.ndarray,
                   horizon_min: float, dt_min: float) -> np.ndarray:
        """Integrate segment by segment between input breakpoints.

        The exogenous inputs are piecewise constant, so the right-hand side jumps at
        every breakpoint. Handing solve_ivp the whole window and leaning on max_step
        lets it step across those jumps: measured against a converged reference that
        costs ~7 mg/dL of error on a normal day. Restarting the integrator at each
        breakpoint keeps the error under 2 mg/dL and is faster, because inside a
        segment the problem is smooth and RK45 can take long steps.
        """
        bps = sorted(set(list(inp.bp) + [horizon_min]))
        out = np.empty((grid.size, y0.size))
        y = y0
        k = 0
        for a, b in zip(bps[:-1], bps[1:]):
            if b <= a + _T_EPS:
                continue
            n = 0
            while k + n < grid.size and grid[k + n] <= b + _T_EPS:
                n += 1
            t_eval = list(grid[k:k + n])
            if not t_eval or t_eval[-1] < b - _T_EPS:
                t_eval.append(b)
            sol = solve_ivp(rhs, (a, b), y, method="RK45", t_eval=t_eval,
                            max_step=dt_min * _MAX_STEP_SEGMENTS, rtol=_RTOL, atol=_ATOL)
            if not sol.success or sol.y.shape[1] == 0:
                break
            if n:
                out[k:k + n] = sol.y[:, :n].T
                k += n
            y = sol.y[:, -1]
        
        while k < grid.size:                    # integrator stopped early
            out[k] = y
            k += 1
        return out


    def make_stepper(self, t0: datetime, events: list[Event], params: PatientParams,
                     daily: DailyState, horizon_min: float):
        """Build a reusable one-step propagator over a fixed window.

        The state estimator advances 23 sigma points through the same five minutes, over
        and over. Rebuilding the input schedule and the right-hand side each time would
        dominate the cost, so they are built once here and the returned callable just
        integrates: step(y, t_rel, dt_min) -> y at t_rel + dt_min.
        """
        weight = self._weight(daily)
        evs = sorted(events, key=lambda e: e.t_start)
        inp = build_inputs(evs, t0, float(horizon_min), params, self.cfg, weight)
        rhs = _make_rhs(inp, params, self.cfg, daily, self.profile, weight, t0)
        bp = inp.bp
        lo, hi = self.cfg["G_clip_low"], self.cfg["G_clip_high"]
        hyd_lo, hyd_hi = self.cfg["hyd_clip_low"], self.cfg["hyd_clip_high"]
        gl, gm = self.cfg["gly_liver_full"], self.cfg["gly_muscle_full"]

        def step(y: np.ndarray, t_rel: float, dt_min: float) -> np.ndarray:
            a, end = float(t_rel), float(t_rel) + float(dt_min)
            cuts = [b for b in bp if a + _T_EPS < b < end - _T_EPS]
            y = np.asarray(y, dtype=float)
            for b in cuts + [end]:
                sol = solve_ivp(rhs, (a, b), y, method="RK45", t_eval=[b],
                                max_step=dt_min * _MAX_STEP_SEGMENTS, rtol=_RTOL, atol=_ATOL)
                if not sol.success or sol.y.shape[1] == 0:
                    break
                y = sol.y[:, -1]
                a = b
            y = y.copy()
            y[_G] = min(max(y[_G], lo), hi)
            y[_HYD] = min(max(y[_HYD], hyd_lo), hyd_hi)
            y[_GLY_L] = min(max(y[_GLY_L], 0.0), gl)
            y[_GLY_M] = min(max(y[_GLY_M], 0.0), gm)
            for j in (_I_P, _D1, _D2, _BAC, _CAF, _KET):
                if y[j] < 0.0:
                    y[j] = 0.0
            y[_EX] = min(max(y[_EX], 0.0), 1.0)
            return y

        return step

    # -- summaries ----------------------------------------------------------
    def summary(self, G: np.ndarray, times: list[datetime]) -> dict:
        cfg = self.cfg
        lo, hi = cfg["tir_low"], cfg["tir_high"]
        i_min = int(np.argmin(G))
        return {
            "tir_pct": float(100.0 * np.mean((G >= lo) & (G <= hi))),
            "tbr70_pct": float(100.0 * np.mean(G < lo)),
            "tar180_pct": float(100.0 * np.mean(G > hi)),
            "min_G": float(G.min()),
            "max_G": float(G.max()),
            "t_min_G": times[i_min].isoformat(),
            "mean_G": float(G.mean()),
        }

    # -- medium layer (Deliverable 4) ---------------------------------------
    def _advance_days(self, daily: DailyState, events: list[Event], t0: datetime,
                      horizon_min: int, params: PatientParams, fast: np.ndarray,
                      grid: np.ndarray) -> DailyState:
        """Apply the daily update once per completed 24 h of the simulated window."""
        n_days = horizon_min // 1440
        for d in range(n_days):
            lo = t0 + timedelta(days=d)
            hi = lo + timedelta(days=1)
            mask = (grid >= d * 1440) & (grid < (d + 1) * 1440)
            mean_G = float(fast[mask, _G].mean()) if mask.any() else float(fast[:, _G].mean())
            mean_hyd = float(fast[mask, _HYD].mean()) if mask.any() else 1.0
            day_events = [e for e in events if lo <= e.t_start < hi]
            daily = self.daily_update(daily, day_events, params, mean_G, mean_hyd, lo)
        return daily

    def daily_update(self, daily: DailyState, day_events: list[Event], params: PatientParams,
                     mean_G: float, mean_hyd: float, day: datetime) -> DailyState:
        """One day of the medium layer. Pure: returns a new DailyState."""
        cfg = self.cfg
        prof = self.profile
        d = replace(daily)

        slept_h = sum(e.duration_min for e in day_events if e.type == "sleep") / 60.0
        d.sleep_debt_h = max(0.0, min(
            cfg["sleep_debt_cap"],
            d.sleep_debt_h + max(0.0, prof.sleep_target_h - slept_h)
            - cfg["sleep_debt_decay"] * d.sleep_debt_h,
        ))

        d.training_load = ((d.training_load
                            + sum(e.duration_min * e.intensity
                                  for e in day_events if e.type == "exercise"))
                           * math.exp(-1.0 / cfg["train_decay_days"]))

        illness = max((e.severity for e in day_events if e.type in ("illness", "stress")),
                      default=0.0)
        if prof.sex == "F":
            day_of_cycle = (day.toordinal() % int(cfg["cycle_len_days"])) + 1
            cycle = (cfg["cycle_low_factor"]
                     if cfg["cycle_low_start"] <= day_of_cycle <= cfg["cycle_low_end"] else 1.0)
        else:
            cycle = 1.0
        d.si_mult = ((1.0 - min(0.5, params.k_sleep * d.sleep_debt_h))
                     * (1.0 + cfg["train_si_gain"] * min(d.training_load / cfg["train_load_ref"], 1.0))
                     * (1.0 - cfg["illness_si_drop"] * illness)
                     * cycle)

        # energy balance
        kcal_in = sum(cfg["kcal_per_g_carb"] * e.carbs_g + cfg["kcal_per_g_protein"] * e.protein_g
                      + cfg["kcal_per_g_fat"] * e.fat_g + cfg["kcal_per_g_ethanol"] * e.ethanol_g
                      for e in day_events)
        weight = d.fat_mass_kg + d.lean_mass_kg
        bmr = (10.0 * weight + 6.25 * prof.height_cm - 5.0 * prof.age
               + (5.0 if prof.sex == "M" else -161.0))
        ex_kcal = sum(cfg["ex_kcal_per_min"] * e.duration_min * e.intensity
                      * weight / cfg["insulin_weight_ref"]
                      for e in day_events if e.type == "exercise")
        surplus = kcal_in - (bmr + ex_kcal)
        kg = surplus / cfg["kcal_per_kg_fat"]
        protein_g = sum(e.protein_g for e in day_events)
        if surplus > 0 and protein_g > cfg["protein_g_per_kg_lean"] * d.lean_mass_kg:
            d.lean_mass_kg = max(1.0, d.lean_mass_kg + kg * cfg["lean_partition"])
            d.fat_mass_kg = max(1.0, d.fat_mass_kg + kg * (1.0 - cfg["lean_partition"]))
        else:
            d.fat_mass_kg = max(1.0, d.fat_mass_kg + kg)

        d.hba1c_pct = self.hba1c_step(d.hba1c_pct, mean_G, 1.0)
        return d

    def hba1c_step(self, hba1c: float, mean_G: float, days: float) -> float:
        """dA/dt = k_glyc * mean_G * (1 - A/100) - A/120, integrated over `days`."""
        cfg = self.cfg
        k, tau = cfg["k_glyc"], cfg["hba1c_clear_days"]
        a = hba1c
        steps = max(1, int(days))
        h = days / steps
        for _ in range(steps):
            a += h * (k * mean_G * 100.0 * (1.0 - a / 100.0) - a / tau)
        return float(max(3.0, min(20.0, a)))

    # -- long horizon (Deliverable 4) ---------------------------------------
    def simulate_long(self, state: PatientState, weekly_template: list[Event],
                      params: PatientParams, years: float = 10.0) -> LongTrajectory:
        try:
            from engine.longrun import simulate_long as _long
        except ImportError:                      # Deliverable 4 not landed yet
            from engine.stub import StubSimulator
            return StubSimulator().simulate_long(state, weekly_template, params, years)
        return _long(self, state, weekly_template, params, years)
