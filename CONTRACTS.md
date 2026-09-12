# GlucoPilot — Contracts

GlucoPilot is a personalized physiological simulator for people with Type 1 diabetes. An ODE
core models glucose, insulin, gut carbohydrate, glycogen, alcohol, caffeine, ketones, hydration
and exercise; per-patient parameters are fitted from the patient's own CGM and pump history, and
a learned residual model corrects whatever the ODEs miss. The simulation runs on three
timescales — **fast** (minutes, the state vector below), **medium** (days: sleep debt, training
load, insulin sensitivity drift) and **slow** (years: organ damage accumulators and 10-year
complication risk). On top of that sits a decision layer that answers permission questions
("can I have two IPAs tonight?") by simulating the proposal against alternatives, and a hard
safety layer of clinician rules that can deny, demand data, or escalate regardless of what the
simulation says. An LLM is used in exactly two places — photo → macros and free text → `Event`
— and never for clinical reasoning.

**All data in this repo is synthetic.** The demo patient is `data/synthetic/patient_01`, and the
API's `patient_id` default value `"default"` maps to it.

---

## Fast state vector — `schema/state.py`

`FAST_NAMES` order **is** the contract. Index with `FAST_IDX`, never with a literal integer.

| # | Name | Unit | Notes |
|---|------|------|-------|
| 0 | `G` | mg/dL | plasma glucose |
| 1 | `I_p` | mU/L | plasma insulin |
| 2 | `D1` | g | gut carbohydrate compartment 1 |
| 3 | `D2` | g | gut carbohydrate compartment 2 |
| 4 | `gly_liver` | g | liver glycogen, nominal full ~100 |
| 5 | `gly_muscle` | g | muscle glycogen, nominal full ~400 |
| 6 | `BAC` | g/dL | blood alcohol, 0.08 = US legal limit |
| 7 | `caf` | mg | caffeine in body |
| 8 | `ket` | mmol/L | blood ketones |
| 9 | `hyd` | fraction | hydration, 1.0 = euhydrated |
| 10 | `ex` | 0..1 | residual exercise effect, decays after activity |

`DailyState`: `sleep_debt_h`, `si_mult`, `training_load`, `fat_mass_kg`, `lean_mass_kg`,
`hba1c_pct`. `SlowState`: one damage accumulator per organ in `ORGANS`
(`eye`, `kidney`, `nerve`, `cardio`, `liver`), dimensionless, `0` = this patient's baseline.

`PatientState` bundles `t`, `fast`, `daily`, `slow`, and `fast_cov` (the estimator's covariance,
`None` when unknown).

---

## Events — `schema/events.py`

One `Event` is one discrete thing the patient did or proposes to do. Adapters, parsers, taps, the
synthetic generator and the decision engine all emit these; `simulate()` consumes them.

| Field | Unit | Applies to | Notes |
|-------|------|-----------|-------|
| `type` | — | all | `meal, insulin, alcohol, caffeine, water, exercise, sleep, illness, stress` |
| `t_start` | datetime | all | |
| `duration_min` | min | exercise, sleep, basal insulin, illness, stress | `0` for instantaneous events |
| `source` | — | all | `cgm_export, pump, apple_health, photo, text, tap, synthetic, proposed, profile` |
| `uncertainty` | fraction | all | 1-sigma on the primary amount, e.g. `0.20` on a photo carb estimate |
| `label` | — | all | display only |
| `carbs_g` | g | meal (and drinks that carry carbs) | |
| `fat_g` | g | meal | |
| `protein_g` | g | meal | |
| `fiber_g` | g | meal | |
| `units` | U | insulin | bolus: total units at `t_start`. basal: units **per hour** over `duration_min` |
| `insulin_kind` | — | insulin | `bolus` \| `basal` |
| `ethanol_g` | g | alcohol | one US standard drink = 14 g |
| `caffeine_mg` | mg | caffeine | |
| `water_ml` | mL | water | |
| `intensity` | 0..1 | exercise | 0.3 walk, 0.6 moderate, 0.9 hard |
| `exercise_kind` | — | exercise | `aerobic` \| `anaerobic` \| `mixed` |
| `sleep_quality` | 0..1 | sleep | |
| `severity` | 0..1 | illness, stress | |

`Profile` carries the static patient facts (demographics, `carb_ratio`, `correction_factor`,
`basal_u_per_h`, targets, `sleep_target_h`, complications, meds).

---

## Fitted parameters — `schema/params.py`

`PARAM_NAMES` order is the contract for `to_array` / `from_array`. Population constants that are
**not** fitted per patient live in `engine/params.yaml`, not here.

| Param | Unit | Meaning |
|-------|------|---------|
| `si_night` | mg/dL per U | insulin sensitivity, 00:00–08:00 |
| `si_day` | mg/dL per U | insulin sensitivity, 08:00–16:00 |
| `si_evening` | mg/dL per U | insulin sensitivity, 16:00–24:00 |
| `carb_ratio` | g per U | grams of carbohydrate covered by one unit |
| `k_abs` | 1/min | gut absorption rate constant |
| `k_ex` | ×population | scale on exercise-driven sensitivity and glucose uptake |
| `k_alc` | ×population | scale on alcohol suppression of hepatic glucose output |
| `basal_drift` | U/h | added to the logged basal; captures under/over-basaled patients |
| `k_sleep` | fraction per h | insulin-sensitivity drop per hour of sleep debt, capped at 0.5 total |

`ParamPosterior` holds a `mean` plus a `(n_samples, 9)` sample matrix; `population_prior()` gives
a wide lognormal starting point for patients with no history.

---

## Routes

All under `/api`, JSON bodies, `patient_id` query parameter defaults to `"default"`.

| Method | Route | Request | Response |
|--------|-------|---------|----------|
| GET | `/state` | — | `StateResponse` |
| GET | `/forecast?hours=8` | — | `ForecastResponse` |
| POST | `/permission` | `PermissionRequest` | `PermissionResponse` |
| POST | `/plan` | `PlanRequest` | `PlanResponse` |
| POST | `/change` | `ChangeRequest` | `ChangeResponse` |
| GET | `/budget` | — | `BudgetResponse` |
| GET | `/alerts` | — | `AlertsResponse` |
| POST | `/replay` | `ReplayRequest` | `ReplayResponse` |
| GET | `/trends` | — | `TrendsResponse` |
| GET | `/body?change_id=` | — | `BodyResponse` |
| POST | `/parse` | `{text}` | `ParseResponse` |
| POST | `/parse_photo` | multipart image | `Event` (type `meal` or `alcohol`) with `uncertainty` |
| POST | `/events` | `Event` | `204` (tap-logged intake) |
| POST | `/upload` | multipart `kind=cgm\|pump\|health`, `file` | `{"events": n, "glucose_points": n}` |
| GET | `/clinician` | — | `ClinicianResponse` |
| GET | `/health` | — | `{"ok": true, "simulator": "real"\|"stub", "inference": "real"\|"naive"}` |

The models in `schema/api.py` are the **only** shapes the frontend renders. `data/fixtures/*.json`
holds one populated example per response model; the frontend builds against those until the API
is live, and `tests/test_contracts.py` round-trips every one of them.

---

## Conventions

- **Glucose series** are always `pd.Series`, float mg/dL, `DatetimeIndex` on a **5-minute grid**,
  `NaN` for gaps. Never a list, never irregular, never mmol/L.
- **Weekly template**: a `list[Event]` whose `t_start` values fall in the reference week,
  Monday `2000-01-03 00:00` through Sunday `2000-01-09 23:59` (`schema.simulate.REF_WEEK_START`).
  `simulate_long()` tiles that week forward.
- **`energy_score(fast, daily)` in `schema/state.py` is the single source of truth** for readiness.
  Do not compute a second one anywhere.
- **Nobody imports `engine/` or `inference/` directly.** Call `schema.simulate.get_simulator()`
  and `schema.inference.get_inference()`. They return the real implementations when present and
  the stub / naive versions otherwise, so every workstream runs against a working system from
  hour zero.
- `Trajectory.summary` always carries `tir_pct`, `tbr70_pct`, `tar180_pct`, `min_G`, `max_G`,
  `t_min_G` (ISO string), `mean_G`. Time in range is 70–180 mg/dL.
- Internal state is plain dataclasses + numpy. Anything crossing an HTTP boundary is pydantic v2.
- **`engine/stub.py` is disposable.** It is deterministic and instant, and it contains no real
  physiology — it exists only so the API, decision layer, frontend and fixtures can be built
  before `engine/simulate.py` lands. Its one documented departure from the original contract
  sheet is the damped insulin term (see the comment on `BOLUS_EFFECT_SCALE`), without which the
  specified fixture day flatlines on the 40 mg/dL clip.

---

## Ownership

| Area | Owner |
|------|-------|
| `engine/`, `inference/` | Person A |
| `web/` | Person B |
| `api/`, `decision/`, `safety/` | Person C |
| `adapters/`, `parsers/`, `data/`, `scripts/`, `tests/validation/` | Person D |
| `schema/`, `CONTRACTS.md` | shared — see the rule below |

## The one rule about `schema/`

**Schema changes must be additive and announced.** Add a field with a default; do not rename,
reorder, retype or remove anything, and do not change a unit. `FAST_NAMES`, `PARAM_NAMES` and the
`Event` field set are indexed positionally by code you did not write. If a change genuinely cannot
be additive, say so in chat before you push it and get the affected owners to agree.

---

## Running it

```bash
make setup      # pip install -r requirements.txt (+ npm install if web/package.json exists)
make fixtures   # regenerate data/fixtures/*.json
make test       # pytest -q
make api        # uvicorn api.main:app --reload --port 8000
make web        # cd web && npm run dev
make demo       # fixtures, cohort if needed, fit, then api + web
```
