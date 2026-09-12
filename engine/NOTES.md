# engine / inference notes

Where the implementation departs from the specification hand-off, and why. Nothing here
touches `schema/`: every signature and every field name is as delivered.

## Things the specification named that do not exist

**`params.si_scale_from_tod`.** The SI_eff formula referenced it, but `PARAM_NAMES` has no
such entry. The same sentence also says the time-of-day sensitivities are "converted to a
rate with `k_si` from yaml", so `k_si` is doing that job and `si_scale_from_tod` is dropped.

**`Profile` is not in the `Simulator` protocol.** The daily update needs `sleep_target_h`,
`sex`, `height_cm` and `age`; `simulate_long` needs `systolic_bp` and weight; ketogenesis
needs `meds`. Rather than change a signature, `RealSimulator.__init__` takes an optional
profile and defaults to the population one, so `get_simulator()` keeps working unchanged and
a caller with a real patient sets `sim.profile`.

**Body weight.** Taken from `DailyState.fat_mass_kg + lean_mass_kg`, which is in the state
vector and is what the body-composition update moves. Falls back to `profile.weight_kg`.

## Equation changes

These were not cosmetic. In each case the specified form fails a shape the specification
itself asks for.

**Hepatic output is split into two branches.** Specified:
`HGO = HGO_base * gly_factor * alc_factor * (G_ref/G)^hgo_G_exp` with
`gly_factor = gly_liver/gly_liver_full` floored at 0.1. Implemented:

```
glyc_share = glycogenolysis_share * min(1, gly_liver / gly_nominal)
HGO        = HGO_base * (glyc_share + (1 - glyc_share) * alc_factor) * (G_ref/G)^hgo_G_exp
```

Two separate failures forced this.

1. *A fasting patient drifted downward.* With output proportional to the glycogen store and
   the store depleting, hepatic output decayed all night and glucose settled near 80 mg/dL
   on correct basal. Real livers autoregulate: gluconeogenesis takes over what glycogenolysis
   cannot supply, so total output holds. The `(1 - glyc_share)` term is that.
2. *The alcohol hypo arrived immediately instead of overnight.* Ethanol inhibits
   gluconeogenesis, not glycogenolysis. Multiplying all hepatic output by `alc_factor`
   removed 100% of it within 30 minutes of the first drink, so glucose fell 45 mg/dL by
   midnight. Applying it to the gluconeogenic branch only means a well-stocked liver covers
   the first few hours and the low appears as the store runs down, which is both the real
   mechanism and the shape the smoke test asks for.

**Liver glycogen drains from the glycogenolytic branch only.** Specified:
`dgly_liver/dt = refill_gain*Ra - HGO_g - fast_drain`. Draining the whole of hepatic output
from the glycogen pool empties a 100 g liver in about eight hours of ordinary fasting. Only
`HGO_glyc` is subtracted now.

**Stress events behave like illness** in the fast layer. The specification wires `illness`
into `illness_factor` and leaves `stress` with no effect anywhere, which would make a logged
stress event silently do nothing. Both raise insulin resistance, so both feed the term.

**Tried and discarded:** an `alc_counterreg_blunt` term (ethanol blunts the glucagon and
growth-hormone response to hypoglycaemia — real, and well documented). It was an attempt to
separate the mild two-hour effect from the deep overnight one before the autoregulation fix
landed. Once hepatic output was split properly it was not needed, and it cost accuracy on the
midnight target, so it is gone rather than sitting at zero.

## Integration

The specification suggests one `solve_ivp` call with `t_eval` on the output grid and
`max_step=5`. The exogenous inputs are piecewise constant, so the right-hand side jumps at
every breakpoint, and the solver steps straight across those jumps. Measured against a
converged reference (`rtol=1e-9`, `max_step=1`) that costs about **7 mg/dL** of error on an
ordinary day.

Integrating segment by segment between input breakpoints - still `solve_ivp`, still RK45 -
holds the error under **2 mg/dL** and is *faster*, because inside a segment the problem is
smooth and RK45 can take long steps. A 24-hour simulation with 10 events runs in ~21 ms
against the 30 ms budget.

## Constants that are tuned rather than measured

Flagged here because they are the ones to revisit if Person D's validation plots disagree.
All of them carry a `source` string in `params.yaml` saying the same thing.

- **`V_g: 3.6` dL/kg**, against Hovorka's 1.6. The specified `dG/dt` has no
  insulin-independent (CNS) uptake term and no hepatic first-pass extraction, so roughly
  half of a meal's glucose has nowhere to go. Inflating the effective distribution volume is
  the least invasive way to get a 60 g meal to peak near +140 mg/dL rather than +500.
- **`bolus_ramp_min: 80`**, against the specified 15. There is a single insulin compartment,
  so the 50-90 minute delay to peak plasma insulin has to live in the input ramp. With a
  15 minute ramp, insulin acts before the carbohydrate does and a correctly bolused meal
  comes out flat.
- **`caf_si_drop: 0.25`**, against the specified "~10% at 200 mg". At 10% the caffeine smoke
  test only moved the peak 7 mg/dL against a required 10-30. The literature band
  (Lane 2004/2008) is 8-25% on postprandial glucose, so this sits at the top of it rather
  than outside it.
- **`HGO_base`, `k_u2mUL`, `k_si`** are derived, not fitted. Requiring that basal 1 U/h holds
  plasma insulin at the 10 mU/L in `schema.state.default_fast`, and that one unit of bolus
  lowers glucose by exactly `params.si_*` mg/dL, pins all three: `k_u2mUL = 600 * k_ins`,
  `k_si = 1/600`, `HGO_base = si_day * k_si * 10`. `k_ins` can be retuned freely as long as
  `k_u2mUL` moves with it.

## Smoke-test scenarios the specification left open

- **Basal insulin is running** in every test except the first. "60 g carbs, no insulin" is
  taken literally; "6 U bolus, no food" is read as no *food*, with a pump still delivering.
- **The alcohol test runs on a fed day** (three bolused meals). "With normal basal" implies a
  normally fed patient, and the delayed hypo is a story about the liver running out of
  glycogen - a patient who has not eaten for fifteen hours has none left to run out of.
- **The caffeine test uses a bolused meal.** Comparing two unbolused runs that both pin near
  the top of the range measures the ceiling, not caffeine.
- **"Rises during the first 20 min, then falls"** for anaerobic work is read as falling after
  the session. The adrenaline term is constant while the work continues, so during the
  session glucose plateaus rather than turning over.
- `test_basal_only_patient_holds_steady` is an addition. Every other test compares against a
  baseline run, so a wandering resting set point would make all of them meaningless.
