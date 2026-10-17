# Expanded model evaluation

See RESEARCH_PLAN.md for the predeclared design. Final test is January–June 2026; 2025 is development data now.

| Hazard | Selected | County-days (+) | Precision | Recall | AP | Brier | False alerts /100 county-days |
|---|---|---:|---:|---:|---:|---:|---:|
| flood | frozen_baseline | 3258 (42) | 0.073 | 0.595 | 0.104 | 0.0123 | 9.76 |
| storm | lightgbm_enhanced | 3258 (80) | 0.088 | 0.613 | 0.130 | 0.0234 | 15.56 |

## Compatible comparison: same ten original counties, same 2026 dates

| Hazard / version | Precision | Recall | AP | Brier | False alerts /100 county-days |
|---|---:|---:|---:|---:|---:|
| flood / original_threshold_baseline | 0.036 | 0.875 | 0.120 | 0.0168 | 41.49 |
| flood / updated | 0.076 | 0.625 | 0.120 | 0.0168 | 13.37 |
| storm / original_threshold_baseline | 0.050 | 0.812 | 0.150 | 0.0250 | 40.55 |
| storm / updated | 0.083 | 0.667 | 0.130 | 0.0257 | 19.45 |

The original threshold is unchanged in this comparison; the updated threshold comes only from 2025 validation. Differences combine expanded fitting, calibration, features and threshold policy; they do not isolate one intervention. Original 2025 metrics cannot be directly compared with a different 2026 population. Paired week-block intervals, all candidate results (including failures), prevalence, reliability counts, geographic tests and quarterly/county variability are in models/evaluation.json. Confusion matrices are [[TN, FP], [FN, TP]]. AP means average precision.

Geographic transfer without inspected holdouts: the leave-one-state-out diagnostic in [MODEL_CARD.md](MODEL_CARD.md#geographic-transfer-leave-one-state-out) and `models/spatial-cv.json`.

The separate [measured river flood audit](FLOOD_SPRINT_RESULTS.md) stopped before modeling because historical threshold and gauge-reference validity remained unverified. It does not change the county-report results above or the production models.

The separate [forecast-memory hydrology experiment](HYDROLOGY_RESEARCH_RESULTS.md) evaluates measured high discharge at held-out watersheds. It retains experimental candidates only; a discharge percentile is not a certified flood threshold or site-safety outcome.

The separate [emerging high-flow study](ONSET_RESEARCH_RESULTS.md) evaluates newly emerging measured-discharge episodes at fresh watersheds. Its onset target and warning policies remain experimental and are not comparable with persistent-high-flow AP. Production models are unchanged.

The separate [catchment-response study](RESPONSE_RESEARCH_RESULTS.md) tests day-of-onset neural models, response conditioning and auxiliary trajectories on polygon-screened fresh watersheds. A prespecified neural comparator improves event detection, but AP uncertainty and regional false alerts prevent integration.

The separate [onset-supervision and warning-budget study](WARNING_RESEARCH_RESULTS.md) compares matched neural objectives and delayed-feedback regional warning budgets on another fresh watershed population. Neither frozen improvement claim passes, so all candidates remain experimental.

The separate [connected-river feasibility study](UPSTREAM_RESEARCH_RESULTS.md) stopped before fitting because a frozen upstream gauge lacked historical discharge. It preserves potential geographic test outcomes and provides no new predictive-performance claim.

The separate [historical-transition and weather-correction study](ANALOG_RESEARCH_RESULTS.md) improves AP on fresh watersheds using local pre-evaluation flow histories. Its event-recall and false-alert criteria fail, so the selected model remains experimental.

## Limits

- NOAA-reported county events, not site incidents or injuries; unreported does not mean safe.
- 24 convenience-sampled counties; geographic test states excluded from new fitting; no nationwide validation.
- Previous Runs is fixed 48h nominal lead per hour, not one initialization; dissemination timestamps unavailable.
- UTC target day begins 24h after midnight decision time; not the next local work shift.
- Antecedent precipitation is forecast, not observed; no reanalysis or future observations.
- Spatial and seasonal dependence and reporting revisions limit inference; reliability bins retain sample counts.
- Warning budget (20%) and recall floor (30%) are unvalidated operational hypotheses.
- Index weights and scope restriction are conservative heuristics, not calibrated injury probability or safety boundaries.
- SHAP explains calibrated model probability using 16 training-background rows and independent feature masking; correlated inputs can create implausible combinations.
- Frozen baseline includes Arizona fitting; its Arizona geographic result is not an independent holdout.
