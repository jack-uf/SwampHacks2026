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
