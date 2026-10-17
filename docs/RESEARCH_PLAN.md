# Research and experiment protocol — frozen before expanded-data inspection

Decision date: October 16, 2026. Governing request: the updated “Improve PlanIT Control” prompt.

The original benchmark was reproduced exactly (all numeric evaluation fields) using `python -m scripts.train` at commit 7904b74. Recoverable original CSV, provenance, model, metrics and scripts are retained under `experiments/baseline/`, with SHA-256 manifest. This is an experiment archive, not a second application codepath. Persistent SQLite data and uploads are preserved.

## Material validity findings

The target is a **reported NOAA county-day event**, never a plant incident, injury probability or verified absence of danger. Flood means Flood or Flash Flood; storm means Thunderstorm Wind, Tornado or Hail. These five categories are designated county records in the official bulk format. Other hazards, including Coastal Flood, High Wind and Tropical Storm, are forecast-zone records and are outside these targets. Preserve an explicit audit of every event type / C-Z-M combination; fail if a target category appears outside county geography. Do not silently attach zones to counties.

Previous Runs is a fixed lead time series, not a single initialization. For target UTC day D and decision time D−1 at 00 UTC, target hours at previous_day2 have nominal initialization D−2 00 through D−2 23: before decision time. Exact original dissemination timestamps are unavailable; this is a nominal availability audit, not proof of historical delivery. GFS Single Runs starts April 2, 2026, so cannot establish a consistent 2024–2025 training contract. Keep Previous Runs and disclose that limitation.

Baseline calibration had only four positives per hazard; selection had nine flood and seven storm positives. The 150 km transfer guard has not been validated. Existing 2025 results have already been inspected and are no longer an untouched test for new experiments.

## Predeclared design

Expand from 12 to 24 county representatives, adding official Census gazetteer coordinates for 12 counties across Pacific, Mountain, Midwest, Southeast and Northeast settings. Retain original representative coordinates for original counties. No outcome-driven county selection. Preserve natural prevalence; use complete hourly forecast records only.

Download scope: 24 public Previous Runs requests, four hourly variables for January 2024–June 2026 (~2.1 million scalar values, estimated <100 MB JSON); three NOAA annual compressed detail files (2024/2025 cached), one Census gazetteer (<1 MB). Long ranges may count as multiple API calls. Budget conservatively 100 normalized calls/request = 2,400 calls total, below 10,000/day and 5,000/hour. Serialize requests at ten-second intervals (<=600 normalized calls/minute estimate), honor Retry-After and stop on repeated rate limits. Noncommercial research only; no paid endpoint. Do not extend collection automatically beyond this scope.

Temporal partitions: train through December 2024; calibration January–June 2025; selection July–December 2025; final test January–June 2026 (three-month label-maturation margin relative to the September 18, 2026 NOAA revision). This final period has not been used by the original benchmark. Freeze model/threshold selection before evaluating final outcomes. Report final results once; no post-test tuning.

Geographic test: all records from California, Arizona, Washington and Utah excluded from fitting/calibration/selection. Separate geographic validation: Georgia and Colorado, likewise excluded from fitting/calibration. All other sampled states form temporal development/test populations. This is a diverse convenience sample, not a national sample. No neighboring county from these holdout states is included in fitting.

Limited search: balanced logistic, XGBoost and LightGBM using the original fixed hyperparameters; logistic without balancing; enhanced-feature logistic and boosted variants. Base eight features versus forecast maximum hourly precipitation, hours with precipitation, preceding 72-hour **forecast** precipitation, rain × humidity and gust × temperature interactions. Antecedent values are previous_day2 forecasts, never realized observations or reanalysis. Reject soil moisture: its consistent historical forecast availability has not been established. No synthetic positives or negative subsampling.

Choose using mean temporal and geographic-validation average precision, with Brier as tie-break. Evaluate threshold tradeoffs on temporal validation only: F1, F2, and a bounded alert-budget policy (maximum 20% alert rate; among policies reaching >=30% recall maximize F1). If none meets the budget/recall requirements, explicitly mark the warning threshold unsupported instead of hiding a degenerate policy. Compare the frozen baseline on the same final rows for original supported counties; do not call different-population metrics an improvement.

Keep deployment winner unless expanded selection evidence improves on the frozen baseline under the same validation population. Inference remains restricted to sampled development counties by explicit inventory county/state matching, with forecasts at the evaluated representative point. Geographic holdouts remain unsupported even if retrospective scores look favorable. This scope rule is conservative population matching, not a validated safety boundary.

Report prevalence, average precision, ROC-AUC, Brier, equal-width reliability bins/counts, confusion matrices and false alerts per 100 county-days; quarterly and per-county results show heterogeneity. Use a paired calendar-week block bootstrap only on comparable final rows, preserving spatial dependence within a week. Intervals are descriptive, not nationwide uncertainty.

Jev work proceeds independently against official documentation. No live inference without credentials and explicit text-sharing consent; never transmit photographs. Fixtures are hypothetical functionality checks, not field validation. Deterministic lightning rules and index weights stay unchanged; weights remain documented heuristics.
