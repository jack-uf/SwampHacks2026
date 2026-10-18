# Six-hour high-flow development research

This study investigates the regional errors reported in [MLP_GEO_RESULTS.md](MLP_GEO_RESULTS.md). Every previously inspected holdout remains development evidence. The statistical and operational promotion requirements in [SHIFT_HYDROLOGY_PROTOCOL.md](SHIFT_HYDROLOGY_PROTOCOL.md) remain binding.

The implementation is `scripts/high_flow_research.py`. The target, two-hour assumed observation delay, 169-hour complete history, 24-hour quiet period and six-hour decision cadence follow `scripts/shift_hydrology.py`. The target is high-flow onset above a historical daily discharge percentile. It does not establish flooding at a work site.

## Hypotheses and comparisons

The initial hypothesis is that coordinate dependence and insufficient representation of rise persistence cause regional false alerts. Cycle 1 compares the original LightGBM features, coordinate removal, added physical crossing features and equal HUC4 training weights. All configurations retain the existing tree capacity and iteration count. The hydrologic features use the correct reference crossing at `log(2)` because the existing inputs are `log1p(discharge/reference)`.

Cycle 2 tests whether broader training resolves sparse regional coverage. It pools the retained development datasets from the shift-hydrology, MLP onset, MLP hybrid and MLP geographic studies. It repeats the feature and weighting ablations and adds conditional calibration using season, flow margin and drainage area. Earlier outcomes from those studies remain inspected development evidence. No 2025 rows enter these experiments.

Proximity, linear extrapolation and damped extrapolation receive both scalar and conditional logistic calibration. Every candidate receives identical eligible observations, calibration dates and threshold-selection rules within its cycle. Pooling changes the comparison population between cycles, so the expansion effect also needs scoring on common stations. No architecture change is justified before these comparisons establish a representation failure.

## Leakage controls and operational metrics

All HUC4 groups assigned geographic test in the latest geographic cohort remain excluded from training and calibration across the expanded pool. This grouping prevents direct HUC4 overlap within the experiment. It does not certify polygon independence or independence from earlier development choices. The retained data carry source request provenance; each run verifies dataset certificates and records source hashes.

Training uses data before 2023. Calibration uses January through June 2023, and threshold selection uses July through December 2023. Validation uses 2024, split into temporal and geographic partitions. Every boundary purges decisions whose following 24-hour label crosses it. Calibration and threshold selection therefore cannot see validation labels. Warmup readings may precede these dates, as required by the original causal feature contract.

The ceiling rule chooses the lowest inclusive threshold satisfying 1.5 pooled false alerts per 100 decisions and three in every HUC2. A headroom ablation uses one pooled and two regional false alerts. Both rules apply to every comparator. These are development selection policies; final acceptance ceilings remain unchanged. Ties enter together, and a zero-alert threshold is admissible when no observed score meets the budget.

Report unique event recall, lead on jointly detected events, false-alert decisions and episodes, regional burden, AP, Brier loss and reliability bins. Retain paired HUC4 bootstrap intervals with 500 draws for selected development comparisons. Those intervals describe correlated development observations and cannot establish a post-selection promotion claim. Shared weather dependence across HUC4 groups remains a limitation. At a six-hour horizon and cadence, each eligible onset has at most one scheduled warning opportunity.

## Confirmation and stopping rule

Geographic AP remains a hard gate. Better recall at a larger realized false-alert rate does not establish equal-cost operational superiority. A future proposal to make AP supporting evidence would require a utility definition and a frozen event-recall comparison at matched alert budgets, with uncertainty and lead requirements. Such a proposal cannot retrospectively change any existing result.

A promising development configuration requires an immutable pipeline, thresholds, comparator choice and evaluation gates before collecting fresh confirmation outcomes. Its geographic cohort must pass watershed-polygon overlap screening against every fitted catchment. Prior station reservations and inspected outcome years must be recorded explicitly. Newly downloaded stations in previously studied regions alone do not establish an independent confirmation population.

Deployment additionally requires historical input-availability evidence and a site-to-gauge applicability assessment. The approved USGS archive and an assumed delay cannot clear that requirement. If available evidence cannot support fresh temporal and geographic confirmation, stop with a documented non-promotion decision and the specific acquisition work required. Production model hashes are checked for each run, and the experiment never writes production artifacts or deterministic hazard rules.

Each cycle writes its input, code, configuration and production hashes before fitting. Outputs refuse overwrite. Predictions, fitted artifacts and negative results remain under `experiments/high-flow-20261018/`. The results document records commands that actually ran and separates measured findings from hypotheses.
