# Frozen confirmation protocol for the six-hour high-flow candidate

This document freezes the cohort, pipeline, comparator, metrics and gates for the independent
confirmation run. It is written before any outcome for the confirmation cohort has been acquired or
inspected. Nothing here may be changed after the first outcome is seen; a change invalidates the
confirmation and restarts it on a fresh cohort.

## Why a new cohort is needed

Every 2024 partition used during development was inspected repeatedly, and the 2025 outcomes were
inspected during the HUC2 04 regional diagnosis. The retained CAMELS selection is also exhausted at
HUC4 level: as recorded in `experiments/high-flow-20261018/independence-audit.json`, all 208 unused
metadata-eligible gauges sit in HUC4 groups an earlier study already touched.

Independence is a watershed property rather than a HUC4 property, so the cohort was rebuilt by
positive-area catchment screening instead:
`experiments/high-flow-signals-20261018/fresh-geography/camels-screen.json` measures every unused
catchment in the retained full-resolution boundary set against every catchment any earlier experiment
exposed. 278 unused catchments were screened, **193 are polygon-independent, and 179 of those are
mutually disjoint**, spread over sixteen HUC2 regions (Upper Missouri 40, Pacific Northwest 38,
South Atlantic-Gulf 21, Missouri 15, Ohio 14, Texas-Gulf 13, California 11, and smaller groups
elsewhere). Only boundary geometry was used;
no discharge, reference or outcome was acquired or inspected for that screen.

## Cohorts and periods

- **Confirmation stations.** The 179 mutually disjoint, polygon-independent catchments listed in the
  screen artifact, in the deterministic order recorded there. Acquisition proceeds in HUC2-stratified
  batches so that a partial batch cannot bias the cohort toward one region.
- **Training population.** Unchanged: the retained 88-gauge pooled development population. Expanding
  training with the confirmation stations would make them development data and is explicitly not done.
- **Outcome periods.** 2020-2024 remain development evidence. The confirmation outcome period is
  **2025-01-01 to 2026-09-30**, which no earlier study inspected for these stations or otherwise.
  Rows up to 2024-12-31 may be acquired to satisfy the 169-hour warm-up contract, and they enter no
  metric.
- **Decision hours.** Stage A scores 00, 06 and 18 UTC decisions, which the frozen pipeline handles
  without forecast fields. Stage B adds 12 UTC decisions once HRRR forecast fields for the cohort are
  acquired. Stage A is the primary confirmation; Stage B is reported separately and never pooled with
  it.

## Frozen pipeline and comparator

The pipeline is cycle 11's `winner_forecast` in `scripts/high_flow_signals.py`, frozen at commit
time with its artifact hash recorded in `experiments/high-flow-signals-20261018/candidate.json`:
the original discharge features, the ERA5 lag-six weather block, the two precipitation ratios, the two
wetness features, the seven rise-shape features and the as-issued HRRR forecast block, fitted as a
250-tree LightGBM at learning rate .04, calibrated conditionally on season, flow margin and drainage
area, and thresholded at the 2023 inclusive pooled-one and regional-two false-alert budget.

The comparator is conditional linear extrapolation, refitted on the same cohort rows with the same
calibration and threshold rules. No threshold may be re-selected on confirmation outcomes; the
threshold is carried from the frozen artifact.

## Metrics and gates

The sixteen existing promotion gates are preserved exactly: at least 20 eligible events per partition
and 50 in total, AP improvement with a paired lower bound above zero in both partitions, event-recall
gain of at least five points in both partitions with a pooled paired lower bound above zero, Brier no
worse, no HUC2 above three false alerts per hundred with pooled burden at most 1.5, and no loss of
median lead on jointly detected events. Intervals use 500 paired HUC4-cluster bootstrap draws.

Partitions: a **geographic partition** of every confirmation station, and a **temporal partition**
restricted to confirmation stations with pre-2023 training records, matching the development
partition rule. Passing the gates is necessary but not sufficient for an external SOTA claim; the
internal benchmark stands separately.

## Recorded limitations

Polygon independence removes nesting and overlap. It does not remove spatial weather correlation
between confirmation stations, which the cluster bootstrap only partly addresses. The rain features
still rest on ERA5 reanalysis with an assumed six-hour availability, so an operational deployment
additionally needs as-issued precipitation receipt evidence and a site-to-gauge applicability
assessment. Neither exists.
