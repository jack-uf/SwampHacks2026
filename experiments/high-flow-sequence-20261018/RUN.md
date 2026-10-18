# Hourly sequence experiment: run record

Status: complete on 2026-10-18. The sequence model did not beat the frozen cycle-11 tree candidate on
either 2024 partition. `results.json` holds its metrics, `comparison.json` the paired comparison.

## What was scored

No decision-time subsample. The population is every eligible decision of the 88-gauge signals dataset
(`experiments/high-flow-signals-20261018/development.csv.gz`), split by
`scripts/high_flow_research.split(cycle=4)`:

| partition | rows | gauges | events | window |
| --- | --- | --- | --- | --- |
| train (early-stopping tail in the last 10% of rows) | 85,550 | 49 | 547 | 2020-01-08 to 2021-12-30 |
| calibration (2022) | 50,936 | 49 | 296 | 2022-01-01 to 2022-12-30 |
| threshold (2023) | 89,230 | 85 | 491 | 2023-01-01 to 2023-12-30 |
| temporal_validation (2024) | 51,851 | 47 | 306 | 2024-01-01 to 2024-12-30 |
| geographic_validation (2024, reserved HUC4s) | 17,826 | 16 | 102 | 2024-01-01 to 2024-12-30 |

Scored rows: 209,843 of 295,393 (calibration, threshold and both 2024 partitions). Each row's
identity, target and event time is in `index/<partition>.csv.gz`, and `series.json` records the row
counts, the per-gauge series that produced the windows, and the source page hashes.

## Storage

Windows are never materialized on disk. `assemble` writes one float16 hourly four-channel series per
gauge under `series/` (about 20 MB) and `fit` slices windows from it in the worker's memory. The
series was deleted after training and scoring, per the disk cleanup request: rebuilding it costs
about a minute of cached-page parsing (`assemble` after deleting `series/` and `index/`). `index/`
and `series.json` stay because they are the record of which rows were scored.

## Population note

A one-decision-hour-per-day subsample was considered and not taken: storage scales with gauge hours,
not decision rows, so it would have freed no space, and it would have broken the identical-row
pairing against the stored cycle-11 predictions that `paired_uncertainty` requires.
