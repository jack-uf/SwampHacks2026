# Connected-river feasibility study

The frozen upstream-flow pilot stopped before model fitting. Three of its four development pairs had sufficient approved reference-period discharge. The fourth returned no observations for the requested historical interval despite passing the USGS catalog screen. This study demonstrates no predictive improvement and supports no production integration. Thirteen potential geographic test pairs remain unopened.

The [protocol](UPSTREAM_RESEARCH_PROTOCOL.md) owns the proposed targets, timing, model settings and evaluation criteria. [coverage.json](../experiments/sprint-20261017-upstream/coverage.json) records the source failure. [audit.json](../experiments/sprint-20261017-upstream/audit.json) records the independently checked source hashes, metadata constraints and preservation checks.

## Research rationale

Connected upstream measurements might distinguish an approaching high-flow episode from persistent local high water. The proposed feature measures how much upstream flow changes a reference-period prediction of the downstream trajectory. It does not assume that a more complex network will outperform local flow and weather.

The primary-source review informed the design but supplied no transferable performance estimate:

| Source | Relevant evidence | Limit for this study |
| --- | --- | --- |
| CAMELS dataset article, HESS 21, 5293, 2017 | Documents 671 catchments for comparative hydrology, with limited human influence | Membership does not establish a useful connected upstream measurement network |
| USGS NLDI introduction | Documents upstream navigation with tributaries and whole-catchment indexing | A returned gauge is not proof of a precise hydraulic connection or travel time |
| *Accelerating Flood Warnings by 10 Hours: The Power of River Network Topology in AI-enhanced Flood Forecasting*, arXiv:2410.05536v3 | Abstract motivates testing topology and information propagation | Abstract reviewed; architecture and reported gains were not reproduced |
| *Graph Transformer Network for Flood Forecasting with Heterogeneous Covariates*, arXiv:2310.07631v1 | Abstract describes network, temporal and heterogeneous environmental inputs | Abstract reviewed; its data and performance do not validate this cohort |

The retained request manifests in [research](../experiments/sprint-20261017-upstream/research/) contain the exact primary-source URLs, retrieval times and response hashes. The catalog and navigation responses have their own manifests. No secondary account supplies a model claim.

## Geographic and source coverage

The metadata screen checked all 62 existing development gauges and selected 31 new basins. Each new simplified basin polygon had zero positive-area intersection with the 215 previously reserved gauges' polygons and with earlier accepted new basins. This separates catchments, not necessarily climate regions. Simplified polygons and NLDI indexing remain imperfect evidence of hydrological independence.

The 93 navigation requests produced 839 distinct eight-digit USGS upstream identifiers. Of these, 33 passed the daily-mean discharge catalog filters. Drainage-area, site-type and location checks left four development pairs and thirteen potential test pairs. The selection uses navigation with tributaries within 500 km, rather than geographic proximity alone. Full selection rules and immutable snapshots are in the protocol and experiment directory.

The pilot requested approved daily mean discharge for January 1, 2000 through December 31, 2022. Reference coverage uses dates before January 1, 2016.

| Downstream gauge | Upstream gauge | Approved days in requested interval | Approved reference days | Reference eligibility |
| --- | --- | ---: | ---: | --- |
| 12358500 | 12357500 | 0 | 0 | Failed |
| 03213700 | 03212980 | 8,168 | 5,651 | Passed |
| 07056000 | 07055875 | 8,157 | 5,669 | Passed |
| 07376000 | 07375800 | 7,624 | 5,069 | Passed |

For upstream gauge 12357500, the catalog reported a lifetime range of October 1, 1910 through October 16, 2026 and 7,269 records. The actual historical request returned HTTP 200 with an 81-byte response saying no sites matched the criteria. These observations were unavailable, not negative hazard labels. Gauge 07375800 also illustrates the coverage problem: its catalog begins in 1956, but the first accepted approved observation in the requested interval is October 1, 2001.

Lifetime endpoints and record counts therefore do not establish overlapping historical coverage. The approved reference coverage checks passed for three pairs, but no real-data transport fit established complete usable trajectories for those pairs. The protocol required all four pairs and prohibited replacements based on outcomes. The worker stopped; the audit completed the remaining source checks without changing that population.

## Implemented prototype and unrun comparisons

[upstream_pilot.py](../scripts/upstream_pilot.py) contains the tested transport-feature function and the source-coverage command. The feature prototype fits separate standardized ridge regressions to reference-period downstream trajectories, with and without upstream flow. Their prediction difference is the proposed transport innovation. It uses calendar lags and rejects missing required observations rather than shifting across gaps.

At issue time, the latest predictor is local D-3. Predicting D+1 through D+3 therefore requires useful information four to six days after that measurement. Fast upstream responses may already have passed the downstream gauge. The study did not measure propagation delays or test whether this hypothesis holds.

The frozen comparisons were rainfall ranking, local-hydrology LightGBM, combined weather/hydrology/snow LightGBM, combined logistic regression, combined plus raw upstream flow, and combined plus transport innovations. They reused archived ECMWF HRES features, not GFS. None ran. Unexecuted classifier machinery was removed after the coverage failure; the protocol remains the record of the proposed experiment.

No event counts, AP, calibration, false-warning rates, lead times, ablation results or geographic model results were computed. There is no performance table because there are no fitted candidates. Prior study metrics describe different populations and targets and cannot fill that gap.

The prototype's unit tests use synthetic fixtures only to test implementation behavior. They do not supply experimental training data or evidence of real-world predictive value.

## Verification and reproduction

All 185 repository tests passed, with three existing SHAP pending-deprecation warnings. The three new tests check navigation identifier validation, calendar-lag and post-reference invariance, and the difference between an official empty response and an unknown response schema. The audit also verifies source byte counts and hashes, metadata hash chains, polygon separation, pair constraints, development-only discharge collection and the absence of fitted artifacts or prediction tables. Frozen production inputs retain their recorded hashes.

The retained collection contains 187 raw source files totaling 6,970,144 bytes. The four discharge responses account for 784,053 bytes. No dependency, paid API or GPU experiment was added. No real-data model runtime or peak-memory benchmark exists because fitting never started.

From the repository root, with the retained raw files present:

```sh
.venv/bin/python -m scripts.audit_upstream
.venv/bin/python -m pytest -q tests/test_upstream_research.py
.venv/bin/python -m pytest -q
```

The collection commands were `scripts.upstream_research prepare`, `catalog`, `expand` and `catalog-expanded`, each invoked with `.venv/bin/python -m`. The final source command was `.venv/bin/python -m scripts.upstream_pilot coverage`. These commands protect existing snapshots and are not rerun instructions for an already populated experiment directory. The first historical download failed before coverage recording; the coverage command retained that response and audited all four pairs.

Raw research, basin, navigation and discharge responses are retained locally and ignored by Git. Committed manifests and the audit index preserve their URLs and hashes, but a clean checkout does not include those bytes. Reacquiring a revised upstream response is not an exact reproduction. Preserve the retained raw directory when transferring this experiment.

## Decision and next experiment

Keep this candidate out of production. The connected-flow hypothesis remains untested, and no new model can contribute to the hazard index from this study. The measured target remains a high-flow percentile exceedance, not verified flooding. Approved historical observations may contain retrospective revisions; D-3 observation availability and forecast publication timing remain assumptions without historical dissemination records.

The next study should select gauges from verified overlapping discharge records first, then construct connected development networks and reserve disconnected test networks. Expanding this CAMELS-based selection alone has weak support after the full development screen. Larger connected rivers with measurable multi-day travel times are a better candidate population, subject to direct connectivity and regulation checks.

Before freezing that study, verify hourly timestamps, publication latency and usable event counts. Compare local persistence and weather baselines with an authoritative historical hydrological forecast wherever a valid archive exists. Only then test whether upstream transport correction improves event onset and false-alert control. The thirteen unopened metadata pairs are a possible starting pool, not a ready independent evaluation set.
