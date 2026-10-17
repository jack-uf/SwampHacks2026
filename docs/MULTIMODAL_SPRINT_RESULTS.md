# Multimodal environmental hazard pilot: results

The pilot asked whether satellite context, conventional or Jev-derived, adds measurable predictive value to measured wind-gust exceedance at power-plant weather stations. The answer is **no measurable value in this pilot**, and the Jev visual path failed its quality gate before it could be scaled. The frozen design is in [MULTIMODAL_SPRINT_PROTOCOL.md](MULTIMODAL_SPRINT_PROTOCOL.md); all numbers come from `experiments/sprint-20261017-multimodal/`. Nothing is integrated and no application code changed.

## Datasets and features

Sentinel-2 L2A COGs from the open `sentinel-cogs` bucket via the Element 84 STAC API, no credentials. For each station-day the pipeline picked the recent acquisition with the lowest cloud fraction inside a 5.12 km window, subject to a 21-day lookback and a one-day publication-latency allowance, and computed NDVI, NDWI, SCL water/vegetation/bare fractions, valid fraction, cloud fractions and days since acquisition. 294 scene renders are cached under `data/raw/s2` (147 MB); 1,859 of 1,939 pilot station-days (95.9 percent) carry a scene, 80 are missing under the cloud rule, and the mean age of the newest usable scene is 2.4 days (maximum 20).

Rendered crops were also sent to Jev with the images header. Two pre-freeze feasibility calls verified the API: a 422 for a malformed Score criteria list, then a valid desert judgment in 0.5 seconds (1,465 input and 89 output tokens). Thirty-eight judgments were cached in total before the gate below stopped collection, of which 21 answered cleanly and 17 failed the experiment's own ordinal-consistency check that the reported score equals the probability-weighted mean; the parser now keeps the score and records the gap.

## Jev image pilot: negative result

The pilot's quality gate failed, before any ablation number existed. On a Lane County, Oregon crop showing farmland, forest and a thin river (numerical water fraction 0.002), Jev v1.13.0 reported water covering 0.92 to 0.97 of the crop on four separate scenes, and a scale-aware rewrite of the instructions still returned 0.95 on the same image, so the error is not a prompt artifact. Across the 19 judged scenes with numerical water fraction below one percent, Jev claimed water extent above 0.5 on 8 of them. Two dry Albuquerque calls were plausible (0.17 to 0.69), so the failure is not uniform, but a feature that claims near-total flooding on a dry river valley cannot carry weight in a hazard model. Judgment quality is recorded in `pilot-quality.json`, and Jev was dropped from the architecture exactly as the task's fallback clause requires.

Operationally the image path is comfortable: 0.5 to 0.8 seconds per call, about 1,550 tokens per call, no separate image rate limit. Cost is reported as tokens (30,765 input and 1,869 output for the 21 clean calls) because the documentation index publishes no price; the failed-ordinal responses were still billed by the provider even though the experiment discarded them, which is the main cost risk in using this API.

## Ablation

Split inside the fresh window, chronological and station-disjoint: fit on July 1 to August 31 development station-days (927 rows, 99 positives), test on September 1 to October 14 development station-days (660 rows, 57 positives), geographic slice on September 1 to October 14 geographic-test stations (352 rows, 5 positives). One worker process per candidate; TabPFN on the MPS device with the benchmarked configuration.

| Candidate | Test AP | Precision | Recall | Alert rate | False alerts /100 | Matched-budget recall | Matched false alerts /100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weather_xgboost (16 features) | 0.2892 | 0.256 | 0.404 | 0.136 | 10.15 | 0.386 | 9.85 |
| weather_sat_xgboost (30) | 0.3151 | 0.314 | 0.474 | 0.130 | 8.94 | 0.509 | 9.85 |
| weather_sat_tabpfn (30) | 0.2565 | 1.000 | 0.035 | 0.003 | 0.00 | 0.614 | 17.58 |

Average-precision difference against weather-only, 1,000 station-day bootstrap resamples, seed 42:

| Candidate | Delta AP 95 percent |
| --- | --- |
| weather_sat_xgboost | [-0.022, +0.080] |
| weather_sat_tabpfn | [-0.105, +0.038] |

The conventional satellite features moved every directional metric the right way on the temporal slice: higher AP, higher precision, higher recall, fewer false alerts, better Brier (0.0966 against 0.1011), and at the matched false-alert budget recall rose from 0.386 to 0.509. The bootstrap interval on the AP difference still includes zero, so the frozen gate is not met and the pilot reports no measurable incremental value. The TabPFN control is worse on AP and, at its matched budget, buys recall 0.614 only by alerting on 17.6 percent of station-days.

Geographic transfer is uninformative in this window: the unseen-station slice has 5 positives in 352 rows. Weather-only found none of them, the satellite model found one (recall 0.200, AP 0.160 against 0.069), and TabPFN found none. One positive is not evidence of transfer.

## Costs and resources

Everything except Jev is free and open: STAC queries and COG window reads need no credentials, the PNG cache is 147 MB, and the whole pilot directory is 37 MB. Renders took about 10 to 20 seconds per new scene and roughly 90 minutes of wall time for 23 stations; the ablation fits take 0.7 to 1.9 seconds per candidate. The machine never approached its memory budget: peak resident use was under a gigabyte during the build, and the TabPFN control used the MPS configuration benchmarked at 28 MB of device memory.

## Limitations

The window is 15 weeks and was already inspected by the transformer sprint, so this is exploratory. The label is measured gust exceedance, which is only weakly coupled to surface water and vegetation; the satellite features were never expected to be strong here, and the coherent pairing is a water or flood-stage outcome, which is not yet labelled. Cloud and shadow missingness is structural (4.1 percent of rows, concentrated in autumn and the Pacific Northwest), scenes are up to 20 days old, and no satellite acquisition can be shown to have been published at the decision hour beyond the one-day latency allowance. The Jev judgment failure could be specific to this rendering, this prompt set and v1.13.0; a different image product or a box-based formulation was not tested.

## Integration decision

Do not integrate satellite or Jev features. The forecast threshold and the served models stay as they are. The pilot's positive signs (recall at matched budgets, better calibration) justify exactly one follow-up, and it is the one the earlier audits already pointed at: build the flood-Stage exceedance label from USGS gage heights against NWS flood categories, where surface-water features have a causal reason to matter, and re-run this ablation there with a pre-registered protocol. Jev image judgments should not be revisited without a different image formulation and a scene-level truth set to measure them against, because the current evidence says its water claims cannot be trusted.

## Reproduce

```sh
.venv/bin/python -m scripts.multimodal_sprint build          # add --skip-jev to avoid Jev calls
.venv/bin/python -m scripts.multimodal_sprint evidence
.venv/bin/python -m scripts.multimodal_sprint analyze
.venv/bin/python -m pytest tests/test_multimodal_sprint.py -q
```

Artifacts: per-station satellite caches, the Jev judgment cache and its evidence file, the pilot quality table, `features.csv.gz`, per-candidate frames, probabilities, `evaluation.json` and the pilot provenance. The experiment writes only inside its own directory and `data/raw/s2`.

