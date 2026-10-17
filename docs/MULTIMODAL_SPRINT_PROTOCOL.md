# Multimodal environmental hazard pilot: frozen protocol

Frozen October 17, 2026, before the satellite feature build and before any ablation number is computed. The pilot asks a narrow question: do satellite context features, conventional or Jev-derived, add measurable predictive value to measured wind-gust exceedance at power-plant weather stations, especially where the model has not fitted those stations?

No fresh holdout exists. The wind sprint inspected 2026 H1, and the transformer sprint spent the July to October 2026 window. Every number here is **exploratory pilot evidence** for deciding whether the multimodal path is worth a properly powered experiment; it cannot promote a model by itself. Nothing is wired into the application.

## Data and decision-time contract

Satellite source: Sentinel-2 L2A COGs on the open `sentinel-cogs` bucket, discovered through the Element 84 STAC API (no credentials). For a decision at 00:00 UTC on day D minus 1 for target day D, candidate acquisitions are items intersecting the station point in the 21 days ending one day before the decision (publication latency allowance), with STAC scene cloud cover below 60 percent, at most the four most recent. The chosen item is the one with the lowest cloud fraction inside the 5.12 km window, ties broken by newest acquisition; if the best candidate still exceeds 50 percent masked pixels, or no candidate exists, the station-day carries a missing-satellite indicator instead of an imputed value.

Rendered per (station, item): 512 by 512 pixels at 10 m, true colour (red, green, blue at 2 to 98 percent stretch), masked pixels left near white. Conventional features per station-day, all computed only from the chosen window and its SCL layer: days since acquisition, scene cloud cover, local cloud fraction, valid fraction, NDVI mean and 10th and 90th percentiles, NDWI mean and 10th and 90th percentiles, SCL water, vegetation and bare fractions.

Jev: the pinned text model `jev-1.13.0` also accepts images. One request per rendered scene, with the `TypeSafe-Danger-Zone-Images: true` header and the image in the request assets, asks three questions: whether surface water is visible (Noul), how much water is present on a 0 to 2 scale (Score), and which single description matches the ground (Choice: dry bare, vegetated, water dominant, cloud or masked). Judgments are unverified experimental features; they never set a rating. Latency and usage tokens are recorded per call, and no retries are made to avoid duplicate paid inference. The pilot feasibility checks before this freeze were two calls: one rejected for a malformed Score criteria list (HTTP 422, not billed as answered) and one valid response on the Albuquerque desert crop in 0.5 seconds with 1,465 input and 89 output tokens.

## Population and split

Stations: the 15 development stations of the frozen wind protocol for fitting and the temporal slice, plus the 8 geographic-test stations as an unseen-station slice.

Split inside the July 1 to October 14, 2026 window, chronological: fit on July 1 to August 31 station-days; test on September 1 to October 14. Geographic-test stations appear only in the test slice, never in fitting. This keeps the pilot honest within a window that earlier sprints already inspected: the comparison is between feature sets on identical rows, not a fresh claim.

## Candidates

One worker process per candidate, mirrored from the transformer sprint so xgboost, lightgbm and torch never share a process. Every candidate uses the same fitting rows and the same split:

| Name | Features |
| --- | --- |
| weather_xgboost | The frozen 16 weather features (the wind sprint's extended set) |
| weather_sat_xgboost | Weather plus the conventional Sentinel-2 features |
| weather_jev_xgboost | Weather plus the Jev judgments |
| weather_sat_jev_xgboost | Weather plus both satellite sets |
| weather_sat_jev_tabpfn | The combined set on TabPFN v2, MPS device, the benchmarked configuration |

Missing satellite or Jev values stay missing; the tree models handle that natively. No imputation, no calibration refit for this pilot, and no hyperparameter search. TabPFN runs on MPS with the benchmarked settings; the other candidates run on CPU. Weather-only baselines for the other libraries are out of scope for a pilot with this row count.

## Metrics and gates

Average precision is the primary metric with a percentile bootstrap over station-days (1,000 resamples, seed 42) comparing each satellite candidate with weather_xgboost on the test slice. Recall, precision, alert rate and false alerts per 100 station-days are reported at a matched-budget operating point: the highest threshold whose fit-slice false-alert count does not exceed the weather-only candidate's, chosen once on the fit slice only. Calibration is reported as Brier score and ten-bin reliability. Geographic performance is reported per station and per state for the unseen-station slice.

The pilot shows measurable incremental value only if a satellite candidate beats weather_xgboost on average precision with a bootstrap interval whose lower bound exceeds zero on the test slice, and does not lose recall at the matched budget. Anything less is reported as no signal, and the recommendation becomes conventional features only, a larger fitting population, or stopping.

## Constraints and stop conditions

Respect the machine budget: one process per model family, MPS for TabPFN, cached scenes under `data/raw/s2/`, and bounded Jev calls (one per cached scene, no retries). Stop the build if the scene cache exceeds 400 scenes or the Jev token total exceeds 600,000 without a decision. Preserve the served models, the application, and every earlier experiment artifact. Run the focused tests, commit verified work, and report costs, limits and the integration decision honestly. Do not force Jev into the architecture if it adds nothing.

