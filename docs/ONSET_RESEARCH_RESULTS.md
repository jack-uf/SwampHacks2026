# Emerging high-flow warning results

The new target improves detection of emerging high-flow episodes relative to training on high water alone. It does not establish an advantage for the more complex trajectory hybrid or reliable geographic transfer of alert budgets. Keep all candidates experimental and leave production models unchanged.

The protocol was committed in `62a4433` before collecting fresh outcomes. Implementation and dataset provenance were committed in `528d9df`; candidate selection, code hashes and thresholds were committed in `1c5c46b` before the final evaluation. [Implementation and reproduction](ONSET_RESEARCH.md) explains the data and worker. [The frozen protocol](ONSET_RESEARCH_PROTOCOL.md) defines all decisions and claim criteria.

## Independent population and target

The metadata-only registry reserved 35 gauges in 11 HUC4 drainage regions absent from every role in the previous experiment. The final usable population contains 34 gauges in 10 HUC4s, spanning New England, the Mid-Atlantic and Great Lakes. One reserved gauge, `04233000`, produced no eligible complete approved-flow decisions; no replacement was selected. Development includes 74 usable gauges. Additional development exclusions and zero-row gauges are listed in the committed provenance.

The experiment collected 103,739 development decisions. The actual fitting partitions contain 49,980 fitting decisions with 581 events, 13,562 calibration decisions with 122 events, 13,322 temporal-selection decisions with 107 events, and 4,712 geographic-selection decisions with 56 events. Earlier geographic-selection years are retained in the collected dataset but not fitted. The fresh test covers January 8, 2023 through September 25, 2024: 14,591 eligible decisions, 489 positive decision windows, and 173 distinct station/onset events. Positive prevalence is 3.35%.

An event is measured daily mean discharge above the station's pre-2016 95th percentile after seven accepted quiet days. The prediction window is local D+1 through D+3, with 28 to 77 nominal hours to local-day start across the retained population and seasons. Eligibility requires seven known quiet flows ending D-3. This is an onset target, not persistent high flow, official flood-stage exceedance, site inundation or verified flooding. It requires an established gauge, historical reference and lagged measurements; it is not ungauged-basin prediction.

USGS exact-approved daily mean discharge provides labels and lagged inputs. Caravan MultiMet v1.1 supplies archived ECMWF HRES basin precipitation and temperature forecasts. Forecast rain and temperature, static/seasonal attributes, normalized flow lags, trends and causal snow/liquid-memory features reuse the previous audited implementation. Only 9.98 MB of new source bytes were needed. Source licenses and forecast aggregation limits remain those documented in [the previous hydrology report](HYDROLOGY_RESEARCH_RESULTS.md). No new satellite, river-network, stage-threshold or infrastructure-impact data enter this experiment.

## All frozen candidates

AP is onset average precision. Recall below counts unique events, not repeated decision windows. The alert thresholds target one false alert per 100 decisions on 2022 selection data and are frozen before test evaluation. Their actual test false-alert rates differ; these rows are not a matched test-budget comparison. Brier measures probability error, with smaller values better.

| Candidate | Selection AP | Fresh AP | Brier | Event recall | False alerts /100 decisions | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lagged-flow proximity | .0548 | .0502 | .0324 | 3.5% | 1.460 | 3.6% |
| Linear flow extrapolation | .0507 | .0501 | .0327 | 4.6% | 1.823 | 4.0% |
| Calibrated HRES rain ranking | .2202 | .2572 | .0287 | 35.8% | .802 | 44.0% |
| High-water LightGBM, calibrated for onset | .2479 | .2230 | .0289 | 33.5% | 1.035 | 35.5% |
| Direct onset logistic regression | .2575 | .3235 | .0280 | 50.9% | 1.474 | 40.3% |
| Direct onset LightGBM, weather | .2927 | .2754 | .0281 | 39.3% | 1.083 | 42.1% |
| Direct onset LightGBM, hydrology | .0670 | .0831 | .0322 | 13.3% | 1.570 | 14.6% |
| Direct onset LightGBM, combined | .3036 | .2744 | .0285 | 43.4% | 1.446 | 37.0% |
| **Selected: direct onset LightGBM, snow** | **.3233** | **.3027** | **.0277** | **48.6%** | **1.295** | **40.0%** |
| Residual quantile trajectories + onset head | .2729 | .2830 | .0279 | 37.6% | .966 | 40.8% |

The selected model detects 84 of 173 events, compared with 58 for the high-water classifier. It adds 32 detected events and loses six, for a net gain of 26. It produces 189 false warning decisions versus 151. Its decision-window recall is 25.8%, compared with 17.0%; event recall is higher because several windows can precede one event. Fifty-seven events receive a warning at least 48 hours before the onset day's start.

Median earliest warning among detected events is 52 hours for the selected model and 29 hours for the high-water comparator. This is conditional on different detected subsets. Among the 52 events both detect, the median paired lead difference is zero. The result supports broader early detection, not a claim that every shared event is warned earlier.

The AP gain over high-water training is .0797. Frozen paired 500-draw bootstrap 95% intervals are [.0469, .1038] by HUC4 and [.0344, .1300] by decision month. Event-recall difference intervals are [.0146, .1985] and [.0709, .2238], respectively. These are separate, descriptive cluster analyses, not multiplicity-adjusted guarantees.

The strict claim of an advance beyond the strongest direct onset baseline fails: the selected candidate is that baseline itself, so its self-comparison has zero improvement. The more ambitious hybrid did not earn selection. Logistic regression has the highest final AP, but choosing it now would reuse test outcomes for selection. It remains a prespecified comparator, not a newly selected winner.

## Geographic transfer and false alerts

| Held-out HUC2 | Events | Selected AP | Event recall | False alerts /100 decisions |
| --- | ---: | ---: | ---: | ---: |
| 01, New England | 58 | .3858 | 43.1% | .966 |
| 02, Mid-Atlantic | 96 | .3589 | 54.2% | 1.205 |
| 04, Great Lakes | 19 | .1399 | 36.8% | 1.672 |

Selected AP is .2755 in 2023 and .3300 in 2024. Mean station AP is .3512, versus .2945 for the high-water comparator; two stations have no positive test windows and undefined AP. The committed evaluation retains every station, year, HUC2 and reliability-bin count. The Great Lakes result remains weak and has only 19 events.

The conservative maximum-region threshold reduces selected false alerts to .123 per 100 decisions but detects only 12.1% of events. Thus conservative geographic alert control can destroy usefulness. The pooled 5% policy also transfers poorly: its actual test rate is 7.875 false alerts per 100 decisions.

Delayed station adaptation preserves aggregate event recall, increasing it from 48.6% to 49.1%, but increases false warning decisions from 189 to 205 and the rate from 1.295 to 1.405 per 100. Great Lakes recall falls to 31.6%. It fails the frozen improvement criterion. The high-water comparator's adaptation likewise increases false alerts. An empirical source prior plus delayed local negatives is insufficient here; no conformal or regional guarantee is claimed.

For a descriptive matched-budget diagnostic, test-label oracle thresholds give the selected model 42.2% event recall and the high-water comparator 33.5%, both at .973 false alerts per 100. Rainfall ranking reaches 43.4% at .994; the trajectory hybrid reaches 38.7% at .980. The selected ML model therefore does not show better event recall than simple forecast rainfall at approximately the same test alert burden. These oracle policies are not deployable and were never used for selection.

## What the trajectory hypothesis did and did not achieve

The hybrid predicts residual normalized log flow at three leads and three quantiles, with an independent-year logistic onset head. Its marginal central 80% intervals cover 81.9%, 81.5% and 81.5% of test outcomes at D+1, D+2 and D+3. These are marginal, not simultaneous trajectory or extreme-event coverage.

Median pinball loss is .03288, .03562 and .03746, versus .03924, .04212 and .04383 for lagged-flow persistence. This is a real improvement in continuous-flow prediction. Nevertheless, AP .2830 and event recall 37.6% do not beat direct snow-aware onset classification. Better average trajectory loss does not automatically produce better rare-event warnings.

Hydrology-only and extrapolation candidates are weak on this eligibility-screened target. Causal weather forecasts contain much more onset information than lagged quiet flow. Adding hydrology to weather reduces fresh AP from .2754 to .2744, while adding snow features raises it to .3027. These single frozen ablations suggest a useful snow-memory contribution but do not establish a universal causal benefit or its statistical significance against the weather-only model.

Graph, temporal-transformer and foundation candidates were not added: no independently audited upstream network or compatible checkpoint/data contract was established, and this experiment exposes a target/calibration gap without evidence that additional architecture would fix it. The reviewed papers and their exact retrieval manifests are linked in the protocol. None establishes a false-alert theorem for this dataset.

## Verification, limits and decision

Training all ten candidates takes 7.42 seconds at 452 MB peak RSS; final evaluation including both cluster bootstrap schemes takes 8.55 seconds at 273 MB on this arm64 Mac. These are process measurements for cached local data, not download time or a portable hardware benchmark. Workers use two threads. The existing forecast and outcome caches remain untouched.

`audit_onset.py` independently reconstructs every one of the 14,591 final event labels and local-day lead times from original USGS records, checks all saved alert-event counts, and reproduces all ten saved probability vectors from their frozen artifacts to absolute tolerance 1e-14. The full Python suite passes 174 tests. Preserved production and prior-experiment hashes pass. The frontend production build passes.

Historical publication timestamps and revisions remain unverified. D-3 discharge and D-1 forecast availability are explicit assumptions. Adaptive feedback assumes labels are usable eight days later. Daily means miss within-day crests, station normalization needs long historical records, warning windows overlap, and a finite convenience sample does not establish national transfer. A false onset alert can coincide with other dangerous conditions. The prior study's AP .884/.842 is a different persistent-high-flow target and cannot be compared numerically with this onset AP.

Do not integrate these models into ratings or the 0-10 index. The defensible contribution is a reproducible onset-specific benchmark with independent geographic evidence, paired uncertainty, warning-lead accounting, a useful trajectory forecast, and falsified architecture/adaptation hypotheses. It does not meet the frozen criterion for a new architecture outperforming direct onset learning.

The strongest next research direction is an hourly rise-and-crest target with audited as-issued discharge and forecast timestamps, reserving a new Great Lakes cohort before model changes. Compare simple forecast-rain warnings, the direct onset model and event-weighted trajectory learning at the same frozen alert costs. If an upstream graph and travel times can be independently verified, add upstream residual innovations as one separate ablation. Use those untouched outcomes to test whether localized calibration and propagation features recover the events missed here before considering a hazard-index contribution.
