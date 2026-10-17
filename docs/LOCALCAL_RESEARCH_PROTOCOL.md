# Local probability adaptation for emerging high water

This study tests whether station-specific calibration makes the frozen historical-transition hybrid more reliable at physically separate catchments. It changes calibration rather than fitting a larger predictor. Models remain outside production.

## Hypothesis and evidence

The preceding analog study improved ranking but failed its regional warning criteria. A shared probability map may miss local onset frequencies. Estimate a station intercept from one earlier year, shrink it toward the shared calibration, then apply it without feedback during the final evaluation. This is adaptation to gauges with measured history, not prediction at ungauged rivers.

Kull, Silva Filho and Flach (AISTATS 2017, PMLR 54, pages 623-631) describe calibration maps and the overfitting risk of flexible calibration on small datasets. Their publisher abstract is retained in `research/beta-calibration.html` with a request manifest. It motivates separating calibration from discrimination, but does not establish hydrological transfer. We use a simpler intercept correction to preserve each station's ordering and reduce the number of fitted parameters. A normal prior on the intercept supplies partial pooling; no exchangeability or distribution-free coverage guarantee is claimed.

## Population and independence

Before requesting discharge, reserve up to 24 unused CAMELS gauges using the metadata-only cohort script. Exclude every previously registered or requested gauge, including the four upstream sources. Require a verified NLDI gauge identifier within one kilometre of inventory coordinates and a valid upstream polygon with zero positive-area intersection with every previously reserved polygon and with every new selected polygon. Use one gauge per HUC4, deterministic SHA256 order of `localcal-20261017-<station>`, and at most 80 candidate mapping attempts. Retain exclusions and geometry hashes. Freeze the final registry and protocol hashes in a commit before discharge access.

Earlier studies exhausted most unused HUC4 groups. This study permits new disjoint catchments inside represented HUC4 groups. It tests new catchments within represented broad regions, not entirely unseen regional climates. Simplified polygons limit the independence audit. Require ten usable HUC4 groups and 50 final events for a substantive claim. Do not replace gauges after inspecting outcomes.

## Sources, targets and chronological boundaries

Reuse the approved daily mean USGS discharge parser, pre-2016 p95 reference, exact calendar lags, local onset-day definitions, HRES basin forecasts and analog features. New onset is daily mean flow above the reference after seven quiet days, occurring on local D+1 through D+3. At issue time seven observed quiet days end at D-3. This predicts high flow, not verified flooding. Final revised observations and assumed D-3 measurement and D-1 forecast availability cannot establish an as-issued replay.

Use 2021 decisions from January 8 through December 24 for each station's calibration. The last calibration target precedes 2022. Request discharge through December 31, 2021 first. Before opening any final outcomes, commit the code, selected method, thresholds and station intercepts. Then request the final histories and evaluate January 8, 2023 through September 25, 2024 once. All pre-2023 calibration and selection populations are development data. Never use final labels to adjust a threshold or intercept. Preserve identical complete rows across comparisons. Exclude a station without at least 150 complete 2021 decisions. Constant-label calibration records remain eligible under the specified positive penalties.

Reuse the analog study's development registry and frozen offset model. Reconstruct its 2021 calibration records from immutable sources. Select using its two already inspected 2022 temporal and geographic cohorts. Retain the same fixed response-study rainfall, logistic, direct LightGBM and pooled neural models as comparators. Existing GFS and transformer studies do not supply a river forecast. No archived authoritative river forecast with matching historical availability has been verified; rainfall is only a weather comparator.

## Fixed methods and selection

Let p be the frozen hybrid's globally calibrated probability and z its clipped log odds, using clipping at 1e-6. For each station fit the scalar b minimizing summed binary log loss of sigmoid(z+b) plus lambda*b*b/2 on 2021 only. Compare lambda values .1, 2 and 10 against unchanged global calibration. Solve within [-30,30] by bounded scalar minimization and reject invalid inputs. No new neural training, slope, seasonal terms, satellite inputs or tuning grid is allowed. Local event counts and sample sizes must travel with the intercepts.

For every method freeze pooled 2022 cutoffs targeting .5, one and two false warnings per 100 decisions. Also record the maximum HUC2 cutoff targeting one as a conservative diagnostic. Select among the three local methods by highest mean event recall over the two selection cohorts, considering only methods with pooled false warnings at most 1.5 per 100 and maximum HUC2 at most three. If none qualify, select by lowest worst-region false-warning rate, then mean event recall. Break remaining ties by lower mean Brier, then method name. Keep all comparisons visible and never choose a final-test winner.

## Evaluation and decision

Report AP, Brier, log loss, reliability bins, distinct-event recall, precision, false warnings per 100 decisions, event lead time and per-station/HUC2/year results. Keep decision-level false warnings distinct from merged operational warning episodes. Compare first warning lead only on common detected events and list added and lost events separately. Use the existing seed-42 paired 500-draw HUC4 and month bootstrap for AP and event-recall differences, and add paired Brier differences. Overlapping onset windows and adjacent catchments are not independent trials; neither bootstrap fully captures climate uncertainty.

A useful improvement over unchanged global hybrid requires at least 50 events in ten catchments; at least five percentage points greater event recall; a positive lower event-recall difference bound in both bootstraps; lower Brier with a negative upper difference bound in both; pooled false warnings at most 1.5 per 100; no HUC2 above three; and no reduction in median paired lead on common events. AP may change across stations, but within-station ordering must remain unchanged. Report failures without revising these criteria. A retrospective success still needs verified input availability and prospective validation before contributing probabilities to PlanIT's 0-10 index. Mandatory safety rules remain independent.

## Resources and reproducibility

Use CPU with two threads, the existing process separation for neural baselines, no new dependencies, at most 100 MB of new source responses, and at most 6 GB peak RSS per worker. Stop a worker beyond 15 minutes. Record sources, exclusions, code, artifact and dataset hashes, package versions and runtime. Tests cover calibration behavior and invalid inputs, time-boundary rejection and frozen replay. Verify every source hash, geographic separation and production hashes, then run repository tests. Stop after the predeclared experiment rather than optimize against its final outcomes.
