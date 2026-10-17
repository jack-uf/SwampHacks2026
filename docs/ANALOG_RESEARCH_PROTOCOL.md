# Historical-transition priors for emerging high water

This experiment tests whether a river's historical transition distribution improves weather-driven onset warnings at new watersheds. The connected-river study found insufficient verified development coverage. This alternative needs no unverified upstream joins. All candidates remain outside production.

## Research rationale

Local flow dynamics may transfer poorly through a shared classifier. Estimate each station's conditional transition risk from its own approved 2000-2015 observations, then learn a shared correction from weather and recent flow. This is transfer to previously unseen watersheds with historical gauge records, not prediction at ungauged rivers. No recent target-station labels enter training, calibration or threshold selection.

Lall and Sharma's *A Nearest Neighbor Bootstrap For Resampling Hydrologic Time Series*, DOI 10.1029/95WR02966, motivates nearest-neighbor conditional hydrological distributions. The publisher-deposited abstract was retrieved through Crossref and retained with its request manifest. It describes resampling, including monthly streamflow; it does not validate this daily onset implementation. The response study's retained primary-source review of entity-aware and differentiable hydrology motivates catchment conditioning. This study tests a combination of established methods and makes no architectural priority claim.

## Frozen population and availability

Reserve all 31 basins in the upstream study's metadata-only `expanded-metadata.json` selection. They have no positive-area intersection with the 215 previously reserved gauge polygons or with one another under the retained simplified NLDI geometries. Reverify the polygons and indexed gauge identity. This is catchment separation, not separation of all climate regions. Do not replace gauges following source-quality exclusions.

Reuse the response study's exact development data, fitting in 2017-2020, calibration in 2021 and temporal/geographic selection in 2022. All these cohorts are development data already inspected. Fit reference transitions only on dates whose complete D+3 target ends before January 1, 2016. Collect development reference histories and select candidates first. Commit implementation, artifacts, calibration and thresholds before downloading any new station's discharge. Then evaluate the 31 reserved stations from January 8, 2023 through September 25, 2024 once.

Reuse approved USGS daily mean discharge, pre-2016 p95 reference, exact calendar lags and onset rules. Require at least 3,000 accepted reference days, 2,000 consecutive reference pairs for response descriptors, and 1,000 complete eligible analog windows. New onset means daily mean discharge above p95 following seven quiet days. At issue time, seven known quiet days end at D-3. Targets are onset on local D+1 through D+3. Report lead to local onset-day start. This is high flow, not verified flooding.

Reuse archived ECMWF HRES D-1 forecasts and existing causal rain/snow memories. Final approved discharge may contain retrospective revisions. D-3 measurement availability and D-1 forecast dissemination remain assumptions without as-issued historical records. No observed future rain, satellite measurement, current flood-stage threshold or recent local label is a predictor.

## Historical analog distribution

For each station, normalize discharge as log1p(flow / pre-2016 p95). Describe an issue by q(D-3), the differences q(D-3)-q(D-4), q(D-3)-q(D-7), q(D-7)-q(D-14), and seasonal sine/cosine at issue date. Standardize using that station's eligible pre-2016 library only. Select the 128 Euclidean nearest complete library windows, uniformly weighted, with deterministic library date order. Do not tune the distance, neighbor count or smoothing.

Return onset probability `(neighbor positives + 1) / 178`, corresponding to a Beta(1,49) prior; the mean and standard deviation of normalized flow at D+1, D+2 and D+3; and mean neighbor distance. The prior shrinks sparse local event counts toward two percent. These quantities are empirical historical distributions, not confidence intervals or physical simulations. Reject incomplete issue histories instead of shifting across missing days. All candidates use identical retained rows.

## Fixed comparisons and selection

Retain the response study's frozen rainfall, logistic, direct LightGBM and pooled neural artifacts. Add calibrated proximity q(D-3), a strict persistence warning q(D-3)>p95, and four analog candidates:

1. Local analog onset prior alone.
2. Hydrology-only LightGBM using the existing hydrology inputs plus analog outputs.
3. Combined weather/hydrology/snow LightGBM plus analog outputs.
4. The same combined LightGBM trained as a log-odds correction to the analog prior. Pass prior log odds as training `init_score` and explicitly add them to the raw tree score at inference.

Use the existing 200 trees, learning rate .04, seven leaves, depth three, two threads and seed 42. Calibrate new candidates on 2021 development log odds with logistic regression C=1. Do not refit the frozen old baselines. Choose the analog candidate with highest mean AP over the two 2022 selection cohorts, breaking ties by lower mean Brier then name. Choose a comparator from direct LightGBM and pooled neural by the same rule. Report all models regardless of winner.

Freeze pooled 2022 thresholds targeting .5, one and two false alerts per 100 decisions and the maximum-HUC2 threshold targeting one. Report empirical calibration and thresholds without claiming a distribution-free guarantee. No test-label feedback enters an operational policy. Show test-label matched-one-percent results only as non-deployable oracle diagnostics.

## Evaluation and success criteria

Report AP, Brier, log loss, reliability bins, precision, decision recall, distinct-event recall, false alerts per 100 decisions, events warned at least 48 hours ahead, and per-station/HUC2/year metrics. Compare first-alert lead only for common detected events and report added/lost events separately. Use 500 seed-42 paired HUC4 bootstrap draws and 500 calendar-month draws for AP and event-recall differences using the established response study convention.

A substantive improvement requires at least 50 events across ten usable HUC4 groups; positive lower AP-difference bounds under both bootstrap schemes; at least five percentage points more event recall at the frozen one-percent policy; pooled false alerts at most 1.5 per 100 decisions; and no HUC2 above three per 100. Compare the selection winner against the prespecified comparator. Report the prior-only, hydrology-only, combined and offset ablations without substituting a final-test winner. No result authorizes production use while availability and actual-flood validity remain unresolved.

## Resources, verification and stop

Run LightGBM and Torch in separate processes. Reuse the existing neural artifact without retraining. CPU only, two threads, no new dependencies. Limit new sources to 100 MB and each worker to 6 GB RSS and 15 minutes. Record package versions, source and artifact hashes, fit and inference time, RSS and exclusions. Preserve production model and safety code hashes.

Test reference-only fitting, calendar-gap rejection, causal issue features and correct offset inference. Audit source hashes, reference targets, polygon independence and dataset/artifact reproducibility. Run repository tests, document the fixed comparison and commit. Stop after this experiment; do not optimize against its final geography.
