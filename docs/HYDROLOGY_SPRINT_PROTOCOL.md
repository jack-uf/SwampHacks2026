# Shift-level hydrologic sprint protocol

This protocol freezes how a shift-level flood model would be built and judged. It exists because the repository can run the pipeline but cannot currently acquire the data: the development sandbox has no network, and no archived NWPS gauge or USGS discharge series for the served sites is committed. Nothing here is a result.

## Question

Can a six, twelve or twenty-four hour high-water warning be produced from the gauge series, rainfall accumulation, terrain and weather forecasts the application already touches, at a false-alert burden a supervisor would accept?

## Signals to acquire, in order of expected value

1. Observed gauge stage series per served site, hourly, from the NWPS map service the provider already calls, plus the published flood-stage categories for the same gauge.
2. The gauge stage forecast the same service publishes, kept separate from observations so a forecast is never labelled as an outcome.
3. Antecedent rainfall from the hourly forecast archive the planner already uses, and the soil moisture layers the assessment already carries.
4. The five-point terrain relief and gradient already computed for each site.

`providers.gauge_trend` computes direction, six and twenty-four hour change and headroom to flood stage from a reading series, and abstains when there are fewer than two usable readings. It is the first candidate feature and is covered by contract tests.

## Labels and leakage

The label is the gauge crossing its published flood stage inside the horizon, from observed stage only. Features at decision time may use observations up to the issue hour and forecasts issued no later than that hour; anything later is leakage. Gauge series are split by watershed group and forward in time, with a holdout never used for threshold selection. Sites whose gauge has no published flood stage are excluded rather than given an invented one.

## Baselines and promotion criteria

Compare, in this order: persistence of the last stage, persistence of the trend, rainfall-only, and gradient boosting on the assembled features. Promotion requires all of: average precision above the best baseline with a bootstrap lower bound above zero; event recall at the operational budget at least five points above the baseline; no region above three false alerts per hundred decisions; headroom and trend contributions visible in an ablation; and a genuinely untouched holdout. A shift-level model that clears these could appear in an assessment as a disclosed, non-overriding line; it would never replace the deterministic rules.

## What is deliberately not claimed

No shift-level model exists yet, no threshold has been selected, and no comparison has been run. The gauge trend is a feature, not a forecast, and a rising stage is not a flood.
