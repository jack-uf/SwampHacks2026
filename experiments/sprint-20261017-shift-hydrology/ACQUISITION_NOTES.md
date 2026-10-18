# Acquisition and pre-fit review record

The frozen scientific protocol is unchanged. These implementation corrections preceded all model fitting and final-year evaluation.

## Observation certification

The observable decision eligibility rule uses only readings through issue time minus two hours. Outcome certification separately requires the two buffer readings to exist. Missing buffer readings can make a later threshold crossing's preceding quiet interval unknowable. Such decisions are excluded from evaluation, rather than assigned negative outcomes. A known crossing inside the buffer remains a negative future-onset label. No buffer value enters a predictive feature or the quiet eligibility test. The regression test covers both cases.

## Inclusive endpoint exposure

USGS interval endpoints are inclusive. Early development requests ending at January 1, 2025, midnight may have acquired that single boundary reading. `boundary-request-exposure.json` lists the exact affected request manifests. Development materialization always removes the endpoint, and the forward split additionally purges labels that cross a year boundary. No model had been fitted when this was discovered. The raw payloads remain preserved. The final year therefore cannot be described as entirely unacquired before selection; its boundary exposure is disclosed even though that reading entered no development row.

Canonical final-quarter requests now stop one second before the following year. Their filenames end in `-bounded`; earlier responses remain evidence of the original requests and are never a fallback input. Other quarterly chunks retain their inclusive request endpoint and explicitly discard that endpoint before concatenation.

## Network diagnosis and recovery

Sandbox DNS resolution fails, whereas approved network execution reaches the official USGS OGC service. The legacy NWIS instantaneous endpoint returned a successful empty response. The current OGC service returns real approved observations and explicit HTTP 429 responses with `Retry-After`. The collector honors integer waits up to one hour, in sleep chunks of at most one minute, with three retries per page. Persistent service failures abort acquisition rather than exclude the station as a data-quality failure. Verified compressed pages are reused on an explicit rerun; changed URLs or bytes fail checksum validation.

`acquisition-interrupted/` preserves an incomplete materialization produced before rate-limit failures were distinguished from data-quality exclusions. It was moved aside before any model fitting, is not read by the experiment, and must not be used for evaluation. Only the canonical dataset whose checksum is bound into family results and `selection.json` is admissible.

## Evaluation lock

Each model family records the development dataset, protocol and implementation hashes. Selection verifies these hashes and every candidate artifact before freezing. Final-year acquisition and inference verify the same lock. This prevents family results from silently referring to different datasets or implementations.

Test acquisition starts in January 2025 without preceding-year warmup. The seven-day history requirement therefore removes the opening week even where observations are complete. Event coverage reports all certified onsets and the subset with eligible decisions; conditional event recall is not whole-year surveillance recall.

## Citation correction

The frozen protocol incorrectly attributes the paper at DOI `10.5194/hess-27-139-2023` to Lees et al. Its authors are Arsenault, Martel, Brunet, Brissette and Mai; its title is *Continuous streamflow prediction in ungauged basins: long short-term memory neural networks clearly outperform traditional hydrological models*. The interpretation remains unchanged: the present experiment conditions on local measured flow and does not test ungauged forecasting. The Gauch et al. multiple-timescale paper citation was checked against its original arXiv metadata. Neither paper supplies a performance claim for this implementation.
