# Pre-model data-quality amendment

October 17, 2026. No model has been fitted and no selection or test metrics inspected. Collection stopped on a physical-range check before producing the training table.

The 2016 forecast warm-up in the first HRES precipitation chunk contains small negative daily totals, with minimum -0.030521303 mm. They are not physically possible rainfall. Their origin has not been certified; numerical differencing or packing is a hypothesis, not an established diagnosis.

Freeze this quality rule before modeling: truncate precipitation in [-0.1, 0) mm to zero for predictors and proxy states, retain the original archive unchanged, count affected values, and reject any value below -0.1 mm. This bounds the transformation to a tenth of a millimetre and prevents negative water input to the snow proxy. It changes no measured discharge or label. The results must disclose this transformation rather than describing processed predictors as exact measurements.

Timestamp resolution also required normalization from decoded seconds to pandas nanoseconds for the daily-order validator. This changes representation only. Tests reproduce both cases.
