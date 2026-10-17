# Amendment 1: round-trip parsing of saved predictions

This amendment records a loader correction made after the fold selection was committed, and why it changes no result.

The first report attempt rejected its own integrity check. One 2021 control decision had a probability exactly equal to the frozen threshold derived from it, so the saved alert is true while a re-read of the saved probability fell one unit in the last place below that threshold. The cause is pandas' default float parser, which is not round-trip exact. The worker now reads saved predictions with `float_precision='round_trip'`, matching how the local-calibration study reads its frozen predictions.

Nothing else changed. The four candidates, their feature sets, the fit recipe, calibration, thresholds, selection rule and the seven gate criteria are as frozen. Two reproductions record that the correction is inert:

- The development collection was rebuilt and returned the same dataset digest `9fc9c6ff0e18ba1fd95667a5e662b3696d9ffd0704f8cc7b35d4c24fed9e16fc` and identical source manifests.
- Selection was rerun and returned the same eight artifact and prediction checksums, and the same summary: control mean AP .29750 and Brier .024342, bridge .29601 and .024378, revision .30313 and .024220, combined .30649 and .024172.

Only the protocol digest and the wall-clock fields differ between the two runs.

The correction is a parsing fix in the reporting loader. It cannot alter a fitted probability, a threshold, an alert or a gate outcome.
