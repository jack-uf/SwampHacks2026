# Amendment 1: a class-balanced loss variant for the hybrid network

This amendment adds one candidate before any model is fitted and before any final-year observation is acquired.

The frozen candidate list pairs each MLP architecture with both threshold rules, which isolates the operating point from the architecture. It does not test whether the training objective itself matters. The task asks about both, so `mlp_hybrid_balanced` adds a positive-class weight of negatives over positives to the hybrid network's binary cross-entropy loss. It is the same architecture on the same rows with the same early stopping, so any difference is attributable to the objective rather than to capacity.

No other candidate, feature, threshold rule, partition or gate changes. The amendment is recorded before the first fit, so it cannot have been chosen against a result.

Deletion criteria: if a future study shows the weighted objective changes only the calibrated operating point rather than the ranking, drop the variant and keep the unweighted hybrid.
