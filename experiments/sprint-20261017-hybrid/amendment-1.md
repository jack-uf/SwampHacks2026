# Amendment 1: randomized trees replace the pooled neural expert

This amendment replaces the architecturally distinct expert before any evaluation-fold result was produced.

The frozen design listed the pooled response mixture as the second expert. Three separate runs of `scripts.hybrid_research run` stopped inside that expert's training: each wrote the gbm artifact, then made no further progress for 3.5, 13 and 3.5 minutes respectively, with resident memory far below what a loaded torch runtime holds, and no epoch line printed. A standalone `import torch` in the same environment completed in 1.5 seconds and reported torch 2.14.1, so the stall is in the training path rather than the interpreter's ability to load torch at all.

The replacement is `ExtraTreesClassifier` with 300 trees, square-root feature sampling and minimum leaf five, fitted and calibrated exactly like every other expert. Bagged randomized trees differ from the boosted shallow trees of the gbm expert in their inductive bias, which is the property the complementarity question needs, and they carry no new dependency.

Nothing else changed: the feature set, the calibration step, the combiner definitions, the threshold and selection rules, the gate criteria and the folds are as frozen. The amendment is recorded before the fold 2022 evaluation exists, so it cannot have been chosen against its result.

Deletion criteria: if the pooled response mixture becomes trainable within the study's resource budget in this environment, rerun this study with it in place of the randomized trees and compare; this experiment's protocol tracks that follow-up.
