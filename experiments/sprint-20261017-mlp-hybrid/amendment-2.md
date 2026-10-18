# Amendment 2: the event-maximizing threshold is replaced by a ceiling rule

This amendment changes one threshold rule before any model is fitted and before any final-year observation is acquired.

The frozen protocol paired the row rule with a rule that maximizes unique validation events under the one-percent false-alert budget. On inspection before fitting, that rule is equivalent to the row rule and tests nothing. Row true positives and unique events detected are both monotone non-decreasing as the threshold falls, and the feasible set where false alerts stay inside the budget is an interval, so both rules select the same lowest feasible threshold. Two rules that always agree cannot separate an effect.

The replacement is a ceiling rule: take the lowest validation threshold whose pooled false-alert rate is at most 1.5 per 100 decisions and whose every validation region is at most 3 per 100, which are the promotion gate's own operational ceilings. Because recall is monotone in the threshold, this is the highest-recall operating point the gate permits. It deliberately spends more false-alert budget than the row rule, which self-imposed a one-percent target the gate does not require. That self-imposed target is the most plausible reason the earlier MLP detected fewer events than the linear baseline while ranking better, so the rule is the hypothesis under test and the row rule remains its control.

Both rules still see only pooled 2024 validation rows, and neither can use final-year outcomes. No candidate, feature, architecture, partition or gate changes.

Deletion criteria: if a future study shows the ceiling rule raises false alerts past the pooled ceiling on independent data without recovering events, drop it and keep the row rule.
