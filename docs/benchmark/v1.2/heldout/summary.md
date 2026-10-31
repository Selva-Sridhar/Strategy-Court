# Strategy Court benchmark (heldout)

Cases: 110 - rubric 1.2.0+90f6459e

| Metric | Value |
|---|---|
| verdict accuracy | 85% (93/110, 95% CI 77%-90%) |
| flaw conviction rate | 77% (46/60, 95% CI 65%-86%) |
| flaw rejection rate | 92% (55/60, 95% CI 82%-96%) |
| false conviction rate genuine | 15% (3/20, 95% CI 5%-36%) |
| false acquittal rate no edge | 0% (0/30, 95% CI 0%-11%) |

| Ground truth | Target | Meets target | Convicted | Acquitted | Expected tool flagged |
|---|---|---|---|---|---|
| genuine | not_convicted | 85% (17/20, 95% CI 64%-95%) | 15% (3/20, 95% CI 5%-36%) | 45% (9/20, 95% CI 26%-66%) | n/a |
| no_edge | not_acquitted | 100% (30/30, 95% CI 89%-100%) | 7% (2/30, 95% CI 2%-21%) | 0% (0/30, 95% CI 0%-11%) | 100% (30/30, 95% CI 89%-100%) |
| lookahead | convicted | 100% (10/10, 95% CI 72%-100%) | 100% (10/10, 95% CI 72%-100%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| scaler_leak | convicted | 100% (10/10, 95% CI 72%-100%) | 100% (10/10, 95% CI 72%-100%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| best_of_n_disclosed | convicted | 90% (9/10, 95% CI 60%-98%) | 90% (9/10, 95% CI 60%-98%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| best_of_n_hidden | convicted | 10% (1/10, 95% CI 2%-40%) | 10% (1/10, 95% CI 2%-40%) | 50% (5/10, 95% CI 24%-76%) | 90% (9/10, 95% CI 60%-98%) |
| zero_cost | convicted | 100% (10/10, 95% CI 72%-100%) | 100% (10/10, 95% CI 72%-100%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| survivorship | convicted | 60% (6/10, 95% CI 31%-83%) | 60% (6/10, 95% CI 31%-83%) | 0% (0/10, 95% CI 0%-28%) | 80% (8/10, 95% CI 49%-94%) |

Confusion (rows = ground truth, columns = verdict):

| | acquitted | not_proven | convicted |
|---|---|---|---|
| genuine | 9 | 8 | 3 |
| no_edge | 0 | 28 | 2 |
| lookahead | 0 | 0 | 10 |
| scaler_leak | 0 | 0 | 10 |
| best_of_n_disclosed | 0 | 1 | 9 |
| best_of_n_hidden | 5 | 4 | 1 |
| zero_cost | 0 | 0 | 10 |
| survivorship | 0 | 4 | 6 |
