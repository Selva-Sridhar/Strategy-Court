# Strategy Court benchmark (heldout)

Cases: 110 - rubric 1.1.0+52e6670e

| Metric | Value |
|---|---|
| verdict accuracy | 79% (87/110, 95% CI 71%-86%) |
| flaw conviction rate | 68% (41/60, 95% CI 56%-79%) |
| flaw rejection rate | 88% (53/60, 95% CI 78%-94%) |
| false conviction rate genuine | 5% (1/20, 95% CI 1%-24%) |
| false acquittal rate no edge | 10% (3/30, 95% CI 3%-26%) |

| Ground truth | Target | Meets target | Convicted | Acquitted | Expected tool flagged |
|---|---|---|---|---|---|
| genuine | not_convicted | 95% (19/20, 95% CI 76%-99%) | 5% (1/20, 95% CI 1%-24%) | 90% (18/20, 95% CI 70%-97%) | n/a |
| no_edge | not_acquitted | 90% (27/30, 95% CI 74%-97%) | 13% (4/30, 95% CI 5%-30%) | 10% (3/30, 95% CI 3%-26%) | 93% (28/30, 95% CI 79%-98%) |
| lookahead | convicted | 100% (10/10, 95% CI 72%-100%) | 100% (10/10, 95% CI 72%-100%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| scaler_leak | convicted | 100% (10/10, 95% CI 72%-100%) | 100% (10/10, 95% CI 72%-100%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| best_of_n_disclosed | convicted | 60% (6/10, 95% CI 31%-83%) | 60% (6/10, 95% CI 31%-83%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| best_of_n_hidden | convicted | 0% (0/10, 95% CI 0%-28%) | 0% (0/10, 95% CI 0%-28%) | 70% (7/10, 95% CI 40%-89%) | 40% (4/10, 95% CI 17%-69%) |
| zero_cost | convicted | 100% (10/10, 95% CI 72%-100%) | 100% (10/10, 95% CI 72%-100%) | 0% (0/10, 95% CI 0%-28%) | 100% (10/10, 95% CI 72%-100%) |
| survivorship | convicted | 50% (5/10, 95% CI 24%-76%) | 50% (5/10, 95% CI 24%-76%) | 0% (0/10, 95% CI 0%-28%) | 90% (9/10, 95% CI 60%-98%) |

Confusion (rows = ground truth, columns = verdict):

| | acquitted | not_proven | convicted |
|---|---|---|---|
| genuine | 18 | 1 | 1 |
| no_edge | 3 | 23 | 4 |
| lookahead | 0 | 0 | 10 |
| scaler_leak | 0 | 0 | 10 |
| best_of_n_disclosed | 0 | 4 | 6 |
| best_of_n_hidden | 7 | 3 | 0 |
| zero_cost | 0 | 0 | 10 |
| survivorship | 0 | 5 | 5 |
