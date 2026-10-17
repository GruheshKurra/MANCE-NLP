# Learning-rate checks

Short runs used to pick the learning rate before the main experiments. The choice was made on the validation score only; the run files also record test scores, which were not used.

| Dataset | lr | Epochs | Validation | Chosen |
|---|---|---|---|---|
| AG News | 1e-3 | 2 | 63.05, 62.03 (loss stuck near 1.0) | no |
| AG News | 3e-4 | 3 | 87.37 | no |
| AG News | 1e-4 | 3 | 89.45 | yes (also used for DBpedia) |
| GermEval 2014 | 3e-4 | 8 | 59.63 F1 | yes |
| GermEval 2014 | 1e-4 | 8 | 55.33 F1 | no |

The 1e-3 run was stopped after two epochs and its log was not kept; its numbers were read from the console output.
