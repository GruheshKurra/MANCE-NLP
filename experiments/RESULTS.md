# Results

Accuracy (%) for AG News and DBpedia, entity-level F1 for GermEval 2014. Measured values are mean ± sd over seeds. Built by `python run.py table` from `experiments/runs/`.

| Dataset | Model | Test score | Seeds | Drop with unseen variants: rule set | attested set | Params |
|---|---|---|---|---|---|---|
| ag_news | avg_chars | 85.38 | 1 | 0.82 (10.0% of words) | 0.11 (1.4% of words) | 4.3M |
| ag_news | mance | 90.99 ± 0.34 | 3 | 0.51 ± 0.14 (10.0% of words) | 0.00 ± 0.03 (1.4% of words) | 12.7M |
| ag_news | no_char_order | 90.08 | 1 | 0.86 (10.0% of words) | 0.07 (1.4% of words) | 12.7M |
| ag_news | word_baseline | 91.89 | 1 | 1.17 (10.0% of words) | -0.01 (1.4% of words) | 29.8M |
| dbpedia | mance | 98.48 ± 0.11 | 3 | – | – | 12.7M |
| dbpedia | word_baseline | 98.69 | 1 | – | – | 29.8M |
| germeval14 | mance | 63.02 ± 1.15 | 3 | – | – | 12.7M |
| germeval14 | word_baseline | 48.16 | 1 | – | – | 15.7M |
