# Few-Shot Prompting for Sentiment Classification on SST-2

## Problem
Does few-shot prompting improve sentiment classification accuracy on SST-2
relative to a TF-IDF + logistic regression baseline, and how does accuracy
change as the number of in-context examples increases from 0 to 3 to 10?

## Data
- **SST-2** (Stanford Sentiment Treebank v2), loaded via
  `load_dataset("nyu-mll/glue", "sst2")`.
- Public benchmark dataset of movie-review sentences.
- Binary labels: 0 = negative, 1 = positive.
- Splits used: `train` (for fitting baseline + few-shot pool),
  `validation` (for model/prompt selection),
  `test` (reserved — evaluated only once at the end).

## Method
- **Baseline:** TF-IDF vectorizer (unigrams + bigrams, max_features=10000)
  + Logistic Regression (max_iter=1000, random_state=42).
- **Few-shot condition:** [not yet implemented] — will use k ∈ {0, 3, 10}
  in-context examples sampled from the training split with a fixed seed.

## Results
| Condition | Val Accuracy | Val Macro-F1 |
|---|---|---|
| TF-IDF + LR baseline | <paste your number> | <paste your number> |
| 0-shot prompt | TBD | TBD |
| 3-shot prompt | TBD | TBD |
| 10-shot prompt | TBD | TBD |

Baseline is the number to beat.

## Limitations
- SST-2 is sentence-level movie reviews; results may not transfer to other domains.
- Few-shot results will depend on which examples are sampled; multiple seeds
  will be used to report mean ± SD.
- Additional limitations to be added as the project progresses.

## Privacy
All data used is the public SST-2 benchmark. No private or institutional
records are used or uploaded to any external service.