## Error analysis

Of 425 valid LLM predictions across the three conditions, 16 were wrong
(3.8%). By category:

| Category | Count | Share | Example |
|---|---|---|---|
| Mixed sentiment / contrast | 9 | 56% | "does paint some memorable images ... , but makhmalbaf keeps her distance from the characters" (true: positive, pred: negative) |
| Sarcasm / surface-negative praise | 3 | 19% | "hilariously inept and ridiculous ." (true: positive, pred: negative) |
| Negation | 2 | 12% | "you wo n't like roger , but you will quickly recognize him ." (true: negative, pred: positive) |
| Subtle intensity | 2 | 12% | "fresnadillo 's dark and jolting images ... like the nightmare you had a week ago" (true: positive, pred: negative) |

**Key finding.** Four sentences failed at **every shot count** (k=0, 3, and
10): the "we root for clara" mixed-sentiment sentence, the "riveting world
war ii" sentence, the "does paint some memorable images ... but" contrast
sentence, and the "hilariously inept" sarcasm sentence. Few-shot examples
did not correct any of these systematic failures — the model fell into the
same traps regardless of how many examples were in the prompt.

This explains why the accuracy differences across k were within the noise
band: the errors are not random. They are concentrated on a fixed subset
of sentences whose sentiment depends on contrast, irony, or implicit
praise/criticism — properties that cannot be taught by a handful of
sentiment-labelled examples. A benchmark that targets these phenomena
directly (e.g., contrast/negation subsets, or SST-5 fine-grained labels)
would be needed to detect a shot-count effect.