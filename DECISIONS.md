# Technical decisions and fine-tuning justification

## Arabic preprocessing

Keep negation and function words: deleting «لا»، «لم»، «مو» would remove sentiment cues. Apply NFKC, remove elongation/diacritics, normalize alef and alef-maqsura, and mask common emails/phones. Preserve ta-marbuta rather than aggressive stemming. Character n-grams improve spelling/dialect coverage without downloading a large encoder. No external LLM receives customer text.

## Fair comparison

Both systems see exactly the same training rows and held-out test rows. Baseline settings are fixed in advance. Select improved C using validation macro-F1 only; no test-selected model, feature change or threshold. Stable class ordering is negative/neutral/positive. Bootstrap intervals and paired deltas use 500 resamples with seed 42. Some phenomenon slices lack one or more classes; their macro-F1 still averages over the same three labels and must be read with accuracy and counts.

## Why no encoder/LoRA fine-tuning in this submission

The formal requirement offers a written justification alternative. Evidence is in `reports/training.json`, `reports/learning_curve.csv`, `reports/benchmark.json` and `reports/metrics.json`:

- Only 102 authored training items (34/class), 30 validation items, and no independent annotation agreement. A fine-tuned encoder could overfit this author's style. A larger model does not correct unreliable labels or nonrepresentative data.
- The measured validation learning curve is nonmonotonic; adding synthetic rows is not demonstrated to steadily improve generalization. See the actual scores in the evaluation report rather than assuming more model capacity is necessary.
- The observed test macro-F1 gain is small and its paired bootstrap interval includes zero. We cannot claim a statistically reliable improvement or justify investing in a larger model from this experiment alone.
- The actual local pipeline and timing measurements demonstrate that a runnable, inexpensive teaching prototype exists on CPU. We did not benchmark a transformer or prove that fine-tuning would be slower, worse or unaffordable.
- High review rates and neutral-question/negation errors argue for independently annotated real reviews, aspect labels and probability calibration first.

This is a defensible **scope decision**, not evidence that fine-tuning is ineffective. A next experiment should obtain at least a materially larger, licensed Arabic review corpus, grouped train/validation/test splits, two-annotator review, and compare a pinned multilingual sentence encoder + linear head against fine-tuning. Choose all settings on validation, evaluate once on a new held-out test, report latency, slice F1, calibration and ≥10 errors again. Do not reuse this project's exposed test set for a future confirmatory claim.

## Summaries and clustering

Use SVD embeddings rather than claim unavailable pretrained semantic knowledge. Cluster input batches without sentiment labels. Generate count summaries and representative quotes so every insight has visible local evidence. Majority-topic rules are descriptive heuristics. Five clusters are a fixed demo default, not an empirically optimal number; silhouette is reported as exploratory. No hallucinated causal explanations or fabricated business statistics are produced.
