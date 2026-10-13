# Data card — Voice of Customer

## Provenance and permission

`data/feedback.csv` contains **198 original, manually composed educational examples** created for this project with coding-assistant support. It is **synthetic**, not scraped, not customer records, and not a public-review benchmark. The project MIT license covers these original examples. No inference about actual customers may be made from this corpus.

| Split | Total | Arabic MSA | Arabic Gulf | English | Each sentiment |
|---|---:|---:|---:|---:|---:|
| train | 102 | 45 | 45 | 12 | 34 |
| validation | 30 | 15 | 15 | 0 | 10 |
| test | 66 | 30 | 30 | 6 | 22 |

The test set is deliberately stress-heavy: ordinary wording, negation, sarcasm, mixed sentiment, questions quoting opinions, dialect idioms and insufficient evidence. Test construction was fixed before model fitting. Validation has no English examples, so English results (only 6 test rows) are exploratory.

## Annotation guide

- `positive`: the author's overall satisfaction or approval; includes praise expressed through negation.
- `negative`: dissatisfaction, defects or unresolved service failures; sarcastic praise is negative when the adverse event is explicit.
- `neutral`: factual requests, information questions, quoted opinions without endorsement, or no evaluative evidence.
- Mixed comments use the final explicit satisfaction judgment when present; otherwise the decisive unresolved defect determines the overall label. This is a simplification; production should annotate aspects separately.

Columns: `id`, `text`, `label`, `split`, `lang`, `dialect`, `phenomenon`, `source`. Only `text` is a model input. Labels train the classifier; metadata is used solely for reporting. No ratings, gold labels, phenomena or language tags are input features. Topics in the demo are heuristic, not gold annotations.

## Leakage controls and limitations

Splits are explicit and immutable in the checked-in CSV. Training rejects exact duplicates after Unicode/Arabic normalization. Vectorizers, SVD and classifiers fit **train only**; hyperparameter selection uses validation only. No test-driven choice of C or model refit occurs. Seeds are 42. The data SHA-256 is recorded in `reports/training.json`.

Shared author, vocabulary, topical phrases and related sentence structures remain across splits. Exact deduplication does not establish absence of semantic overlap. This small, balanced, authored corpus is not representative of authentic prevalence, dialect diversity, spelling noise or long reviews. Labels have no independent annotator agreement measurement. Real-world claims require independently collected, licensed and reviewed data, grouped splits, and a fresh untouched test set.

## Privacy

Examples contain no real personal details. Local inference masks common email/phone patterns before feature extraction and output. This does **not** detect names, addresses, all Arabic-digit phones, or all identifiers. Do not upload personal customer records. No LLM API, external analytics, or public Gradio share tunnel is enabled. Delete uploaded files from the runtime when appropriate; masking is not a storage-retention guarantee.
