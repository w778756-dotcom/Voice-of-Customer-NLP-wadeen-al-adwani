# Evaluation report — تقرير التقييم

**Wadeen Khalid Al adwani** · SDAIA Academy · instructor: مضر محمد الفضلي · project 3

## Protocol and measured outcomes

198 authored synthetic comments: 102 train, 30 validation, 66 test. Labels are balanced within each split. Metrics below use the **same 66 fixed test rows** for both models, including difficult phenomena. Train-only fitting; C selected on validation from 0.5, 2, 8; selected C=0.5. No test tuning, no test refit. The best validation macro-F1 was 0.602.

| System | Test N | Accuracy | Macro-F1 | 95% bootstrap CI | p95 single-comment ms |
| --- | --- | --- | --- | --- | --- |
| baseline | 66 | 0.530 | 0.533 | [0.402, 0.656] | 0.69 |
| improved | 66 | 0.576 | 0.575 | [0.440, 0.705] | 1.45 |

Paired macro-F1 delta: **0.042**, 95% paired-bootstrap interval **[-0.032, 0.128]**. The interval includes zero: improvement is observed on this sample, **not demonstrated to be statistically reliable**. 500 row-resamples, seed 42, fixed three-class macro averaging. These exploratory intervals describe the authored corpus and do not establish population generalization.

The fixed confidence threshold 0.55 is an uncalibrated demonstration policy: accepted 0/66, review 66/66. Full-coverage model metrics are above; no selective score is invented for an empty accepted set. Human review is required for every test item in this run.

## Quantitative slices

| System | Slice | N | Accuracy | Macro-F1 |
| --- | --- | --- | --- | --- |
| baseline | lang=ar | 60 | 0.550 | 0.551 |
| baseline | lang=en | 6 | 0.333 | 0.267 |
| baseline | dialect=english | 6 | 0.333 | 0.267 |
| baseline | dialect=gulf | 30 | 0.700 | 0.701 |
| baseline | dialect=msa | 30 | 0.400 | 0.381 |
| baseline | phenomenon=dialect | 4 | 0.500 | 0.267 |
| baseline | phenomenon=insufficient_evidence | 4 | 0.250 | 0.133 |
| baseline | phenomenon=mixed | 5 | 0.400 | 0.300 |
| baseline | phenomenon=negation | 7 | 0.143 | 0.083 |
| baseline | phenomenon=ordinary | 33 | 0.697 | 0.699 |
| baseline | phenomenon=question | 7 | 0.429 | 0.200 |
| baseline | phenomenon=sarcasm | 6 | 0.500 | 0.222 |
| improved | lang=ar | 60 | 0.583 | 0.585 |
| improved | lang=en | 6 | 0.500 | 0.389 |
| improved | dialect=english | 6 | 0.500 | 0.389 |
| improved | dialect=gulf | 30 | 0.700 | 0.701 |
| improved | dialect=msa | 30 | 0.467 | 0.462 |
| improved | phenomenon=dialect | 4 | 0.500 | 0.267 |
| improved | phenomenon=insufficient_evidence | 4 | 0.250 | 0.133 |
| improved | phenomenon=mixed | 5 | 0.600 | 0.389 |
| improved | phenomenon=negation | 7 | 0.286 | 0.148 |
| improved | phenomenon=ordinary | 33 | 0.727 | 0.724 |
| improved | phenomenon=question | 7 | 0.143 | 0.083 |
| improved | phenomenon=sarcasm | 6 | 0.833 | 0.303 |

Some phenomenon slices contain fewer than all three labels: macro-F1 retains negative/neutral/positive with zero_division=0, so read it alongside accuracy and N. Gulf/standard labels refer to authored style, not verified linguistic annotation. English N=6, no English validation; language scores are unstable. No English superiority or fairness guarantee is claimed.

## Actual error analysis — 28 categorized failures

Every row below is a real incorrect improved-model prediction from `reports/predictions.csv`, not an invented example or a changed gold label. Categories use the dataset's predeclared phenomenon tag; these are hypothesis-driven analyst categories, not independently validated causes. All 28 errors are included, exceeding the ≥10 requirement.

| Category | Actual failures |
| --- | --- |
| dialect_or_idiom | 2 |
| insufficient_evidence | 3 |
| lexical_generalization | 9 |
| mixed_sentiment | 2 |
| negation | 5 |
| quoted_or_questioned_sentiment | 6 |
| sarcasm | 1 |

| ID | Comment | Gold | Predicted | Cause | Analysis / next step |
| --- | --- | --- | --- | --- | --- |
| VOC-0121 | حصلت على مساعدة ممتازة لحل مشكلة الحساب | positive | negative | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0122 | أداء الجهاز رائع بعد شهر من الاستخدام | positive | neutral | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0123 | وصلت الشحنة مبكرا وبحالة جيدة | positive | negative | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0125 | التطبيق جميل ويسهل متابعة المشتريات | positive | negative | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0126 | لم يتأخر الطلب هذه المرة | positive | negative | negation | Negation scope is not represented reliably; add independently reviewed negation examples. |
| VOC-0127 | ليست الجودة سيئة كما كنت أخشى | positive | negative | negation | Negation scope is not represented reliably; add independently reviewed negation examples. |
| VOC-0129 | السعر مرتفع لكن المنتج يستحق كل ريال | positive | negative | mixed_sentiment | Overall label loses aspect-level disagreement; annotate sentiment per aspect. |
| VOC-0134 | التطبيق سهل وواضح ويختصر الوقت | positive | negative | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0138 | مو غالي إذا شفت جودته | positive | negative | negation | Negation scope is not represented reliably; add independently reviewed negation examples. |
| VOC-0144 | الدعم تجاهل المشكلة ولم يساعدني | negative | positive | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0148 | ليست تجربة تستحق التكرار | negative | positive | negation | Negation scope is not represented reliably; add independently reviewed negation examples. |
| VOC-0150 | الجهاز ممتاز شكلا لكنه لا يعمل إطلاقا | negative | positive | mixed_sentiment | Overall label loses aspect-level disagreement; annotate sentiment per aspect. |
| VOC-0157 | ما شاء الله سرعة أسبوعين عشان شحنة | negative | neutral | sarcasm | Surface praise reverses the intended sentiment; collect annotated sarcastic reviews. |
| VOC-0158 | البرنامج يفصل معي وش هالبهذلة | negative | neutral | dialect_or_idiom | Gulf idiom coverage is sparse; broaden dialect data. |
| VOC-0159 | خدمة ترفع الضغط وطفشت | negative | positive | dialect_or_idiom | Gulf idiom coverage is sparse; broaden dialect data. |
| VOC-0164 | ما المستندات المطلوبة لاسترجاع المنتج | neutral | positive | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0166 | هل الخدمة ممتازة فعلا أم مجرد إعلان | neutral | negative | quoted_or_questioned_sentiment | A quoted opinion or question is mistaken for the author's own sentiment. |
| VOC-0167 | أرى تعليقات تقول إن التطبيق سيئ هل أصلحتموه | neutral | negative | quoted_or_questioned_sentiment | A quoted opinion or question is mistaken for the author's own sentiment. |
| VOC-0168 | هل السعر غالي مقارنة بالنسخة السابقة | neutral | negative | quoted_or_questioned_sentiment | A quoted opinion or question is mistaken for the author's own sentiment. |
| VOC-0169 | لم أجرب المنتج بعد لذلك لا أستطيع تقييمه | neutral | negative | insufficient_evidence | No explicit evaluative evidence; human review is appropriate. |
| VOC-0170 | وصل الطلب اليوم ولم أفتح الصندوق بعد | neutral | positive | insufficient_evidence | No explicit evaluative evidence; human review is appropriate. |
| VOC-0175 | وش المطلوب عشان أفعل الحساب | neutral | negative | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |
| VOC-0177 | يقولون خدمتكم سيئة هل تغير الوضع | neutral | negative | quoted_or_questioned_sentiment | A quoted opinion or question is mistaken for the author's own sentiment. |
| VOC-0178 | التطبيق يجنن مثل ما يقولون ولا أجرب غيره | neutral | positive | quoted_or_questioned_sentiment | A quoted opinion or question is mistaken for the author's own sentiment. |
| VOC-0179 | جاني المنتج ولسه ما استخدمته | neutral | negative | insufficient_evidence | No explicit evaluative evidence; human review is appropriate. |
| VOC-0194 | Not a bad product at all | positive | negative | negation | Negation scope is not represented reliably; add independently reviewed negation examples. |
| VOC-0197 | Is the quality really great as advertised | neutral | positive | quoted_or_questioned_sentiment | A quoted opinion or question is mistaken for the author's own sentiment. |
| VOC-0198 | Please explain the warranty coverage | neutral | negative | lexical_generalization | Small training vocabulary or cross-domain wording; collect real permitted reviews. |

## Embeddings and summaries

Training TF-IDF → 32D SVD → L2 embeddings, explained variance 0.473. Demo input: 15 comments, 5 KMeans clusters; cosine silhouette **0.232**. This is internal compactness only; no gold topic labels means topic correctness is **unmeasured**. Full batch runtime including prediction/embeddings/clustering/summary: 21.66 ms in the measured run.

The deterministic summary recounts model predictions and quotes a masked representative comment. Counts are verified by the functional tests; summary interpretation still needs human review. No LLM-based generation quality or factuality metric is claimed.

## Learning-curve evidence for the adaptation decision

| Training N | Validation macro-F1 | Fit seconds |
| --- | --- | --- |
| 25 | 0.478 | 0.0099 |
| 51 | 0.487 | 0.0133 |
| 76 | 0.630 | 0.0184 |
| 102 | 0.602 | 0.0240 |

Fixed selected C; stratified train subsets, validation-only scores. Scores are nonmonotonic, with tiny single samples per size, so they do not prove a scaling trend. See `DECISIONS.md` for the evidence-based no-fine-tuning justification explicitly allowed by the capstone. No encoder or LoRA fine-tuning was run.

## Reproducibility and evidence

- Source corpus SHA-256: `1590fcc1c0659e9f29e0e067d5d11cace10932e4bb9facff3d4a39e20588271b`.
- Numeric raw evidence: `reports/training.json`, `reports/metrics.json`, `reports/slices.csv`, `reports/predictions.csv`, `reports/error_analysis.csv`, `reports/learning_curve.csv`.
- Confusion matrix: `reports/confusion_matrices.png` (true labels on rows).
- Rebuild with `bash scripts/run_all.sh`; regenerate this report using `uv run --frozen python scripts/build_docs.py`.
- This is an educational prototype; real customer data, independent annotations and a new untouched test are required before deployment.
