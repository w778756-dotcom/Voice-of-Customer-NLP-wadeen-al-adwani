# Model card — local educational NLP prototype

**Owner:** Wadeen Khalid Al adwani · **Institution:** SDAIA Academy · **Instructor:** مضر محمد الفضلي

## Intended use

Project 3 capstone: demonstrate sentiment classification, embeddings-based issue clustering, evidence-grounded insight summaries, and reproducible evaluation of Arabic feedback. For learning and assisted review, **not autonomous customer decisions**.

## Models and components

1. Baseline: word unigram TF-IDF + Multinomial Naive Bayes (`alpha=1`).
2. Improvement: word unigrams/bigrams + character boundary 3–5 grams + Logistic Regression. Validation chooses C from 0.5, 2, 8; deterministic first-candidate tie-break.
3. Representation: train-fitted character TF-IDF → TruncatedSVD (32 dimensions) → L2-normalized **latent semantic embeddings**. These are not transformer/sentence-transformer embeddings and have limited semantic generalization.
4. Clustering: deterministic KMeans over embeddings, up to 5 groups per input batch; cluster IDs have no persistent business meaning. Group count shrinks for small or duplicate-only batches.
5. Topic names: explicit keyword rules. These can mislabel ambiguous topics; no trained aspect classifier or gold topic accuracy is claimed.
6. Insights: prediction counts, majority heuristic topic and nearest-centroid representative quote; deterministic templates, not an LLM. Quotes come from masked input and are traceable to the output predictions.

All supervised and embedding fitting uses the 102-row training split only. Validation and test are never fitted into the vocabulary or SVD. No pretrained base-model revision exists because no pretrained model is used. `uv.lock` pins installed packages and artifact hashes; `configs/final.json` records hyperparameters.

## Adaptation decision

The capstone permits fine-tuning **or written evidence-based justification for not fine-tuning**. This submission takes the justification route: see `DECISIONS.md`, the measured learning curve and test limitations. Training a logistic classifier is not presented as transformer fine-tuning. No encoder fine-tuning or LoRA run was performed, and no transformer comparison is claimed.

## Reliability and limitations

Measured performance and slice counts appear in `EVALUATION_REPORT.md`; do not generalize synthetic-set results. Negation, neutral questions containing sentiment words, idioms and mixed sentiment remain failure modes. English training is minimal, English validation absent, and English test has 6 rows. Probabilities are uncalibrated. Confidence <0.55 routes to review as a demonstration heuristic; at the recorded evaluation **all 66 test comments require review**. The system is consequently unsuitable for autonomous decisions.

Unknown words may produce zero vectors and uninformative predictions. Internal silhouette is a geometric diagnostic only, not human topic correctness. CPU-only execution makes no external paid API calls; hosting and compute costs still exist. Read only model pickle files generated locally by `scripts/train.py`; do not load models supplied by strangers.
