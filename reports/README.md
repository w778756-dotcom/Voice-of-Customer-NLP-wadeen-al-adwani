# Voice of Customer — Measured artifacts

- `training.json`: train/validation sizes, C selection, seed, data SHA and SVD variance.
- `predictions.csv`: every fixed test row and both model predictions, confidence/review flag.
- `metrics.json`: accuracy, macro-F1, bootstrap intervals, paired delta and review coverage.
- `slices.csv`: language, dialect and predeclared phenomenon slices.
- `error_analysis.csv`: all actual improved-model errors and categorized diagnoses.
- `learning_curve.csv`: validation performance on stratified train subsets.
- `benchmark.json`: single-comment warmed classifier latency and scope of costs.
- `clustering.json`: exploratory cosine silhouette and full demo batch latency.
- `confusion_matrices.png`: actual counts, row=true and column=prediction.
- `demo_smoke.json`: actual Gradio text/CSV/error-path request results.
- `notebook_validation.json`: executed-cell counts with no errors.
- `test_validation.json`: actual pytest counts, including CSV regression checks.
- `presentation.html`: standalone 8-slide presentation; keys/buttons navigate, browser print exports it.

Metrics measure authored synthetic examples, not real customers. Reproduce using the documented scripts. Timings change between runs. Human-readable root reports are generated from these artifacts with `scripts/build_docs.py`.
