# Benchmarks

Actual local measurements from `reports/benchmark.json`; Python 3.12.14, Linux-6.18.44-x86_64-with-glibc2.41, 5 logical CPUs. CPU execution, single-comment batch size, 5 warmup predictions then 66 timed predictions using `time.perf_counter`. No GPU claim. Latency varies with system load; these are not service-level guarantees.

| System | Test N | Accuracy | Macro-F1 | 95% bootstrap CI | p95 single-comment ms |
| --- | --- | --- | --- | --- | --- |
| baseline | 66 | 0.530 | 0.533 | [0.402, 0.656] | 0.69 |
| improved | 66 | 0.576 | 0.575 | [0.440, 0.705] | 1.45 |

Classification p95 excludes server transport, CSV parsing, embeddings and clustering. The separate 15-comment full pipeline took 21.66 ms; that single batch duration is **not p95**. Cosine silhouette was 0.232; no gold topic accuracy measured.

Paid API cost per 1,000 comments: **USD 0** (no API calls). Compute, hosting, electricity and labor are **not included**. Training durations and validation curve are in `reports/learning_curve.csv`.

Reproduce via `bash scripts/run_all.sh`. Environment setup download time is not inference latency. Do not compare these CPU measurements to unmeasured GPU/transformer models.
