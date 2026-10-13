"""Fixed-test comparison, paired bootstrap, slice metrics, actual error inventory."""

import json
import os
import pickle
import platform
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/voice_of_customer-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/workspace/.cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split

from voice_of_customer.pipeline import LABELS, ROOT, load_analyzer, make_improved


def macro(y, p):
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def interval(y, p, n=500):
    rng = np.random.default_rng(42)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        values.append(macro(y[idx], p[idx]))
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def main():
    frame = pd.read_csv(ROOT / "data/feedback.csv")
    test = frame[frame.split == "test"].copy()
    train = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    analyzer = load_analyzer()
    with (ROOT / "artifacts/baseline.pkl").open("rb") as f:
        baseline = pickle.load(f)
    y = test.label.to_numpy()
    test["baseline_pred"] = baseline.predict(test.text)
    test["improved_pred"] = analyzer.classifier.predict(test.text)
    probabilities = analyzer.classifier.predict_proba(test.text)
    test["confidence"] = probabilities.max(axis=1)
    test["needs_review"] = test.confidence < 0.55
    test.to_csv(ROOT / "reports/predictions.csv", index=False)
    metrics = {}
    slices = []
    for name in ["baseline", "improved"]:
        p = test[f"{name}_pred"].to_numpy()
        metrics[name] = {"n": len(y), "accuracy": float(accuracy_score(y, p)),
                         "macro_f1": macro(y, p), "macro_f1_95ci": interval(y, p),
                         "confusion_matrix_labels": LABELS, "confusion_matrix": confusion_matrix(y, p, labels=LABELS).tolist()}
        for column in ["lang", "dialect", "phenomenon"]:
            for value, group in test.groupby(column):
                slices.append({"system": name, "slice": f"{column}={value}", "n": len(group),
                               "accuracy": float(accuracy_score(group.label, group[f"{name}_pred"])),
                               "macro_f1": macro(group.label.to_numpy(), group[f"{name}_pred"].to_numpy())})
    # Paired sampling: both systems evaluated on precisely the same sampled rows.
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(500):
        idx = rng.integers(0, len(test), len(test))
        deltas.append(macro(y[idx], test.improved_pred.to_numpy()[idx]) - macro(y[idx], test.baseline_pred.to_numpy()[idx]))
    metrics["paired_delta"] = {"macro_f1": metrics['improved']['macro_f1'] - metrics['baseline']['macro_f1'],
                               "95ci": [float(x) for x in np.quantile(deltas, [.025, .975])],
                               "note": "Exploratory bootstrap on a small synthetic test set, not population-level evidence."}
    accepted = test[~test.needs_review]
    metrics["review_policy"] = {"threshold": .55, "coverage": len(accepted) / len(test),
                                "accepted_n": len(accepted), "review_n": int(test.needs_review.sum()),
                                "accepted_macro_f1": macro(accepted.label.to_numpy(), accepted.improved_pred.to_numpy()) if len(accepted) else None,
                                "note": "Confidence is uncalibrated; 0.55 is a demo routing heuristic, not a tuned reliability guarantee."}
    (ROOT / "reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(slices).to_csv(ROOT / "reports/slices.csv", index=False)
    categories = {
        "sarcasm": ("sarcasm", "Surface praise reverses the intended sentiment; collect annotated sarcastic reviews."),
        "negation": ("negation", "Negation scope is not represented reliably; add independently reviewed negation examples."),
        "mixed": ("mixed_sentiment", "Overall label loses aspect-level disagreement; annotate sentiment per aspect."),
        "question": ("quoted_or_questioned_sentiment", "A quoted opinion or question is mistaken for the author's own sentiment."),
        "dialect": ("dialect_or_idiom", "Gulf idiom coverage is sparse; broaden dialect data."),
        "insufficient_evidence": ("insufficient_evidence", "No explicit evaluative evidence; human review is appropriate."),
        "ordinary": ("lexical_generalization", "Small training vocabulary or cross-domain wording; collect real permitted reviews."),
    }
    errors = test[test.label != test.improved_pred].copy()
    errors["error_category"] = errors.phenomenon.map(lambda x: categories[x][0])
    errors["analysis_and_next_step"] = errors.phenomenon.map(lambda x: categories[x][1])
    errors.to_csv(ROOT / "reports/error_analysis.csv", index=False)
    # Learning curve: training subsets, validation scores only; test untouched.
    chosen = json.loads((ROOT / "reports/training.json").read_text())["selected_C"]
    curves = []
    for fraction in [.25, .5, .75, 1.0]:
        subset = train if fraction == 1.0 else train_test_split(train, train_size=fraction, random_state=42, stratify=train.label)[0]
        model = make_improved(chosen, analyzer.config)
        start = time.perf_counter()
        model.fit(subset.text, subset.label)
        curves.append({"train_n": len(subset), "fit_seconds": time.perf_counter() - start,
                       "validation_macro_f1": macro(validation.label.to_numpy(), model.predict(validation.text))})
    pd.DataFrame(curves).to_csv(ROOT / "reports/learning_curve.csv", index=False)
    benchmark = {"python": platform.python_version(), "platform": platform.platform(),
                 "logical_cpus": os.cpu_count(), "batch_size": 1, "warmup": 5, "repeats": 66,
                 "api_cost_per_1000_usd": 0.0, "cost_scope": "No API calls; excludes CPU, electricity and hosting."}
    for name, model in [("baseline", baseline), ("improved", analyzer.classifier)]:
        for text in test.text.iloc[:5]:
            model.predict([text])
        elapsed = []
        for text in test.text:
            start = time.perf_counter()
            model.predict([text])
            elapsed.append((time.perf_counter() - start) * 1000)
        benchmark[name] = {"p50_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95))}
    (ROOT / "reports/benchmark.json").write_text(json.dumps(benchmark, indent=2))
    demo_texts = test.text.iloc[:5].tolist() + test.text.iloc[20:25].tolist() + test.text.iloc[40:45].tolist()
    start = time.perf_counter()
    result = analyzer.analyze(demo_texts)
    end_to_end_ms = (time.perf_counter() - start) * 1000
    (ROOT / "sample_outputs/analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({"text": demo_texts}).to_csv(ROOT / "sample_outputs/demo_input.csv", index=False)
    # Recompute deterministic cluster assignments for an unsupervised diagnostic.
    from sklearn.cluster import KMeans
    embeddings = analyzer.embed(demo_texts)
    cluster_ids = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(embeddings)
    cluster_metrics = {"n": len(demo_texts), "clusters": 5, "silhouette_cosine": float(silhouette_score(embeddings, cluster_ids, metric="cosine")),
                       "end_to_end_batch_ms": end_to_end_ms,
                       "note": "Internal compactness only; no gold topic labels, topic accuracy is NOT established."}
    (ROOT / "reports/clustering.json").write_text(json.dumps(cluster_metrics, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ['baseline', 'improved']):
        matrix = np.array(metrics[name]['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
        ax.set_xticks(range(3), LABELS, rotation=25)
        ax.set_yticks(range(3), LABELS)
        ax.set(title=name, xlabel='Predicted', ylabel='True')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/confusion_matrices.png', dpi=160)
    plt.close(fig)
    print(json.dumps({"metrics": metrics, "actual_improved_errors": len(errors), "error_categories": dict(Counter(errors.error_category))}, indent=2))


if __name__ == '__main__':
    main()
