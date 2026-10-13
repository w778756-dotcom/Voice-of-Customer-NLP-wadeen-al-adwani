"""Train on train only; select C on validation; never read test labels for tuning."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from sklearn.metrics import f1_score

from bayan.pipeline import Analyzer, ROOT, fit_embeddings, make_baseline, make_improved
from bayan.text import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-path', type=Path, default=ROOT / 'reports/training.json',
                        help='Use artifacts/training.json for setup without altering checked-in reports.')
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/final.json").read_text())
    data = pd.read_csv(ROOT / "data/feedback.csv")
    normalized = data.text.map(normalize)
    if normalized.duplicated().any():
        raise ValueError("Normalized duplicate detected; review dataset before training.")
    train = data[data.split == "train"]
    validation = data[data.split == "validation"]
    baseline = make_baseline().fit(train.text, train.label)
    candidates = []
    models = []
    for c in config["candidate_c"]:
        model = make_improved(c, config).fit(train.text, train.label)
        score = f1_score(validation.label, model.predict(validation.text), average="macro")
        candidates.append({"C": c, "validation_macro_f1": float(score)})
        models.append(model)
    best = max(range(len(candidates)), key=lambda i: candidates[i]["validation_macro_f1"])
    vectorizer, reducer = fit_embeddings(train.text, config)
    analyzer = Analyzer(models[best], vectorizer, reducer, config)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    for name, obj in [("model.pkl", analyzer), ("baseline.pkl", baseline)]:
        with (ROOT / "artifacts" / name).open("wb") as f:
            pickle.dump(obj, f)
    metadata = {"seed": config["seed"], "train_n": len(train), "validation_n": len(validation),
                "selection": "Highest validation macro-F1; first candidate wins ties. No train+validation refit.",
                "candidates": candidates, "selected_C": candidates[best]["C"],
                "data_sha256": hashlib.sha256((ROOT / "data/feedback.csv").read_bytes()).hexdigest(),
                "svd_explained_variance": float(reducer.named_steps['svd'].explained_variance_ratio_.sum())}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
