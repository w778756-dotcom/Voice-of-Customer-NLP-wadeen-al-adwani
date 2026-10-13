"""One local entry point for classification, latent embeddings and insight summaries.

SVD embeddings are learned from training TF-IDF only; these are NOT pretrained
transformer embeddings. No external API is used. Serialized models are local,
trusted build outputs only, never arbitrary uploaded pickle files.
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

from bayan.text import mask_pii, normalize

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts/model.pkl"
LABELS = ["negative", "neutral", "positive"]
DISPLAY = {"positive": "إيجابي", "negative": "سلبي", "neutral": "محايد"}
ASPECTS = {
    "التوصيل": ["توصيل", "شحن", "مندوب", "وصل", "delivery", "shipping", "parcel"],
    "التطبيق والدفع": ["تطبيق", "دفع", "حساب", "تسجيل", "خصم", "app", "payment", "pay"],
    "الجودة": ["جوده", "جودة", "جهاز", "خامة", "خربان", "مكسور", "تغليف", "quality", "broken", "product"],
    "السعر": ["سعر", "غالي", "رسوم", "قيم", "price", "expensive", "cost"],
    "الدعم والاسترجاع": ["دعم", "خدم", "استرجاع", "استرد", "فلوس", "موظف", "support", "return", "service"],
}


def aspect(text):
    text = normalize(text)
    scores = {name: sum(term in text for term in terms) for name, terms in ASPECTS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "أخرى"


def make_baseline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 1))),
        ("classifier", MultinomialNB(alpha=1.0)),
    ])


def make_improved(c, config):
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=tuple(config["word_ngram_range"]))),
            ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=tuple(config["char_ngram_range"]))),
        ])),
        ("classifier", LogisticRegression(C=c, max_iter=1500, random_state=config["seed"])),
    ])


class Analyzer:
    def __init__(self, classifier, vectorizer, reducer, config):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.reducer = reducer
        self.config = config

    def embed(self, texts):
        return self.reducer.transform(self.vectorizer.transform(texts))

    def predict(self, texts):
        probabilities = self.classifier.predict_proba(texts)
        indices = probabilities.argmax(axis=1)
        return [
            {"text": mask_pii(text), "sentiment": str(self.classifier.classes_[index]),
             "sentiment_ar": DISPLAY[str(self.classifier.classes_[index])],
             "confidence": round(float(probabilities[i, index]), 4),
             "needs_review": bool(probabilities[i, index] < 0.55),
             "aspect": aspect(text)}
            for i, (text, index) in enumerate(zip(texts, indices))
        ]

    def analyze(self, texts):
        if not texts or len(texts) > 1000:
            raise ValueError("Provide 1–1000 nonempty comments.")
        if any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in texts):
            raise ValueError("Each comment must be nonempty text, at most 4000 characters.")
        predictions = self.predict(texts)
        vectors = self.embed(texts)
        unique_count = len({normalize(t) for t in texts})
        k = min(self.config["clusters"], len(texts), unique_count)
        assignments = KMeans(n_clusters=k, random_state=self.config["seed"], n_init=10).fit_predict(vectors)
        clusters = []
        for cluster in sorted(set(assignments)):
            positions = np.flatnonzero(assignments == cluster)
            members = [predictions[i] for i in positions]
            counts = Counter(m["sentiment"] for m in members)
            topic = Counter(m["aspect"] for m in members).most_common(1)[0][0]
            center = vectors[positions].mean(axis=0)
            representative = int(positions[np.argmin(np.linalg.norm(vectors[positions] - center, axis=1))])
            clusters.append({
                "cluster": int(cluster), "size": len(members), "topic": topic,
                "sentiment_counts": dict(counts),
                "representative": predictions[representative]["text"],
                "summary": f"مجموعة {topic}: {len(members)} تعليقًا، منها {counts['negative']} سلبي، "
                           f"و{counts['positive']} إيجابي، و{counts['neutral']} محايد. "
                           "الأعداد مبنية على تنبؤات النموذج وتحتاج مراجعة بشرية.",
            })
        return {"total": len(texts), "sentiment_counts": dict(Counter(p["sentiment"] for p in predictions)),
                "review_count": sum(p["needs_review"] for p in predictions),
                "predictions": predictions, "clusters": clusters,
                "summary_method": "Local extractive representatives and deterministic count-based summaries; no LLM.",
                "embedding_method": "Training-fitted TF-IDF → 32-dimensional SVD → L2 normalization"}


def fit_embeddings(texts, config):
    vectorizer = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(texts)
    reducer = Pipeline([
        ("svd", TruncatedSVD(n_components=min(config["svd_dimensions"], matrix.shape[0] - 1, matrix.shape[1] - 1), random_state=config["seed"])),
        ("normalize", Normalizer()),
    ])
    reducer.fit(matrix)
    return vectorizer, reducer


def load_analyzer():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model missing. Run: python scripts/train.py")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def assist(text: str) -> dict:
    """Single-comment entry point requested by the capstone repository template."""
    return load_analyzer().analyze([text])
