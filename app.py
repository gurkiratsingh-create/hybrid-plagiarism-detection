from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from src.aggregator import aggregate
from src.exact_match import exact_score
from src.paraphrase import fit_vectorizer, paraphrase_score
from src.pipeline import (
    _passes_candidate_gate,
    _prepare_sentences,
    extract_feature_bundle,
    sbert_only_score,
    tfidf_only_score,
)
from src.semantic import get_model


app = Flask(__name__)

THRESHOLDS = {
    "tfidf": 0.12,
    "sbert": 0.14,
    "hybrid": 0.09,
}


@dataclass
class SentenceMatch:
    source_sentence: str
    suspicious_sentence: str
    hybrid_score: float
    tfidf_score: float
    sbert_score: float
    exact_match: float
    passed_filter: bool


def _round(value: float) -> float:
    return round(float(value), 4)


def _label_from_score(score: float) -> str:
    if score >= 0.50:
        return "High plagiarism risk"
    if score >= THRESHOLDS["hybrid"]:
        return "Potential plagiarism risk"
    return "Low plagiarism risk"


def _confidence_note(score: float) -> str:
    if score >= 0.50:
        return "Strong evidence from the hybrid model. Review the matched sentences carefully."
    if score >= THRESHOLDS["hybrid"]:
        return "The score crosses the tuned hybrid threshold. This should be treated as suspicious, not as final proof."
    return "The score is below the tuned hybrid threshold. The pair looks less suspicious under the current model."


def _top_sentence_matches(source_text: str, suspicious_text: str, limit: int = 5) -> list[SentenceMatch]:
    source_sentences = _prepare_sentences(source_text)
    suspicious_sentences = _prepare_sentences(suspicious_text)

    if not source_sentences or not suspicious_sentences:
        return []

    model = get_model()
    source_embeddings = model.encode(source_sentences, normalize_embeddings=True)
    suspicious_embeddings = model.encode(suspicious_sentences, normalize_embeddings=True)
    similarity_matrix = cosine_similarity(source_embeddings, suspicious_embeddings)

    matches: list[SentenceMatch] = []

    for source_idx, source_sentence in enumerate(source_sentences):
        ranked_indices = similarity_matrix[source_idx].argsort()[::-1][: min(3, len(suspicious_sentences))]

        for suspicious_idx in ranked_indices:
            suspicious_sentence = suspicious_sentences[suspicious_idx]
            exact = exact_score(source_sentence, suspicious_sentence)
            lexical = paraphrase_score(source_sentence, suspicious_sentence)
            semantic = float(similarity_matrix[source_idx][suspicious_idx])
            passed_filter = _passes_candidate_gate(exact, lexical, semantic)
            hybrid = aggregate(exact, lexical, semantic)

            matches.append(
                SentenceMatch(
                    source_sentence=source_sentence,
                    suspicious_sentence=suspicious_sentence,
                    hybrid_score=_round(hybrid),
                    tfidf_score=_round(lexical),
                    sbert_score=_round(semantic),
                    exact_match=_round(exact),
                    passed_filter=bool(passed_filter),
                )
            )

    matches.sort(key=lambda item: item.hybrid_score, reverse=True)
    return matches[:limit]


def _build_report(source_text: str, suspicious_text: str) -> dict[str, Any]:
    # The research experiments fit TF-IDF on the dataset. For a live two-text
    # demo, fit it on both submitted documents plus their sentence fragments.
    fit_vectorizer([source_text, suspicious_text, *_prepare_sentences(source_text), *_prepare_sentences(suspicious_text)])

    features = extract_feature_bundle(source_text, suspicious_text)
    tfidf_score = tfidf_only_score(source_text, suspicious_text)
    sbert_score = sbert_only_score(source_text, suspicious_text)
    hybrid_score = features["final_score"]

    top_matches = _top_sentence_matches(source_text, suspicious_text)

    return {
        "verdict": _label_from_score(hybrid_score),
        "decision_note": _confidence_note(hybrid_score),
        "thresholds": THRESHOLDS,
        "scores": {
            "tfidf": _round(tfidf_score),
            "sbert": _round(sbert_score),
            "hybrid": _round(hybrid_score),
            "local_signal": _round(features["local_signal"]),
        },
        "predictions": {
            "tfidf": bool(tfidf_score >= THRESHOLDS["tfidf"]),
            "sbert": bool(sbert_score >= THRESHOLDS["sbert"]),
            "hybrid": bool(hybrid_score >= THRESHOLDS["hybrid"]),
        },
        "features": {
            "global_tfidf": _round(features["global_para"]),
            "global_sbert": _round(features["global_sem"]),
            "local_signal": _round(features["local_signal"]),
            "peak_local": _round(features["peak_local"]),
            "mean_top_local": _round(features["mean_top_local"]),
            "coverage": _round(features["coverage"]),
            "exact_ratio": _round(features["exact_ratio"]),
            "match_count": int(features["match_count"]),
            "source_sentence_count": int(features["total_source_sentences"]),
        },
        "top_matches": [asdict(match) for match in top_matches],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    source_text = str(payload.get("source_text", "")).strip()
    suspicious_text = str(payload.get("suspicious_text", "")).strip()

    if len(source_text) < 20 or len(suspicious_text) < 20:
        return (
            jsonify(
                {
                    "error": "Please enter at least 20 characters in both text boxes so the model has enough evidence."
                }
            ),
            400,
        )

    try:
        return jsonify(_build_report(source_text, suspicious_text))
    except Exception as exc:  # pragma: no cover - user-facing demo guard
        app.logger.exception("Analysis failed")
        return jsonify({"error": f"Model analysis failed: {exc}"}), 500


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
