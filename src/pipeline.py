from .aggregator import aggregate, aggregate_document
from .exact_match import exact_score
from .paraphrase import paraphrase_score
from .preprocessing import preprocess
from .semantic import get_model, semantic_score
from sklearn.metrics.pairwise import cosine_similarity


MAX_SENTENCES = 15
TOP_K = 3
SEMANTIC_GATE = 0.50
LEXICAL_GATE = 0.08
STRONG_SEMANTIC_MATCH = 0.86
STRONG_LEXICAL_MATCH = 0.30


def _prepare_sentences(text):
    _, sentences = preprocess(text)

    if not sentences:
        sentences = [text]

    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return cleaned[:MAX_SENTENCES]


def _passes_candidate_gate(exact, para, sem):
    if exact == 1.0:
        return True

    if sem >= STRONG_SEMANTIC_MATCH:
        return True

    if para >= STRONG_LEXICAL_MATCH and sem >= 0.35:
        return True

    return sem >= SEMANTIC_GATE and para >= LEXICAL_GATE


def extract_feature_bundle(text1, text2):
    sents1 = _prepare_sentences(text1)
    sents2 = _prepare_sentences(text2)

    if not sents1 or not sents2:
        return {
            "global_para": 0.0,
            "global_sem": 0.0,
            "local_signal": 0.0,
            "final_score": 0.0,
            "peak_local": 0.0,
            "mean_top_local": 0.0,
            "coverage": 0.0,
            "exact_ratio": 0.0,
            "match_count": 0,
            "total_source_sentences": 0,
        }

    model = get_model()

    global_para = paraphrase_score(text1, text2)
    global_sem = semantic_score(text1, text2)

    emb1 = model.encode(sents1, normalize_embeddings=True)
    emb2 = model.encode(sents2, normalize_embeddings=True)
    sim_matrix = cosine_similarity(emb1, emb2)

    candidate_matches = []

    for i, sentence_1 in enumerate(sents1):
        top_indices = sim_matrix[i].argsort()[-min(TOP_K, len(sents2)):]
        best_match = None

        for j in reversed(top_indices):
            sentence_2 = sents2[j]

            if not sentence_1 or not sentence_2:
                continue

            exact = exact_score(sentence_1, sentence_2)
            para = paraphrase_score(sentence_1, sentence_2)
            sem = float(sim_matrix[i][j])

            if not _passes_candidate_gate(exact, para, sem):
                continue

            pair_score = aggregate(exact, para, sem)
            match = {
                "source_idx": i,
                "target_idx": j,
                "score": pair_score,
                "exact": exact,
            }

            if best_match is None or pair_score > best_match["score"]:
                best_match = match

        if best_match is not None:
            candidate_matches.append(best_match)

    candidate_matches.sort(key=lambda match: match["score"], reverse=True)

    used_targets = set()
    selected_matches = []

    for match in candidate_matches:
        if match["target_idx"] in used_targets:
            continue

        selected_matches.append(match)
        used_targets.add(match["target_idx"])

    selected_scores = [match["score"] for match in selected_matches]
    matched_source_count = len({match["source_idx"] for match in selected_matches})
    coverage = matched_source_count / len(sents1) if sents1 else 0.0
    exact_ratio = (
        sum(match["exact"] for match in selected_matches) / len(selected_matches)
        if selected_matches
        else 0.0
    )

    ordered_scores = sorted(selected_scores, reverse=True)
    top_window = ordered_scores[: min(3, len(ordered_scores))]
    peak_local = ordered_scores[0] if ordered_scores else 0.0
    mean_top_local = sum(top_window) / len(top_window) if top_window else 0.0

    final_score = aggregate_document(
        match_scores=selected_scores,
        matched_source_count=matched_source_count,
        total_source_count=len(sents1),
        global_para=global_para,
        global_sem=global_sem,
        exact_ratio=exact_ratio,
    )

    local_signal = (
        (final_score - 0.25 * global_para - 0.35 * global_sem) / 0.40
        if (selected_scores or global_para or global_sem)
        else 0.0
    )
    local_signal = min(max(local_signal, 0.0), 1.0)

    return {
        "global_para": float(global_para),
        "global_sem": float(global_sem),
        "local_signal": float(local_signal),
        "final_score": float(final_score),
        "peak_local": float(peak_local),
        "mean_top_local": float(mean_top_local),
        "coverage": float(coverage),
        "exact_ratio": float(exact_ratio),
        "match_count": len(selected_matches),
        "total_source_sentences": len(sents1),
    }


def compute_score(text1, text2):
    return extract_feature_bundle(text1, text2)["final_score"]


def tfidf_only_score(s1, s2):
    return paraphrase_score(s1, s2)


def sbert_only_score(s1, s2):
    return semantic_score(s1, s2)


def local_signal_score(s1, s2):
    return extract_feature_bundle(s1, s2)["local_signal"]


def global_only_hybrid_score(s1, s2):
    features = extract_feature_bundle(s1, s2)
    return 0.25 * features["global_para"] + 0.35 * features["global_sem"]
