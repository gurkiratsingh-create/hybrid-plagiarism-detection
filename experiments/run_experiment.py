import json
import os
import random
import time

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluator import (
    evaluate,
    find_best_threshold,
    metrics_from_scored_cases,
    score_dataset,
)
from src.pan_loader import load_pan_dataset
from src.paraphrase import fit_vectorizer, paraphrase_score
from src.pipeline import extract_feature_bundle
from src.semantic import semantic_score


DATA_PATH = "data/pan13"
RESULTS_PATH = "results/comparison.json"
SEED = 42
LIMIT = 30
TRAIN_RATIO = 0.7
THRESHOLDS = [round(step / 100, 2) for step in range(0, 101)]
BOOTSTRAP_SAMPLES = 500


def tfidf_only(s1, s2):
    return paraphrase_score(s1, s2)


def sbert_only(s1, s2):
    return semantic_score(s1, s2)


def hybrid(s1, s2):
    return extract_feature_bundle(s1, s2)["final_score"]


def split_dataset(dataset, train_ratio=TRAIN_RATIO):
    split_index = max(1, int(len(dataset) * train_ratio))
    split_index = min(split_index, len(dataset) - 1)
    return dataset[:split_index], dataset[split_index:]


def prepare_vectorizer(dataset):
    all_texts = []

    for case in dataset:
        all_texts.extend([case["s1"], case["s2"]])

    fit_vectorizer(all_texts)


def bootstrap_confidence_interval(scored_cases, threshold, metric_key, seed=SEED, n_samples=BOOTSTRAP_SAMPLES):
    rng = random.Random(seed)
    values = []

    if not scored_cases:
        return {"low": 0.0, "high": 0.0}

    for _ in range(n_samples):
        sample = [rng.choice(scored_cases) for _ in range(len(scored_cases))]
        metrics = metrics_from_scored_cases(sample, threshold)
        values.append(metrics[metric_key])

    values.sort()
    low_index = int(0.025 * len(values))
    high_index = int(0.975 * len(values)) - 1

    return {
        "low": values[max(0, low_index)],
        "high": values[max(0, high_index)],
    }


def attach_confidence_intervals(scored_cases, metrics, seed_offset=0):
    metrics = dict(metrics)
    metrics["confidence_interval"] = {
        "precision": bootstrap_confidence_interval(
            scored_cases, metrics["threshold"], "precision", seed=SEED + seed_offset
        ),
        "recall": bootstrap_confidence_interval(
            scored_cases, metrics["threshold"], "recall", seed=SEED + seed_offset + 1000
        ),
        "f1": bootstrap_confidence_interval(
            scored_cases, metrics["threshold"], "f1", seed=SEED + seed_offset + 2000
        ),
        "accuracy": bootstrap_confidence_interval(
            scored_cases, metrics["threshold"], "accuracy", seed=SEED + seed_offset + 3000
        ),
    }
    return metrics


def run_model(name, score_function, train_data, test_data, seed_offset=0):
    print(f"\n[INFO] Running {name}...")

    train_scored = score_dataset(train_data, score_function)
    best_threshold_metrics = find_best_threshold(train_scored, THRESHOLDS)
    test_scored = score_dataset(test_data, score_function)
    test_metrics = metrics_from_scored_cases(test_scored, best_threshold_metrics["threshold"])

    print(f"[INFO] {name} best validation threshold: {best_threshold_metrics['threshold']:.2f}")
    return {
        "validation": best_threshold_metrics,
        "test": attach_confidence_intervals(test_scored, test_metrics, seed_offset=seed_offset),
    }


def compute_feature_dataset(dataset):
    feature_rows = []

    for case in dataset:
        features = extract_feature_bundle(case["s1"], case["s2"])
        feature_rows.append(
            {
                "label": case["label"],
                "features": features,
            }
        )

    return feature_rows


def make_scored_cases_from_feature_dataset(feature_dataset, score_builder):
    return [
        {
            "label": row["label"],
            "score": score_builder(row["features"]),
        }
        for row in feature_dataset
    ]


def run_ablation_models(train_feature_rows, test_feature_rows):
    ablation_specs = {
        "local_only": lambda f: f["local_signal"],
        "minus_local_signal": lambda f: 0.25 * f["global_para"] + 0.35 * f["global_sem"],
        "minus_global_tfidf": lambda f: 0.40 * f["local_signal"] + 0.35 * f["global_sem"],
        "minus_global_sbert": lambda f: 0.40 * f["local_signal"] + 0.25 * f["global_para"],
    }

    ablation_results = {}

    for index, (name, builder) in enumerate(ablation_specs.items(), start=1):
        train_scored = make_scored_cases_from_feature_dataset(train_feature_rows, builder)
        best_threshold_metrics = find_best_threshold(train_scored, THRESHOLDS)
        test_scored = make_scored_cases_from_feature_dataset(test_feature_rows, builder)
        test_metrics = metrics_from_scored_cases(test_scored, best_threshold_metrics["threshold"])
        ablation_results[name] = {
            "validation": best_threshold_metrics,
            "test": attach_confidence_intervals(test_scored, test_metrics, seed_offset=4000 + index * 100),
        }

    return ablation_results


def feature_vector_from_bundle(features):
    return [
        features["global_para"],
        features["global_sem"],
        features["local_signal"],
        features["peak_local"],
        features["mean_top_local"],
        features["coverage"],
        features["exact_ratio"],
        features["match_count"],
        features["total_source_sentences"],
    ]


def run_learned_classifier(train_feature_rows, test_feature_rows):
    print("\n[INFO] Running learned classifier on hybrid features...")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=SEED, max_iter=1000)),
        ]
    )

    train_x = [feature_vector_from_bundle(row["features"]) for row in train_feature_rows]
    train_y = [row["label"] for row in train_feature_rows]
    test_x = [feature_vector_from_bundle(row["features"]) for row in test_feature_rows]

    model.fit(train_x, train_y)

    train_probabilities = model.predict_proba(train_x)[:, 1]
    test_probabilities = model.predict_proba(test_x)[:, 1]

    train_scored = [
        {"label": row["label"], "score": float(score)}
        for row, score in zip(train_feature_rows, train_probabilities)
    ]
    test_scored = [
        {"label": row["label"], "score": float(score)}
        for row, score in zip(test_feature_rows, test_probabilities)
    ]

    best_threshold_metrics = find_best_threshold(train_scored, THRESHOLDS)
    test_metrics = metrics_from_scored_cases(test_scored, best_threshold_metrics["threshold"])

    feature_names = [
        "global_para",
        "global_sem",
        "local_signal",
        "peak_local",
        "mean_top_local",
        "coverage",
        "exact_ratio",
        "match_count",
        "total_source_sentences",
    ]
    coefficients = model.named_steps["classifier"].coef_[0].tolist()

    print(f"[INFO] Learned classifier best validation threshold: {best_threshold_metrics['threshold']:.2f}")
    return {
        "validation": best_threshold_metrics,
        "test": attach_confidence_intervals(test_scored, test_metrics, seed_offset=9000),
        "feature_weights": {
            name: weight for name, weight in zip(feature_names, coefficients)
        },
    }


def print_results(name, result_bundle):
    validation = result_bundle["validation"]
    test = result_bundle["test"]

    print(f"\n{name}")
    print(
        "Validation Threshold:",
        round(validation["threshold"], 2),
        "| Validation F1:",
        round(validation["f1"], 4),
    )
    print("Precision:", round(test["precision"], 4))
    print("Recall:", round(test["recall"], 4))
    print("F1 Score:", round(test["f1"], 4))
    print("Accuracy:", round(test["accuracy"], 4))
    print(
        "F1 95% CI:",
        f"[{test['confidence_interval']['f1']['low']:.4f}, {test['confidence_interval']['f1']['high']:.4f}]",
    )


def summarize_ablation(ablation_results):
    print("\n===== ABLATION STUDY =====")
    for name, result in ablation_results.items():
        metrics = result["test"]
        print(
            name,
            "| F1:",
            round(metrics["f1"], 4),
            "| Precision:",
            round(metrics["precision"], 4),
            "| Recall:",
            round(metrics["recall"], 4),
        )


def main():
    random.seed(SEED)

    print("Loading dataset...")
    data = load_pan_dataset(DATA_PATH, limit=LIMIT, seed=SEED)
    random.shuffle(data)

    print("Dataset size:", len(data))
    print("Sample labels:", [d["label"] for d in data[:20]])

    train_data, test_data = split_dataset(data)
    print("Train size:", len(train_data))
    print("Test size:", len(test_data))

    print("\n[INFO] Preloading TF-IDF vectorizer...")
    prepare_vectorizer(data)
    print("[INFO] TF-IDF ready")

    start_time = time.time()

    tfidf_results = run_model("TF-IDF baseline", tfidf_only, train_data, test_data, seed_offset=100)
    sbert_results = run_model("SBERT baseline", sbert_only, train_data, test_data, seed_offset=200)
    hybrid_results = run_model("Hybrid model", hybrid, train_data, test_data, seed_offset=300)

    print("\n[INFO] Precomputing hybrid feature bundles...")
    train_feature_rows = compute_feature_dataset(train_data)
    test_feature_rows = compute_feature_dataset(test_data)

    ablation_results = run_ablation_models(train_feature_rows, test_feature_rows)
    learned_classifier_results = run_learned_classifier(train_feature_rows, test_feature_rows)

    end_time = time.time()

    print("\n===== FINAL COMPARISON =====")
    print_results("TF-IDF", tfidf_results)
    print_results("SBERT", sbert_results)
    print_results("HYBRID", hybrid_results)
    print_results("LEARNED CLASSIFIER", learned_classifier_results)
    summarize_ablation(ablation_results)
    print("\nTotal Time:", round(end_time - start_time, 2), "seconds")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    results = {
        "config": {
            "data_path": DATA_PATH,
            "limit": LIMIT,
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "dataset_size": len(data),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "execution_time_sec": end_time - start_time,
        "negative_sampling_note": (
            "Negative examples are sampled from suspicious-source combinations not linked in the PAN pairs file. "
            "This improves label quality, but the resulting subset may still be easier than a full-benchmark evaluation."
        ),
        "evaluation_note": (
            "Metrics are reported on a held-out test split with bootstrap confidence intervals. "
            "Given the modest test size, results should be interpreted as indicative rather than definitive."
        ),
        "tfidf": tfidf_results,
        "sbert": sbert_results,
        "hybrid": hybrid_results,
        "learned_classifier": learned_classifier_results,
        "ablations": ablation_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=4)

    print(f"\n[INFO] Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
