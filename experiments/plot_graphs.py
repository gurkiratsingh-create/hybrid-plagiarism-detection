import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT_DIR / "results" / "comparison.json"


def load_results():
    with RESULTS_PATH.open(encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_metric_comparison(results):
    models = ["TF-IDF", "SBERT", "Hybrid", "Learned"]
    metric_names = ["precision", "recall", "f1", "accuracy"]
    metric_labels = ["Precision", "Recall", "F1", "Accuracy"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    test_results = [
        results["tfidf"]["test"],
        results["sbert"]["test"],
        results["hybrid"]["test"],
        results["learned_classifier"]["test"],
    ]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    for idx, (metric_name, label, color) in enumerate(zip(metric_names, metric_labels, colors)):
        offsets = x + (idx - 1.5) * width
        values = [result[metric_name] for result in test_results]
        bars = ax.bar(offsets, values, width=width, label=label, color=color)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Test-Set Performance Comparison Across Models")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_plot(results):
    models = ["TF-IDF", "SBERT", "Hybrid", "Learned"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    test_results = [
        results["tfidf"]["test"],
        results["sbert"]["test"],
        results["hybrid"]["test"],
        results["learned_classifier"]["test"],
    ]

    fig, ax = plt.subplots(figsize=(7.8, 5.7))

    for model, color, result in zip(models, colors, test_results):
        ax.scatter(result["recall"], result["precision"], s=180, color=color, label=model)
        ax.annotate(
            f"{model}\nT={result['threshold']:.2f}",
            (result["recall"], result["precision"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8.5,
        )

    ax.set_xlim(0.75, 1.01)
    ax.set_ylim(0.65, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Position of Evaluated Models")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "precision_recall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_plot(results):
    models = ["TF-IDF", "SBERT", "Hybrid", "Learned"]
    thresholds = [
        results["tfidf"]["test"]["threshold"],
        results["sbert"]["test"]["threshold"],
        results["hybrid"]["test"]["threshold"],
        results["learned_classifier"]["test"]["threshold"],
    ]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bars = ax.bar(models, thresholds, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    for bar, threshold in zip(bars, thresholds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            threshold + 0.01,
            f"{threshold:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, max(thresholds) + 0.1)
    ax.set_ylabel("Threshold")
    ax.set_title("Validation-Tuned Decision Thresholds")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "threshold_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_hybrid_confusion_matrix(results):
    hybrid = results["hybrid"]["test"]
    matrix = np.array([
        [hybrid["tn"], hybrid["fp"]],
        [hybrid["fn"], hybrid["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Non-Plagiarism", "Predicted Plagiarism"], rotation=15, ha="right")
    ax.set_yticklabels(["Actual Non-Plagiarism", "Actual Plagiarism"])
    ax.set_title("Hybrid Model Confusion Matrix (Test Set)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black", fontsize=12)

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "hybrid_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_ablation_plot(results):
    labels = ["Full Hybrid", "- Local", "- TF-IDF", "- SBERT", "Local Only"]
    values = [
        results["hybrid"]["test"]["f1"],
        results["ablations"]["minus_local_signal"]["test"]["f1"],
        results["ablations"]["minus_global_tfidf"]["test"]["f1"],
        results["ablations"]["minus_global_sbert"]["test"]["f1"],
        results["ablations"]["local_only"]["test"]["f1"],
    ]
    colors = ["#2ca02c", "#9467bd", "#1f77b4", "#ff7f0e", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    bars = ax.bar(labels, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation Study on Hybrid Components")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right")

    fig.tight_layout()
    fig.savefig(ROOT_DIR / "ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    results = load_results()
    save_metric_comparison(results)
    save_precision_recall_plot(results)
    save_threshold_plot(results)
    save_hybrid_confusion_matrix(results)
    save_ablation_plot(results)
    print("Graphs saved successfully.")


if __name__ == "__main__":
    main()
