def evaluate(dataset, score_function, threshold=0.15):
    scored_cases = score_dataset(dataset, score_function)
    return metrics_from_scored_cases(scored_cases, threshold)


def score_dataset(dataset, score_function):
    scored_cases = []

    for case in dataset:
        score = score_function(case["s1"], case["s2"])
        scored_cases.append(
            {
                "label": case["label"],
                "score": score,
            }
        )

    return scored_cases


def metrics_from_scored_cases(scored_cases, threshold):
    tp = fp = fn = tn = 0

    for case in scored_cases:
        actual = case["label"]
        pred = 1 if case["score"] >= threshold else 0

        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "count": total,
    }


def find_best_threshold(scored_cases, thresholds):
    best_metrics = None

    for threshold in thresholds:
        metrics = metrics_from_scored_cases(scored_cases, threshold)

        if best_metrics is None:
            best_metrics = metrics
            continue

        current_key = (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["accuracy"],
            -metrics["threshold"],
        )
        best_key = (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
            best_metrics["accuracy"],
            -best_metrics["threshold"],
        )

        if current_key > best_key:
            best_metrics = metrics

    return best_metrics
