def aggregate(exact, para, sem):
    if exact == 1.0:
        return 1.0

    if sem >= 0.90:
        weights = (0.10, 0.25, 0.65)
    elif sem >= 0.80:
        weights = (0.15, 0.35, 0.50)
    else:
        weights = (0.20, 0.45, 0.35)

    exact_weight, para_weight, sem_weight = weights
    return (
        exact_weight * exact
        + para_weight * para
        + sem_weight * sem
    )


def aggregate_document(
    match_scores,
    matched_source_count,
    total_source_count,
    global_para,
    global_sem,
    exact_ratio=0.0,
):
    if total_source_count == 0:
        return 0.0

    if match_scores:
        ordered_scores = sorted(match_scores, reverse=True)
        top_window = ordered_scores[: min(3, len(ordered_scores))]
        peak_signal = ordered_scores[0]
        mean_signal = sum(top_window) / len(top_window)
        coverage = matched_source_count / total_source_count
        local_signal = (
            0.45 * peak_signal
            + 0.30 * mean_signal
            + 0.15 * coverage
            + 0.10 * exact_ratio
        )
    else:
        local_signal = 0.0

    ensemble_score = (
        0.40 * local_signal
        + 0.25 * global_para
        + 0.35 * global_sem
    )

    return min(max(ensemble_score, 0.0), 1.0)
