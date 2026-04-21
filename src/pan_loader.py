import os
import random


def _read_text(file_path):
    with open(file_path, encoding="utf-8", errors="ignore") as file_obj:
        return file_obj.read()


def _load_pair_map(pairs_path):
    pair_map = {}

    with open(pairs_path, encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            suspicious_file, source_file = parts
            pair_map.setdefault(suspicious_file, set()).add(source_file)

    return pair_map


def _create_positive_cases(pair_map, src_path, susp_path):
    positives = []

    for suspicious_file, source_files in pair_map.items():
        suspicious_text = _read_text(os.path.join(susp_path, suspicious_file))

        for source_file in source_files:
            source_text = _read_text(os.path.join(src_path, source_file))
            positives.append(
                {
                    "s1": suspicious_text,
                    "s2": source_text,
                    "label": 1,
                    "suspicious_file": suspicious_file,
                    "source_file": source_file,
                }
            )

    return positives


def _create_negative_cases(pair_map, src_files, src_path, susp_path, n_samples, rng):
    negatives = []
    suspicious_files = list(pair_map.keys())

    while len(negatives) < n_samples:
        suspicious_file = rng.choice(suspicious_files)
        source_file = rng.choice(src_files)

        if source_file in pair_map[suspicious_file]:
            continue

        suspicious_text = _read_text(os.path.join(susp_path, suspicious_file))
        source_text = _read_text(os.path.join(src_path, source_file))

        negatives.append(
            {
                "s1": suspicious_text,
                "s2": source_text,
                "label": 0,
                "suspicious_file": suspicious_file,
                "source_file": source_file,
            }
        )

    return negatives


def load_pan_dataset(base_path, limit=30, seed=42):
    src_path = os.path.join(base_path, "src")
    susp_path = os.path.join(base_path, "susp")
    pairs_path = os.path.join(base_path, "pairs")

    pair_map = _load_pair_map(pairs_path)
    suspicious_files = sorted(pair_map.keys())[:limit]
    limited_pair_map = {file_name: pair_map[file_name] for file_name in suspicious_files}

    positives = _create_positive_cases(limited_pair_map, src_path, susp_path)

    src_files = sorted(os.listdir(src_path))
    rng = random.Random(seed)
    negatives = _create_negative_cases(
        limited_pair_map,
        src_files,
        src_path,
        susp_path,
        n_samples=len(positives),
        rng=rng,
    )

    dataset = positives + negatives
    rng.shuffle(dataset)
    return dataset
