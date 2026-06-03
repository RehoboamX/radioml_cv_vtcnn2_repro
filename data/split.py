from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np


def _group_key(label):
    mod, snr = label
    return f"{mod}|{int(snr)}"


def build_balanced_split(labels, seed=2016, train_per_group=800, test_per_group=200):
    """Create a deterministic 800/200 train/test split per modulation and SNR."""
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[tuple(label)].append(index)

    rng = np.random.default_rng(seed)
    split = {"train": [], "test": []}
    required = train_per_group + test_per_group
    for label in sorted(groups, key=lambda item: (str(item[0]), int(item[1]))):
        indices = np.asarray(groups[label], dtype=np.int64)
        if len(indices) != required:
            raise ValueError(
                f"Expected {required} samples for {label}, found {len(indices)}."
            )
        shuffled = rng.permutation(indices)
        split["train"].extend(shuffled[:train_per_group])
        split["test"].extend(shuffled[train_per_group:])

    return {name: np.asarray(values, dtype=np.int64) for name, values in split.items()}


def validate_split_counts(labels, split, train_per_group=800, test_per_group=200):
    expected = {"train": train_per_group, "test": test_per_group}
    if set(split) != set(expected):
        raise AssertionError(f"Split keys must be {sorted(expected)}, found {sorted(split)}.")

    counts = {}
    for name, indices in split.items():
        counter = Counter(_group_key(labels[int(index)]) for index in indices)
        counts[name] = dict(sorted(counter.items()))
        bad = {key: value for key, value in counter.items() if value != expected[name]}
        if bad:
            raise AssertionError(
                f"{name} split has incorrect per-group counts: {bad}; "
                f"expected {expected[name]}."
            )

    train, test = (set(map(int, split[name])) for name in ("train", "test"))
    if train & test:
        raise AssertionError("Train and test splits overlap.")
    return counts


def save_split(split, counts, output_dir, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "split_indices.npz", **split)
    payload = dict(metadata)
    payload["counts"] = counts
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_split(output_dir):
    output_dir = Path(output_dir)
    with np.load(output_dir / "split_indices.npz") as archive:
        split = {name: archive[name].astype(np.int64) for name in archive.files}
    with (output_dir / "split_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return split, metadata