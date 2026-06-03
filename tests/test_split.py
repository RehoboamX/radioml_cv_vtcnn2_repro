import numpy as np

from data.split import build_balanced_split, validate_split_counts


def make_labels():
    labels = []
    for mod in ("A", "B"):
        for snr in (-2, 0):
            labels.extend([(mod, snr)] * 1000)
    return labels


def test_balanced_split_has_expected_counts():
    labels = make_labels()
    split = build_balanced_split(labels, seed=7, train_per_group=720, val_per_group=80)

    counts = validate_split_counts(
        labels,
        split,
        train_per_group=720,
        val_per_group=80,
        test_per_group=200,
    )

    assert counts["train"]["A|-2"] == 720
    assert counts["val"]["B|0"] == 80
    assert counts["test"]["A|0"] == 200
    assert len(set(split["train"]) & set(split["test"])) == 0


def test_balanced_split_is_deterministic():
    labels = make_labels()
    first = build_balanced_split(labels, seed=2016)
    second = build_balanced_split(labels, seed=2016)
    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["val"], second["val"])
    assert np.array_equal(first["test"], second["test"])
