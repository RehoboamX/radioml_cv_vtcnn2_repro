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
    split = build_balanced_split(labels, seed=7)
    counts = validate_split_counts(labels, split)

    assert set(split) == {"train", "test"}
    assert counts["train"]["A|-2"] == 800
    assert counts["test"]["B|0"] == 200
    assert len(set(split["train"]) & set(split["test"])) == 0


def test_balanced_split_is_deterministic():
    labels = make_labels()
    first = build_balanced_split(labels, seed=2016)
    second = build_balanced_split(labels, seed=2016)
    assert np.array_equal(first["train"], second["train"])
    assert np.array_equal(first["test"], second["test"])