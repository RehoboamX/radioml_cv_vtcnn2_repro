from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def load_radioml_dict(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"RadioML dataset not found at {path}. See README.md for placement."
        )
    with path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    return {(_decode(mod), int(snr)): np.asarray(samples) for (mod, snr), samples in raw.items()}


def flatten_radioml(dataset_dict):
    mods = sorted({key[0] for key in dataset_dict})
    snrs = sorted({int(key[1]) for key in dataset_dict})
    arrays = []
    labels = []
    for mod in mods:
        for snr in snrs:
            samples = np.asarray(dataset_dict[(mod, snr)], dtype=np.float32)
            if samples.ndim != 3 or samples.shape[1:] != (2, 128):
                raise ValueError(
                    f"Expected {(mod, snr)} samples with shape (N, 2, 128), "
                    f"found {samples.shape}."
                )
            arrays.append(samples)
            labels.extend([(mod, snr)] * len(samples))
    return np.concatenate(arrays, axis=0), labels, mods, snrs


class RadioMLDataset(Dataset):
    def __init__(self, samples, labels, class_names, indices):
        self.samples = samples
        self.labels = labels
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        mod, snr = self.labels[index]
        sample = torch.from_numpy(self.samples[index])
        target = self.class_to_index[mod]
        return sample, target, int(snr)


def load_datasets(dataset_path, split):
    samples, labels, mods, snrs = flatten_radioml(load_radioml_dict(dataset_path))
    datasets = {
        name: RadioMLDataset(samples, labels, mods, indices)
        for name, indices in split.items()
    }
    return datasets, labels, mods, snrs
