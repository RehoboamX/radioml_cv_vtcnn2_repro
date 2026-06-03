import argparse
from pathlib import Path

from config import SEED, TEST_PER_GROUP, TRAIN_PER_GROUP
from data.dataset import flatten_radioml, load_radioml_dict
from data.split import build_balanced_split, save_split, validate_split_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the balanced RadioML2016.10A split.")
    parser.add_argument("--dataset", default="data/RML2016.10a_dict.dat")
    parser.add_argument("--output-dir", default="results/split")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    _, labels, mods, snrs = flatten_radioml(load_radioml_dict(args.dataset))
    split = build_balanced_split(
        labels,
        seed=args.seed,
        train_per_group=TRAIN_PER_GROUP,
        test_per_group=TEST_PER_GROUP,
    )
    counts = validate_split_counts(
        labels,
        split,
        train_per_group=TRAIN_PER_GROUP,
        test_per_group=TEST_PER_GROUP,
    )
    metadata = {
        "dataset_path": str(Path(args.dataset)),
        "seed": args.seed,
        "split_protocol": "per_modulation_snr_800_train_200_test",
        "classes": mods,
        "snrs": snrs,
        "train_per_group": TRAIN_PER_GROUP,
        "test_per_group": TEST_PER_GROUP,
    }
    save_split(split, counts, args.output_dir, metadata)
    print(f"saved balanced split to {args.output_dir}")
    print(f"classes={len(mods)} snrs={len(snrs)} train=800 test=200 per group")


if __name__ == "__main__":
    main()