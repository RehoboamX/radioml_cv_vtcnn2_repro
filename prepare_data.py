import argparse
from pathlib import Path

from config import SEED, TEST_PER_GROUP, TRAIN_PER_GROUP, VAL_PER_GROUP
from data.dataset import flatten_radioml, load_radioml_dict
from data.split import build_balanced_split, save_split, validate_split_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a balanced RadioML2016.10A split.")
    parser.add_argument("--dataset", default="data/RML2016.10a_dict.dat")
    parser.add_argument("--output-dir", default="results/split")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--validation-mode",
        choices=("clean", "paper_like"),
        default="clean",
        help="paper_like reuses the test set as validation and can leak test information.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, labels, mods, snrs = flatten_radioml(load_radioml_dict(args.dataset))
    split = build_balanced_split(
        labels,
        seed=args.seed,
        train_per_group=TRAIN_PER_GROUP,
        val_per_group=VAL_PER_GROUP,
        test_per_group=TEST_PER_GROUP,
        validation_mode=args.validation_mode,
    )
    counts = validate_split_counts(
        labels,
        split,
        train_per_group=TRAIN_PER_GROUP,
        val_per_group=VAL_PER_GROUP,
        test_per_group=TEST_PER_GROUP,
        validation_mode=args.validation_mode,
    )
    metadata = {
        "dataset_path": str(Path(args.dataset)),
        "seed": args.seed,
        "validation_mode": args.validation_mode,
        "classes": mods,
        "snrs": snrs,
        "train_per_group": TRAIN_PER_GROUP,
        "val_per_group": VAL_PER_GROUP,
        "test_per_group": TEST_PER_GROUP,
    }
    save_split(split, counts, args.output_dir, metadata)
    print(f"saved balanced split to {args.output_dir}")
    print(f"classes={len(mods)} snrs={len(snrs)} validation_mode={args.validation_mode}")


if __name__ == "__main__":
    main()
