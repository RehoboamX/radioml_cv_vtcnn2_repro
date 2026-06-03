import argparse
from pathlib import Path
import json

from utils.plotting import save_per_snr_curve, save_training_curves


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training and test results.")
    parser.add_argument("--vtcnn2-history", default="results/vtcnn2_history.json")
    parser.add_argument("--cv-history", default="results/cv_vtcnn2_history.json")
    parser.add_argument("--vtcnn2-metrics", default="results/vtcnn2_test_metrics.json")
    parser.add_argument("--cv-metrics", default="results/cv_vtcnn2_test_metrics.json")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    vt_history = read_json(args.vtcnn2_history)
    cv_history = read_json(args.cv_history)
    vt_metrics = read_json(args.vtcnn2_metrics)
    cv_metrics = read_json(args.cv_metrics)
    save_training_curves(vt_history["history"], figures_dir, "vtcnn2")
    save_training_curves(cv_history["history"], figures_dir, "cv_vtcnn2")
    save_per_snr_curve(
        {
            "VTCNN2": vt_metrics["per_snr_accuracy"],
            "CV-VTCNN2": cv_metrics["per_snr_accuracy"],
        },
        figures_dir / "per_snr_test_accuracy.png",
    )
    print(f"saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
