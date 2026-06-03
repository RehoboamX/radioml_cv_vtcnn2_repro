import argparse
from pathlib import Path
import json

from utils.plotting import save_per_snr_curve, save_training_comparison


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot fixed-budget training and test results.")
    parser.add_argument("--vtcnn2-history", default="results/vtcnn2_history.json")
    parser.add_argument("--same-width-history", default="results/cv_vtcnn2_same_width_history.json")
    parser.add_argument("--param-matched-history", default="results/cv_vtcnn2_param_matched_history.json")
    parser.add_argument("--vtcnn2-metrics", default="results/vtcnn2_test_metrics.json")
    parser.add_argument("--same-width-metrics", default="results/cv_vtcnn2_same_width_test_metrics.json")
    parser.add_argument("--param-matched-metrics", default="results/cv_vtcnn2_param_matched_test_metrics.json")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    histories = {
        "VTCNN2": read_json(args.vtcnn2_history)["history"],
        "CV-VTCNN2 same-width": read_json(args.same_width_history)["history"],
        "CV-VTCNN2 param-matched": read_json(args.param_matched_history)["history"],
    }
    metrics = {
        "VTCNN2": read_json(args.vtcnn2_metrics)["per_snr_accuracy"],
        "CV-VTCNN2 same-width": read_json(args.same_width_metrics)["per_snr_accuracy"],
        "CV-VTCNN2 param-matched": read_json(args.param_matched_metrics)["per_snr_accuracy"],
    }
    save_training_comparison(histories, figures_dir)
    save_per_snr_curve(metrics, figures_dir / "per_snr_test_accuracy.png")
    print(f"saved figures to {figures_dir}")


if __name__ == "__main__":
    main()