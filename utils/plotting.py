from pathlib import Path

import matplotlib.pyplot as plt


def save_training_comparison(histories_by_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in (
        ("train_loss", "Training loss", "train_loss_comparison.png"),
        ("train_accuracy", "Training accuracy", "train_accuracy_comparison.png"),
    ):
        plt.figure()
        for name, history in histories_by_name.items():
            epochs = [row["epoch"] for row in history]
            values = [row[metric] for row in history]
            plt.plot(epochs, values, label=name)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def save_per_snr_curve(series_by_name, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for name, per_snr in series_by_name.items():
        snrs = sorted(int(value) for value in per_snr)
        values = [per_snr[str(snr)] for snr in snrs]
        plt.plot(snrs, values, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()