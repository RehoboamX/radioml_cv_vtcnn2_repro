from pathlib import Path

import matplotlib.pyplot as plt


def save_training_curves(history, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure()
    plt.plot(epochs, [row["train_loss"] for row in history])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_train_loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, [row["val_accuracy"] for row in history])
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_val_accuracy.png", dpi=160)
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
