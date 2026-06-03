import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE
from data.dataset import load_datasets
from data.split import load_split
from models.cv_vtcnn2 import CVVTCNN2
from models.vtcnn2 import VTCNN2
from utils.checkpoint import load_checkpoint
from utils.metrics import evaluate_loader


MODEL_CLASSES = {"vtcnn2": VTCNN2, "cv_vtcnn2": CVVTCNN2}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a final VTCNN2 checkpoint.")
    parser.add_argument("--dataset", default="data/RML2016.10a_dict.dat")
    parser.add_argument("--split-dir", default="results/split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_name = checkpoint["model_name"]
    model = MODEL_CLASSES[model_name](**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    split, metadata = load_split(args.split_dir)
    datasets, _, _, _ = load_datasets(args.dataset, split)
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_loader(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    metrics.update(
        {
            "model_name": model_name,
            "model_kwargs": checkpoint["model_kwargs"],
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_selection": checkpoint["checkpoint_selection"],
            "trainable_parameters": checkpoint["trainable_parameters"],
            "split_protocol": metadata["split_protocol"],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"saved metrics to {output}")


if __name__ == "__main__":
    main()