import argparse

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, LEARNING_RATE, MAX_EPOCHS, SEED
from data.dataset import load_datasets
from data.split import load_split
from models.vtcnn2 import VTCNN2
from utils.seed import set_seed
from utils.training import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train the real-valued VTCNN2 baseline.")
    parser.add_argument("--dataset", default="data/RML2016.10a_dict.dat")
    parser.add_argument("--split-dir", default="results/split")
    parser.add_argument("--checkpoint", default="results/vtcnn2_best.pt")
    parser.add_argument("--history", default="results/vtcnn2_history.json")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split, metadata = load_split(args.split_dir)
    datasets, _, class_names, _ = load_datasets(args.dataset, split)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model_kwargs = {"num_classes": len(class_names), "dropout": 0.5}
    model = VTCNN2(**model_kwargs).to(device)
    print(f"device={device} validation_mode={metadata['validation_mode']}")
    run_training(
        model=model,
        model_name="vtcnn2",
        model_kwargs=model_kwargs,
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
        history_path=args.history,
    )


if __name__ == "__main__":
    main()
