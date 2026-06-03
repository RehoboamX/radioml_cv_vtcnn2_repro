from pathlib import Path
import json

import torch

from utils.checkpoint import count_trainable_parameters, save_checkpoint
from utils.metrics import evaluate_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for samples, targets, _ in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(samples)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total_examples += batch_size
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def run_training(
    model,
    model_name,
    model_kwargs,
    class_names,
    train_loader,
    val_loader,
    device,
    learning_rate,
    max_epochs,
    patience,
    checkpoint_path,
    history_path,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_accuracy = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={record['train_loss']:.5f} "
            f"train_acc={record['train_accuracy']:.4f} "
            f"val_loss={record['val_loss']:.5f} val_acc={record['val_accuracy']:.4f}"
        )

        if record["val_accuracy"] > best_accuracy:
            best_accuracy = record["val_accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                model_name,
                model_kwargs,
                class_names,
                epoch,
                best_accuracy,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping after {epoch} epochs")
                break

    payload = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "trainable_parameters": count_trainable_parameters(model),
        "best_val_accuracy": best_accuracy,
        "history": history,
    }
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
