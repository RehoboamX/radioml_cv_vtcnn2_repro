from pathlib import Path

import torch


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(path, model, model_name, model_kwargs, class_names, epoch, val_accuracy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "class_names": list(class_names),
            "epoch": int(epoch),
            "val_accuracy": float(val_accuracy),
            "trainable_parameters": count_trainable_parameters(model),
        },
        path,
    )


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)
