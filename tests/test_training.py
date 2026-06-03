import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.checkpoint import load_checkpoint
from utils.training import run_training


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class NoOpOptimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        pass


def test_fixed_budget_training_saves_final_epoch_without_validation_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim, "Adam", NoOpOptimizer)
    samples = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    targets = torch.tensor([0, 1, 0, 1])
    snrs = torch.zeros(4, dtype=torch.int64)
    loader = DataLoader(TensorDataset(samples, targets, snrs), batch_size=2, shuffle=False)
    checkpoint_path = tmp_path / "model_final.pt"
    history_path = tmp_path / "history.json"

    run_training(
        model=TinyClassifier(),
        model_name="tiny",
        model_kwargs={},
        class_names=["A", "B"],
        train_loader=loader,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        max_epochs=2,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert checkpoint["epoch"] == 2
    assert checkpoint["checkpoint_selection"] == "final_epoch"
    assert "val_accuracy" not in checkpoint
    assert history["checkpoint_epoch"] == 2
    assert history["checkpoint_selection"] == "final_epoch"
    assert all("val_accuracy" not in row for row in history["history"])