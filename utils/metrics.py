from collections import defaultdict

import torch


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    snr_correct = defaultdict(int)
    snr_total = defaultdict(int)

    for samples, targets, snrs in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(samples)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        correct = predictions.eq(targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct.sum().item()
        total_examples += batch_size
        for snr, is_correct in zip(snrs.tolist(), correct.cpu().tolist()):
            snr_total[int(snr)] += 1
            snr_correct[int(snr)] += int(is_correct)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "per_snr_accuracy": {
            str(snr): snr_correct[snr] / snr_total[snr] for snr in sorted(snr_total)
        },
        "num_examples": total_examples,
    }
