import argparse

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import flatten_radioml, load_radioml_dict
from data.split import build_balanced_split, validate_split_counts
from models.complex_layers import ComplexConv1d, ComplexLinear
from models.cv_vtcnn2 import CVVTCNN2
from models.vtcnn2 import VTCNN2
from utils.checkpoint import count_trainable_parameters


def check_complex_conv():
    layer = ComplexConv1d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        layer.real.weight.copy_(torch.tensor([[[1.0, 2.0, -1.0]]]))
        layer.imag.weight.copy_(torch.tensor([[[0.5, -1.0, 1.5]]]))
    xr = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    xi = torch.tensor([[[0.5, -1.0, 2.0, 1.0]]])
    actual = layer(torch.cat([xr, xi], dim=1))
    expected = torch.cat(
        [
            F.conv1d(xr, layer.real.weight) - F.conv1d(xi, layer.imag.weight),
            F.conv1d(xi, layer.real.weight) + F.conv1d(xr, layer.imag.weight),
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)


def check_complex_linear():
    layer = ComplexLinear(2, 1, bias=False)
    with torch.no_grad():
        layer.real.weight.copy_(torch.tensor([[1.0, -2.0]]))
        layer.imag.weight.copy_(torch.tensor([[0.5, 3.0]]))
    xr = torch.tensor([[2.0, -1.0]])
    xi = torch.tensor([[1.5, 4.0]])
    actual = layer(torch.cat([xr, xi], dim=1))
    expected = torch.cat(
        [
            F.linear(xr, layer.real.weight) - F.linear(xi, layer.imag.weight),
            F.linear(xi, layer.real.weight) + F.linear(xr, layer.imag.weight),
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)


def check_models():
    samples = torch.randn(2, 2, 128)
    real_logits = VTCNN2()(samples)
    complex_logits = CVVTCNN2()(samples)
    assert real_logits.shape == complex_logits.shape == (2, 11)
    assert torch.isfinite(real_logits).all() and torch.isfinite(complex_logits).all()
    print(f"VTCNN2 trainable parameters: {count_trainable_parameters(VTCNN2()):,}")
    print(
        "CV-VTCNN2 paper_faithful trainable parameters: "
        f"{count_trainable_parameters(CVVTCNN2()):,}"
    )
    print(
        "CV-VTCNN2 param_matched trainable parameters: "
        f"{count_trainable_parameters(CVVTCNN2(variant='param_matched')):,}"
    )


def check_split(dataset_path):
    _, labels, _, _ = flatten_radioml(load_radioml_dict(dataset_path))
    split = build_balanced_split(labels)
    validate_split_counts(labels, split)


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproduction sanity checks.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional RadioML2016.10A pickle path for the full split check.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_complex_conv()
    check_complex_linear()
    check_models()
    if args.dataset:
        check_split(args.dataset)
        print("balanced split check: passed")
    else:
        print("balanced split check: skipped (pass --dataset to run it)")
    print("complex arithmetic and model checks: passed")


if __name__ == "__main__":
    main()
