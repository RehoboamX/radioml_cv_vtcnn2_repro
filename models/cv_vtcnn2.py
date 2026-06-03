import torch
from torch import nn

from models.complex_layers import (
    ComplexConv1d,
    ComplexDropout,
    ComplexLinear,
    ComplexReLU,
    split_complex,
)


VARIANTS = {
    "same_width": (256, 80, 256),
    "param_matched": (181, 57, 181),
}


class CVVTCNN2(nn.Module):
    """Complex-valued VTCNN2 with a real-valued classification head."""

    def __init__(self, num_classes=11, dropout=0.5, variant="same_width"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}.")
        conv1_channels, conv2_channels, dense_features = VARIANTS[variant]
        self.variant = variant
        self.conv1 = ComplexConv1d(1, conv1_channels, kernel_size=3, padding=2)
        self.act1 = ComplexReLU()
        self.drop1 = ComplexDropout(dropout)
        self.conv2 = ComplexConv1d(conv1_channels, conv2_channels, kernel_size=3, padding=2)
        self.act2 = ComplexReLU()
        self.drop2 = ComplexDropout(dropout)
        self.fc1 = ComplexLinear(conv2_channels * 132, dense_features, init_mode="he")
        self.act3 = ComplexReLU()
        self.drop3 = ComplexDropout(dropout)
        self.classifier = nn.Linear(dense_features * 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.drop1(self.act1(self.conv1(x)))
        x = self.drop2(self.act2(self.conv2(x)))
        xr, xi = split_complex(x)
        x = torch.cat(
            [torch.flatten(xr, start_dim=1), torch.flatten(xi, start_dim=1)], dim=1
        )
        x = self.drop3(self.act3(self.fc1(x)))
        xr, xi = split_complex(x)
        return self.classifier(torch.cat([xr, xi], dim=1))