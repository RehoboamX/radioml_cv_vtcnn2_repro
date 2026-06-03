import math

import torch
from torch import nn
import torch.nn.functional as F


def split_complex(x):
    if x.shape[1] % 2:
        raise ValueError("Channel-stacked complex tensors need an even channel count.")
    return torch.chunk(x, 2, dim=1)


def complex_weight_init(real_weight, imag_weight, mode="glorot"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(real_weight)
    if mode == "glorot":
        sigma = 1.0 / math.sqrt(fan_in + fan_out)
    elif mode == "he":
        sigma = 1.0 / math.sqrt(fan_in)
    else:
        raise ValueError("mode must be 'glorot' or 'he'.")
    with torch.no_grad():
        uniform = torch.empty_like(real_weight).uniform_(1e-7, 1.0)
        modulus = sigma * torch.sqrt(-2.0 * torch.log(uniform))
        phase = torch.empty_like(real_weight).uniform_(-math.pi, math.pi)
        real_weight.copy_(modulus * torch.cos(phase))
        imag_weight.copy_(modulus * torch.sin(phase))


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        init_mode="glorot",
    ):
        super().__init__()
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.real = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.imag = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.real_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.imag_bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        complex_weight_init(self.real.weight, self.imag.weight, init_mode)

    def forward(self, x):
        xr, xi = split_complex(x)
        yr = self.real(xr) - self.imag(xi)
        yi = self.real(xi) + self.imag(xr)
        if self.real_bias is not None:
            yr = yr + self.real_bias.view(1, -1, 1)
            yi = yi + self.imag_bias.view(1, -1, 1)
        return torch.cat([yr, yi], dim=1)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="glorot"):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=False)
        self.imag = nn.Linear(in_features, out_features, bias=False)
        self.real_bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.imag_bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        complex_weight_init(self.real.weight, self.imag.weight, init_mode)

    def forward(self, x):
        xr, xi = split_complex(x)
        yr = F.linear(xr, self.real.weight) - F.linear(xi, self.imag.weight)
        yi = F.linear(xi, self.real.weight) + F.linear(xr, self.imag.weight)
        if self.real_bias is not None:
            yr = yr + self.real_bias
            yi = yi + self.imag_bias
        return torch.cat([yr, yi], dim=1)


class ComplexReLU(nn.Module):
    def forward(self, x):
        xr, xi = split_complex(x)
        return torch.cat([F.relu(xr), F.relu(xi)], dim=1)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        xr, xi = split_complex(x)
        mask = F.dropout(torch.ones_like(xr), p=self.p, training=True)
        return torch.cat([xr * mask, xi * mask], dim=1)


class ComplexBatchNorm1d(nn.Module):
    """Trabelsi-style 2x2 covariance whitening for stacked complex channels."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.register_buffer("running_mean", torch.zeros(2, num_features))
        self.register_buffer(
            "running_covar",
            torch.stack(
                [
                    torch.ones(num_features),
                    torch.zeros(num_features),
                    torch.ones(num_features),
                ]
            ),
        )
        if affine:
            scale = 1.0 / math.sqrt(2.0)
            self.gamma_rr = nn.Parameter(torch.full((num_features,), scale))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_ii = nn.Parameter(torch.full((num_features,), scale))
            self.beta_r = nn.Parameter(torch.zeros(num_features))
            self.beta_i = nn.Parameter(torch.zeros(num_features))

    def _stats(self, xr, xi):
        dims = (0,) + tuple(range(2, xr.ndim))
        mean_r = xr.mean(dim=dims)
        mean_i = xi.mean(dim=dims)
        shape = (1, -1) + (1,) * (xr.ndim - 2)
        cr = xr - mean_r.view(shape)
        ci = xi - mean_i.view(shape)
        vrr = (cr * cr).mean(dim=dims) + self.eps
        vii = (ci * ci).mean(dim=dims) + self.eps
        vri = (cr * ci).mean(dim=dims)
        return mean_r, mean_i, vrr, vri, vii

    def forward(self, x):
        xr, xi = split_complex(x)
        if self.training:
            mean_r, mean_i, vrr, vri, vii = self._stats(xr, xi)
            with torch.no_grad():
                self.running_mean.lerp_(torch.stack([mean_r, mean_i]), self.momentum)
                self.running_covar.lerp_(torch.stack([vrr, vri, vii]), self.momentum)
        else:
            mean_r, mean_i = self.running_mean
            vrr, vri, vii = self.running_covar

        shape = (1, -1) + (1,) * (xr.ndim - 2)
        cr = xr - mean_r.view(shape)
        ci = xi - mean_i.view(shape)
        covar = torch.stack(
            [
                torch.stack([vrr, vri], dim=-1),
                torch.stack([vri, vii], dim=-1),
            ],
            dim=-2,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covar)
        inv_sqrt = eigenvectors @ torch.diag_embed(eigenvalues.rsqrt()) @ eigenvectors.transpose(-1, -2)
        stacked = torch.stack([cr, ci], dim=-1)
        whitened = torch.einsum("cij,nc...j->nc...i", inv_sqrt, stacked)
        yr, yi = whitened.unbind(dim=-1)

        if self.affine:
            yr_affine = self.gamma_rr.view(shape) * yr + self.gamma_ri.view(shape) * yi
            yi_affine = self.gamma_ri.view(shape) * yr + self.gamma_ii.view(shape) * yi
            yr = yr_affine + self.beta_r.view(shape)
            yi = yi_affine + self.beta_i.view(shape)
        return torch.cat([yr, yi], dim=1)

