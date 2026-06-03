import torch
import torch.nn.functional as F

from models.complex_layers import ComplexBatchNorm1d, ComplexConv1d, ComplexLinear


def test_complex_conv_matches_manual_formula():
    layer = ComplexConv1d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        layer.real.weight.copy_(torch.tensor([[[1.0, 2.0, -1.0]]]))
        layer.imag.weight.copy_(torch.tensor([[[0.5, -1.0, 1.5]]]))

    xr = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    xi = torch.tensor([[[0.5, -1.0, 2.0, 1.0]]])
    output = layer(torch.cat([xr, xi], dim=1))

    expected_r = F.conv1d(xr, layer.real.weight) - F.conv1d(xi, layer.imag.weight)
    expected_i = F.conv1d(xi, layer.real.weight) + F.conv1d(xr, layer.imag.weight)
    assert torch.allclose(output, torch.cat([expected_r, expected_i], dim=1))


def test_complex_linear_matches_manual_formula():
    layer = ComplexLinear(2, 1, bias=False)
    with torch.no_grad():
        layer.real.weight.copy_(torch.tensor([[1.0, -2.0]]))
        layer.imag.weight.copy_(torch.tensor([[0.5, 3.0]]))

    xr = torch.tensor([[2.0, -1.0]])
    xi = torch.tensor([[1.5, 4.0]])
    output = layer(torch.cat([xr, xi], dim=1))

    expected_r = F.linear(xr, layer.real.weight) - F.linear(xi, layer.imag.weight)
    expected_i = F.linear(xi, layer.real.weight) + F.linear(xr, layer.imag.weight)
    assert torch.allclose(output, torch.cat([expected_r, expected_i], dim=1))


def test_complex_batch_norm_returns_finite_stacked_output():
    layer = ComplexBatchNorm1d(3)
    output = layer(torch.randn(16, 6, 32))
    assert output.shape == (16, 6, 32)
    assert torch.isfinite(output).all()
