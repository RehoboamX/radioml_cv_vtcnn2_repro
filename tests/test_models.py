import torch
from torch import nn

from models.cv_vtcnn2 import CVVTCNN2
from models.vtcnn2 import VTCNN2


def test_vtcnn2_returns_real_logits():
    model = VTCNN2()
    logits = model(torch.randn(4, 2, 128))
    assert logits.shape == (4, 11)
    assert logits.is_floating_point()


def test_cv_vtcnn2_returns_finite_real_logits():
    model = CVVTCNN2(variant="same_width")
    logits = model(torch.randn(4, 2, 128))
    assert logits.shape == (4, 11)
    assert logits.is_floating_point()
    assert torch.isfinite(logits).all()
    assert isinstance(model.classifier, nn.Linear)
    assert not hasattr(model, "fc2")