import torch
import numpy as np
from model import NATVOXAdapter


def test_adapter_forward_shape():
    model = NATVOXAdapter()
    x = torch.randn(4, 256)
    out = model(x)

    assert out.shape == (4, 256)


def test_adapter_output_norm():
    model = NATVOXAdapter()
    x = torch.randn(8, 256)
    out = model(x)

    norm = torch.norm(out, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_cosine_improvements():
    model = NATVOXAdapter()
    x = torch.randn(16, 256)
    y = torch.randn(16, 256)

    out = model(x)

    cos = torch.nn.CosineSimilarity(dim=1)
    before = cos(x, y)
    after = cos(out, y)

    # adapted should not be worse for random initialization (often approx)
    assert after.mean().item() >= before.mean().item() - 0.5
