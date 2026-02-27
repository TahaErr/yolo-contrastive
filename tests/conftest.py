"""Shared test fixtures for yolo-contrastive."""

import pytest
import torch


@pytest.fixture
def device():
    """Return 'cuda' if available, else 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=[2, 4, 8], ids=["B=2", "B=4", "B=8"])
def batch_size(request):
    return request.param


@pytest.fixture(params=[64, 256], ids=["D=64", "D=256"])
def embed_dim(request):
    return request.param
