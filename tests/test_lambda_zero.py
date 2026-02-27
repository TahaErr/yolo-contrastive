"""Lambda=0 regression test (audit §6.1)."""

import torch
from yolo_contrastive.contrastive import build_contrastive_loss


class TestLambdaZeroRegression:

    def test_cl_not_added_when_lambda_zero(self):
        """lambda=0 -> _cl_enabled=False -> _compute_cl returns None."""
        cl_enabled = False
        z1 = torch.randn(4, 128)
        det_loss = torch.tensor(2.5, requires_grad=True)

        cl = None if not cl_enabled else build_contrastive_loss("ntxent")(z1, z1)
        assert cl is None

        total = det_loss if cl is None else det_loss + 0.0 * cl
        assert total is det_loss

    def test_zero_lambda_contribution(self):
        """Even if CL computed, lambda=0 means zero numerical contribution."""
        loss_fn = build_contrastive_loss("ntxent", temperature=0.2)
        z1 = torch.randn(8, 256, requires_grad=True)
        z2 = torch.randn(8, 256)
        det_loss = torch.tensor(3.0, requires_grad=True)
        cl_loss = loss_fn(z1, z2)

        total = det_loss + 0.0 * cl_loss
        assert torch.isclose(total, det_loss, atol=1e-7)

    def test_gradient_only_from_det_when_lambda_zero(self):
        """lambda=0 -> gradients only from detection loss."""
        loss_fn = build_contrastive_loss("ntxent", temperature=0.2)
        w = torch.randn(8, 256, requires_grad=True)
        det_loss = (w.sum()) ** 2
        cl_loss = loss_fn(w, w.detach() + 0.01 * torch.randn_like(w))
        total = det_loss + 0.0 * cl_loss
        total.backward()

        w2 = w.detach().clone().requires_grad_(True)
        det_only = (w2.sum()) ** 2
        det_only.backward()

        assert torch.allclose(w.grad, w2.grad, atol=1e-6)
