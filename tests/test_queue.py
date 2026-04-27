"""Tests for FeatureQueue and combine_queues."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from yolo_contrastive.dense import FeatureQueue, combine_queues


K_SMALL = 64
DIM = 32


# ── basic enqueue / get ──────────────────────────────────────────────────


class TestEnqueueGet:
    def test_initially_empty(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        assert len(q) == 0
        assert not q.is_full
        assert q.get().shape == (0, DIM)

    def test_partial_fill(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        keys = torch.randn(10, DIM)
        q.enqueue(keys)
        assert len(q) == 10
        assert not q.is_full
        out = q.get()
        assert out.shape == (10, DIM)
        assert torch.allclose(out, keys)

    def test_full_fill(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        keys = torch.randn(K_SMALL, DIM)
        q.enqueue(keys)
        assert len(q) == K_SMALL
        assert q.is_full
        assert q.get().shape == (K_SMALL, DIM)

    def test_overflow_caps_at_K(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        for _ in range(5):
            q.enqueue(torch.randn(20, DIM))  # 100 total > 64
        assert len(q) == K_SMALL
        assert q.is_full

    def test_get_returns_clone(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        q.enqueue(torch.randn(10, DIM))
        out = q.get()
        out.zero_()
        # Internal buffer untouched by external mutation
        assert q.get().abs().sum() > 0


# ── FIFO semantics ───────────────────────────────────────────────────────


class TestFIFO:
    def test_oldest_evicted(self):
        """After overflow, oldest enqueued items are gone."""
        q = FeatureQueue(dim=DIM, K=4)
        a = torch.full((2, DIM), 1.0)
        b = torch.full((2, DIM), 2.0)
        c = torch.full((2, DIM), 3.0)  # evicts a
        d = torch.full((2, DIM), 4.0)  # evicts b
        q.enqueue(a); q.enqueue(b); q.enqueue(c); q.enqueue(d)
        unique_vals = set(q.get().unique().tolist())
        assert unique_vals == {3.0, 4.0}

    def test_ring_pointer_wraps(self):
        q = FeatureQueue(dim=DIM, K=4)
        q.enqueue(torch.randn(3, DIM))   # ptr=3
        q.enqueue(torch.randn(3, DIM))   # wraps: writes [3:4]+[0:2], ptr=2
        assert q.is_full
        assert int(q.ptr.item()) == 2

    def test_post_full_keeps_writing(self):
        """Once full, subsequent enqueues continue rotating data."""
        q = FeatureQueue(dim=DIM, K=4)
        q.enqueue(torch.full((4, DIM), 1.0))
        assert q.is_full
        q.enqueue(torch.full((2, DIM), 2.0))
        assert q.is_full
        # Should now contain mix of 1.0 and 2.0, with 2.0 being newer
        assert 2.0 in q.get().unique().tolist()


# ── scale tags ───────────────────────────────────────────────────────────


class TestTags:
    def test_with_tags_basic(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        keys = torch.randn(5, DIM)
        tags = torch.tensor([0, 1, 2, 0, 1])
        q.enqueue(keys, tags)
        k_out, t_out = q.get_all()
        assert k_out.shape == (5, DIM)
        assert torch.equal(t_out, tags)

    def test_without_tags_rejects_tags(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=False)
        with pytest.raises(ValueError, match="tags were provided"):
            q.enqueue(torch.randn(3, DIM), tags=torch.zeros(3, dtype=torch.long))

    def test_with_tags_requires_tags(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        with pytest.raises(ValueError, match="no tags provided"):
            q.enqueue(torch.randn(3, DIM))

    def test_get_tags_without_tags_raises(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=False)
        with pytest.raises(RuntimeError, match="with_tags=False"):
            q.get_tags()

    def test_get_all_without_tags_raises(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=False)
        with pytest.raises(RuntimeError, match="with_tags=False"):
            q.get_all()

    def test_tags_follow_ring_buffer(self):
        q = FeatureQueue(dim=DIM, K=4, with_tags=True)
        q.enqueue(torch.randn(2, DIM), tags=torch.tensor([100, 200]))
        q.enqueue(torch.randn(2, DIM), tags=torch.tensor([300, 400]))
        q.enqueue(torch.randn(2, DIM), tags=torch.tensor([500, 600]))
        # 100, 200 should be evicted
        t_list = q.get_tags().tolist()
        assert 100 not in t_list and 200 not in t_list
        assert 600 in t_list

    def test_wrong_tags_shape_raises(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        with pytest.raises(ValueError, match="tags must have shape"):
            q.enqueue(torch.randn(5, DIM), tags=torch.zeros(3, dtype=torch.long))


# ── normalization integration ────────────────────────────────────────────


class TestNormalization:
    def test_stores_normalized_vectors_unchanged(self):
        """If caller normalizes, retrieved norms should be ~1."""
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        normed = F.normalize(torch.randn(20, DIM), dim=-1)
        q.enqueue(normed)
        norms = q.get().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ── edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_enqueue_empty_batch(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        q.enqueue(torch.randn(5, DIM))
        n_before = len(q)
        q.enqueue(torch.zeros(0, DIM))
        assert len(q) == n_before

    def test_enqueue_b_greater_than_k(self):
        """B > K: only the last K entries should remain."""
        q = FeatureQueue(dim=DIM, K=4)
        keys = torch.full((10, DIM), 1.0)
        keys[-4:] = 2.0
        q.enqueue(keys)
        unique = q.get().unique().tolist()
        assert 2.0 in unique
        assert 1.0 not in unique

    def test_wrong_dim_raises(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        with pytest.raises(ValueError, match="dim"):
            q.enqueue(torch.randn(5, DIM + 1))

    def test_wrong_shape_raises(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        with pytest.raises(ValueError, match="2-D"):
            q.enqueue(torch.randn(DIM))

    def test_invalid_constructor(self):
        with pytest.raises(ValueError, match="dim"):
            FeatureQueue(dim=0, K=K_SMALL)
        with pytest.raises(ValueError, match="K"):
            FeatureQueue(dim=DIM, K=0)


# ── lifecycle ────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_reset(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        q.enqueue(torch.randn(20, DIM))
        q.reset()
        assert len(q) == 0
        assert int(q.ptr.item()) == 0
        assert q.get().shape == (0, DIM)

    def test_reset_with_tags(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        q.enqueue(torch.randn(5, DIM), tags=torch.zeros(5, dtype=torch.long))
        q.reset()
        assert len(q) == 0
        assert q.get_tags().shape == (0,)

    def test_state_dict_roundtrip(self):
        q1 = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        keys = torch.randn(20, DIM)
        tags = torch.randint(0, 3, (20,))
        q1.enqueue(keys, tags)

        sd = q1.state_dict()
        q2 = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        q2.load_state_dict(sd)

        assert torch.equal(q1.get(), q2.get())
        assert torch.equal(q1.get_tags(), q2.get_tags())
        assert len(q1) == len(q2)


# ── detach safety ────────────────────────────────────────────────────────


class TestNoGrad:
    def test_enqueue_detaches(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL)
        keys = torch.randn(10, DIM, requires_grad=True)
        q.enqueue(keys)
        out = q.get()
        assert not out.requires_grad


# ── combine_queues ───────────────────────────────────────────────────────


class TestCombineQueues:
    def test_basic_combine(self):
        q3 = FeatureQueue(dim=DIM, K=K_SMALL)
        q4 = FeatureQueue(dim=DIM, K=K_SMALL)
        q5 = FeatureQueue(dim=DIM, K=K_SMALL)
        q3.enqueue(torch.full((5, DIM), 3.0))
        q4.enqueue(torch.full((7, DIM), 4.0))
        q5.enqueue(torch.full((3, DIM), 5.0))
        keys, tags = combine_queues({"P3": q3, "P4": q4, "P5": q5})
        assert keys.shape == (15, DIM)
        assert tags.shape == (15,)
        assert (keys[tags == 0] == 3.0).all()
        assert (keys[tags == 1] == 4.0).all()
        assert (keys[tags == 2] == 5.0).all()

    def test_combine_with_custom_ids(self):
        q3 = FeatureQueue(dim=DIM, K=K_SMALL)
        q5 = FeatureQueue(dim=DIM, K=K_SMALL)
        q3.enqueue(torch.randn(3, DIM))
        q5.enqueue(torch.randn(3, DIM))
        _, tags = combine_queues(
            {"P3": q3, "P5": q5},
            level_to_id={"P3": 8, "P5": 32},
        )
        assert set(tags.unique().tolist()) == {8, 32}

    def test_combine_all_empty(self):
        q3 = FeatureQueue(dim=DIM, K=K_SMALL)
        q5 = FeatureQueue(dim=DIM, K=K_SMALL)
        keys, tags = combine_queues({"P3": q3, "P5": q5})
        assert keys.shape == (0, DIM)
        assert tags.shape == (0,)

    def test_combine_some_empty(self):
        q3 = FeatureQueue(dim=DIM, K=K_SMALL)
        q5 = FeatureQueue(dim=DIM, K=K_SMALL)
        q5.enqueue(torch.randn(4, DIM))
        keys, tags = combine_queues({"P3": q3, "P5": q5})
        assert keys.shape == (4, DIM)
        assert (tags == 1).all()  # P5 → id 1

    def test_dim_mismatch_raises(self):
        q1 = FeatureQueue(dim=32, K=K_SMALL)
        q2 = FeatureQueue(dim=64, K=K_SMALL)
        q1.enqueue(torch.randn(2, 32))
        q2.enqueue(torch.randn(2, 64))
        with pytest.raises(ValueError, match="dim"):
            combine_queues({"a": q1, "b": q2})

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            combine_queues({})


# ── repr ─────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr(self):
        q = FeatureQueue(dim=DIM, K=K_SMALL, with_tags=True)
        q.enqueue(torch.randn(5, DIM), tags=torch.zeros(5, dtype=torch.long))
        r = repr(q)
        assert "FeatureQueue" in r
        assert "dim=32" in r
        assert "K=64" in r
        assert "filled=5" in r
        assert "with_tags=True" in r
