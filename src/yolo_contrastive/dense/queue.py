"""Memory queue for MoCo-style contrastive learning.

Faz 1.2 — Foundation for Dense + Multi-scale CL (WORK_PLAN_v3 §5).

A FIFO ring buffer that stores [B, dim] embedding vectors across many
training iterations. Momentum-encoded keys from past batches form a
large pool of negatives without re-computing them every step.

Key design choices:
    - Caller passes L2-normalized, detached embeddings (MoCo convention).
    - get() returns [num_filled, dim] — never zero-padded; empty slots
      cannot leak into similarity computation.
    - Optional per-entry scale tag for cross-scale SAPS in Faz 2.2.
      Use one FeatureQueue per FPN level normally; combine_queues()
      produces a tagged pool for cross-scale operations.
    - nn.Module subclass — register_buffer gives state_dict /
      .to(device) for free.

Single-GPU only for now. DDP support (gather keys across replicas
before enqueue) deferred — see WORK_PLAN_v3 risk #2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class FeatureQueue(nn.Module):
    """FIFO ring buffer for embedding vectors with optional scale tags.

    Args:
        dim: Embedding dimension.
        K: Maximum capacity.
        with_tags: If True, allocate a parallel long-tensor for per-entry tags.
        dtype: Storage dtype for the queue (float32 recommended for stability).

    Usage:
        queue = FeatureQueue(dim=256, K=65536).to(device)
        keys = F.normalize(momentum_encoder(view2).detach(), dim=-1)
        queue.enqueue(keys)
        negatives = queue.get()              # [N, dim] — N <= K
    """

    def __init__(
        self,
        dim: int,
        K: int = 65536,
        with_tags: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")

        self.dim = int(dim)
        self.K = int(K)
        self.with_tags = bool(with_tags)

        self.register_buffer("queue", torch.zeros(K, dim, dtype=dtype))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("num_filled", torch.zeros(1, dtype=torch.long))
        if with_tags:
            self.register_buffer("tags", torch.zeros(K, dtype=torch.long))

    # ── enqueue ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def enqueue(
        self,
        keys: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch of keys (FIFO, ring-buffer).

        Args:
            keys: [B, dim] — caller's responsibility to L2-normalize.
            tags: [B] long — required iff with_tags=True.

        Behaviour:
            - B == 0: no-op
            - B > K: only the last K rows are kept (older ones overwritten
              within this single call as well)
        """
        if keys.dim() != 2:
            raise ValueError(
                f"keys must be 2-D [B, dim], got shape {tuple(keys.shape)}"
            )
        if keys.shape[1] != self.dim:
            raise ValueError(
                f"keys dim {keys.shape[1]} != queue dim {self.dim}"
            )

        B = keys.shape[0]
        if B == 0:
            return

        # If a single batch overflows K, only the tail K rows survive
        if B > self.K:
            keys = keys[-self.K:]
            if tags is not None:
                tags = tags[-self.K:]
            B = self.K

        if self.with_tags:
            if tags is None:
                raise ValueError("with_tags=True but no tags provided to enqueue()")
            if tags.shape != (B,):
                raise ValueError(
                    f"tags must have shape [{B}], got {tuple(tags.shape)}"
                )
        else:
            if tags is not None:
                raise ValueError("with_tags=False but tags were provided")

        keys = keys.detach().to(device=self.queue.device, dtype=self.queue.dtype)
        if tags is not None:
            tags = tags.detach().to(device=self.queue.device, dtype=torch.long)

        ptr = int(self.ptr.item())
        end = ptr + B

        if end <= self.K:
            self.queue[ptr:end].copy_(keys)
            if tags is not None:
                self.tags[ptr:end].copy_(tags)
            new_ptr = end % self.K
        else:
            # Wraparound: split write
            first = self.K - ptr
            self.queue[ptr:self.K].copy_(keys[:first])
            self.queue[0:B - first].copy_(keys[first:])
            if tags is not None:
                self.tags[ptr:self.K].copy_(tags[:first])
                self.tags[0:B - first].copy_(tags[first:])
            new_ptr = B - first

        self.ptr[0] = new_ptr
        self.num_filled[0] = min(int(self.num_filled.item()) + B, self.K)

    # ── access ────────────────────────────────────────────────────────────

    def get(self) -> torch.Tensor:
        """Return current keys as [num_filled, dim]. Cloned, never zero-padded."""
        n = int(self.num_filled.item())
        if n < self.K:
            return self.queue[:n].clone()
        return self.queue.clone()

    def get_tags(self) -> torch.Tensor:
        """Return current tags as [num_filled]. Requires with_tags=True."""
        if not self.with_tags:
            raise RuntimeError("Queue was created with_tags=False; no tags to return")
        n = int(self.num_filled.item())
        if n < self.K:
            return self.tags[:n].clone()
        return self.tags.clone()

    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, tags) atomically. Requires with_tags=True."""
        if not self.with_tags:
            raise RuntimeError("Queue was created with_tags=False; use get() instead")
        return self.get(), self.get_tags()

    # ── status ────────────────────────────────────────────────────────────

    @property
    def is_full(self) -> bool:
        return int(self.num_filled.item()) >= self.K

    def __len__(self) -> int:
        return int(self.num_filled.item())

    def reset(self) -> None:
        """Clear all entries and reset pointers."""
        self.queue.zero_()
        self.ptr.zero_()
        self.num_filled.zero_()
        if self.with_tags:
            self.tags.zero_()

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FeatureQueue(dim={self.dim}, K={self.K}, "
            f"filled={len(self)}, with_tags={self.with_tags})"
        )


# ─────────────────────────────────────────────────────────────────────────


def combine_queues(
    queues: Dict[str, "FeatureQueue"],
    level_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate multiple queues into a single tagged pool.

    For Faz 2.2 cross-scale SAPS: queries from level L can be compared
    against keys from all levels with scale-similarity weighting based
    on the returned tag tensor.

    Args:
        queues: dict {"P3": queue_p3, "P4": queue_p4, ...}.
        level_to_id: dict mapping level name → integer tag id.
                     Default: enumeration order of queues.keys().

    Returns:
        (keys [N_total, dim], tags [N_total]).
    """
    if not queues:
        raise ValueError("queues dict is empty")

    if level_to_id is None:
        level_to_id = {name: i for i, name in enumerate(queues.keys())}

    keys_list, tags_list = [], []
    dim_ref: Optional[int] = None
    device_ref: Optional[torch.device] = None
    dtype_ref: Optional[torch.dtype] = None

    for name, q in queues.items():
        if name not in level_to_id:
            raise ValueError(f"No id for level {name!r} in level_to_id")
        k = q.get()
        if dim_ref is None:
            dim_ref = k.shape[1]
            device_ref = k.device
            dtype_ref = k.dtype
        elif k.shape[1] != dim_ref:
            raise ValueError(
                f"Queue {name!r} dim {k.shape[1]} != reference dim {dim_ref}"
            )
        if k.shape[0] == 0:
            continue
        tag_value = level_to_id[name]
        tags = torch.full(
            (k.shape[0],), tag_value, dtype=torch.long, device=k.device,
        )
        keys_list.append(k)
        tags_list.append(tags)

    if not keys_list:
        return (
            torch.empty(0, dim_ref or 0, device=device_ref, dtype=dtype_ref),
            torch.empty(0, dtype=torch.long, device=device_ref),
        )
    return torch.cat(keys_list, dim=0), torch.cat(tags_list, dim=0)
