"""Synthetic pinhole scenes with analytic metric depth — test/audit ground truth.

Renders scenes of textured squares of known PHYSICAL size at known METRIC
depths under an ideal pinhole camera (apparent side = f * L / Z), together
with the EXACT inverse-depth map. CPU tests mine pairs on these scenes with a
stub embedder and assert that every recovered label equals
``log(Z_A / Z_B)`` analytically — the geometry of the GASP-Real label factory
under test with zero model dependencies.

A ROW-DECORRELATED layout (same image row, different depths via different
physical sizes) is supported to prove labels derive from depth, not from the
vertical position shortcut (the GASP-Real analogue of the blur shortcut).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SyntheticSquare:
    """One textured square: physical side ``size_m`` at metric depth ``z_m``.

    ``cx, cy`` are the normalized image-center coordinates of the square;
    ``class_id`` selects the texture pattern (squares of the same class are
    "similar content" for the stub embedder).
    """

    z_m: float
    size_m: float
    cx: float
    cy: float
    class_id: int

    def apparent_side_px(self, focal_px: float) -> float:
        return focal_px * self.size_m / self.z_m


@dataclass
class SyntheticScene:
    """A rendered scene + analytic ground truth."""

    image: np.ndarray                 # [H, W, 3] float32 in [0, 1]
    inv_depth: np.ndarray             # [H, W] float32, exact metric 1/Z
    squares: List[SyntheticSquare]
    boxes_norm: np.ndarray            # [N, 4] normalized xyxy of each square
    focal_px: float
    background_z_m: float
    rng_seed: int = 0
    _texture_std: float = field(default=0.0, repr=False)

    @property
    def h(self) -> int:
        return int(self.image.shape[0])

    @property
    def w(self) -> int:
        return int(self.image.shape[1])

    def square_index_for_box(self, box_xyxy_norm: np.ndarray) -> Optional[int]:
        """Index of the square FULLY containing ``box``, or None.

        Used by tests to map a mined candidate box back to its ground-truth
        depth: a candidate inside square i has median Z exactly ``z_m[i]``.
        """
        b = np.asarray(box_xyxy_norm, dtype=np.float64)
        for i, sq in enumerate(self.boxes_norm):
            if (b[0] >= sq[0] - 1e-9 and b[1] >= sq[1] - 1e-9
                    and b[2] <= sq[2] + 1e-9 and b[3] <= sq[3] + 1e-9):
                return i
        return None

    def make_stub_embedder(
        self,
        noise: float = 0.02,
        seed: int = 0,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """A scene-aware stub embedder for the miner (test double for DINO).

        Boxes fully inside a square get (class one-hot + small noise),
        L2-normalized — same-class boxes are near-duplicates (cos ~ 1),
        cross-class cos ~ 0. Boxes NOT fully inside any square each get a
        UNIQUE one-hot direction, exactly orthogonal to every class vector
        and to every other outside box (cos == 0 — no chance similarity, so
        analytic tests stay deterministic).

        Matches the miner's injectable-embedder signature:
        ``embed(image [H,W,3] float, boxes_xyxy_norm [N,4]) -> [N, D]``.
        """
        n_classes = max((sq.class_id for sq in self.squares), default=0) + 1
        rng = np.random.default_rng(seed)

        def embed(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
            del image  # the stub keys on geometry, not pixels
            n = len(boxes)
            dim = n_classes + n  # class slots + one unique slot per box
            out = np.zeros((n, dim), dtype=np.float32)
            for k, box in enumerate(np.asarray(boxes, dtype=np.float64)):
                idx = self.square_index_for_box(box)
                v = np.zeros(dim, dtype=np.float32)
                if idx is not None:
                    v[self.squares[idx].class_id] = 1.0
                    v[:n_classes] += noise * rng.standard_normal(n_classes).astype(
                        np.float32
                    )
                else:
                    v[n_classes + k] = 1.0
                out[k] = v / max(float(np.linalg.norm(v)), 1e-8)
            return out

        return embed


def _square_texture(
    side_px: int, class_id: int, texture_contrast: float, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic per-class texture: class-keyed sinusoid grid + noise.

    NOTE: the texture is generated in IMAGE space at the apparent resolution —
    no patch is ever rendered large and resized down, so the synthetic data
    cannot reintroduce the resampling-artifact channel the design bans.
    """
    yy, xx = np.mgrid[0:side_px, 0:side_px].astype(np.float32)
    freq = 2.0 + (class_id % 5)
    phase = 0.7 * (class_id % 7)
    base = 0.5 + 0.5 * np.sin(2 * np.pi * freq * xx / side_px + phase) \
        * np.sin(2 * np.pi * freq * yy / side_px + phase)
    color = np.array([
        0.35 + 0.5 * ((class_id * 37) % 11) / 11.0,
        0.35 + 0.5 * ((class_id * 53) % 13) / 13.0,
        0.35 + 0.5 * ((class_id * 71) % 17) / 17.0,
    ], dtype=np.float32)
    tex = color[None, None, :] * (0.6 + texture_contrast * base[:, :, None])
    tex += 0.02 * rng.standard_normal((side_px, side_px, 3)).astype(np.float32)
    return np.clip(tex, 0.0, 1.0)


def render_pinhole_scene(
    squares: Sequence[SyntheticSquare],
    h: int = 320,
    w: int = 320,
    focal_px: Optional[float] = None,
    background_z_m: float = 70.0,
    background_level: float = 0.5,
    background_noise: float = 0.005,
    texture_contrast: float = 0.5,
    seed: int = 0,
) -> SyntheticScene:
    """Render squares under an ideal pinhole camera + the exact 1/Z map.

    Apparent side of square i is ``round(focal_px * size_m / z_m)`` pixels;
    the inverse-depth map is ``1 / z_m`` inside square i and
    ``1 / background_z_m`` elsewhere — analytic ground truth with no
    rendering approximation beyond pixel rounding.

    The background is near-uniform (std ``background_noise`` << the texture
    gate) so background candidates fail the texture gate by construction.

    Args:
        squares: square specs; must not overlap (caller's responsibility for
            clean ground truth).
        h, w: image size in pixels.
        focal_px: focal length in pixels (default: ``w``).
        background_z_m: metric depth of the background plane.
        seed: texture rng seed.

    Returns:
        :class:`SyntheticScene`.
    """
    f = float(focal_px) if focal_px is not None else float(w)
    rng = np.random.default_rng(seed)
    image = np.full((h, w, 3), background_level, dtype=np.float32)
    image += background_noise * rng.standard_normal((h, w, 3)).astype(np.float32)
    inv_depth = np.full((h, w), 1.0 / background_z_m, dtype=np.float32)

    boxes = []
    for sq in squares:
        side = int(round(sq.apparent_side_px(f)))
        if side < 2:
            raise ValueError(
                f"square at z={sq.z_m} m, size={sq.size_m} m has apparent side "
                f"{side} px — too small to render"
            )
        x1 = int(round(sq.cx * w - side / 2.0))
        y1 = int(round(sq.cy * h - side / 2.0))
        x2, y2 = x1 + side, y1 + side
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(
                f"square at ({sq.cx}, {sq.cy}) with apparent side {side} px "
                f"falls outside the {w}x{h} image"
            )
        image[y1:y2, x1:x2] = _square_texture(side, sq.class_id, texture_contrast, rng)
        inv_depth[y1:y2, x1:x2] = 1.0 / sq.z_m
        boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])

    return SyntheticScene(
        image=np.clip(image, 0.0, 1.0),
        inv_depth=inv_depth,
        squares=list(squares),
        boxes_norm=np.asarray(boxes, dtype=np.float64).reshape(len(boxes), 4),
        focal_px=f,
        background_z_m=background_z_m,
        rng_seed=seed,
    )


def two_class_scene(
    depths_a: Tuple[float, float] = (5.0, 15.0),
    depths_b: Tuple[float, float] = (4.0, 20.0),
    h: int = 320,
    w: int = 320,
    seed: int = 0,
) -> SyntheticScene:
    """Canonical 4-square test scene: two content classes, two depths each.

    Class 0 at ``depths_a`` (ratio 3x), class 1 at ``depths_b`` (ratio 5x) —
    both inside the [1.5, 6] label band. Physical sizes are chosen so every
    apparent side is ~96 px on a 320 px canvas, comfortably containing the
    0.10/0.16/0.25 candidate grid levels.
    """
    f = float(w)
    target_px = 0.30 * min(h, w)
    sqs = []
    centers = [(0.30, 0.28), (0.72, 0.30), (0.30, 0.74), (0.72, 0.76)]
    for k, (cls, z) in enumerate([(0, depths_a[0]), (0, depths_a[1]),
                                  (1, depths_b[0]), (1, depths_b[1])]):
        size_m = target_px * z / f
        sqs.append(SyntheticSquare(z_m=z, size_m=size_m,
                                   cx=centers[k][0], cy=centers[k][1], class_id=cls))
    return render_pinhole_scene(sqs, h=h, w=w, focal_px=f, seed=seed)


def row_decorrelated_scene(
    depths: Tuple[float, float] = (5.0, 15.0),
    row: float = 0.5,
    h: int = 320,
    w: int = 320,
    seed: int = 1,
) -> SyntheticScene:
    """Same-row squares at DIFFERENT depths via different physical sizes.

    Both squares of one content class sit at the same image row (same center
    y) and the same apparent size, but at depths with a 3x ratio — the
    physical size varies to compensate. Any correct mined pair must carry
    ``log_r = log(z_a / z_b) != 0`` even though the row coordinate carries no
    depth information: labels provably derive from depth, not image row.
    """
    f = float(w)
    target_px = 0.30 * min(h, w)
    sqs = []
    for k, z in enumerate(depths):
        size_m = target_px * z / f
        cx = 0.28 + 0.45 * k
        sqs.append(SyntheticSquare(z_m=z, size_m=size_m, cx=cx, cy=row, class_id=0))
    return render_pinhole_scene(sqs, h=h, w=w, focal_px=f, seed=seed)


def materialize_scene(
    scene: SyntheticScene,
    image_dir,
    image_id: str,
    depth_cache=None,
) -> str:
    """Write the scene image as PNG (+ optionally its exact metric depth).

    The depth is written at HALF the image resolution (matching the shared
    cache convention) by 2x2 average pooling of the exact inverse-depth map.

    Args:
        image_dir: directory for the image PNG.
        image_id: id (also the PNG stem; slashes preserved).
        depth_cache: optional :class:`~.depth_io.DepthCache` to write the
            metric inverse depth into.

    Returns:
        The written image path (str).
    """
    from pathlib import Path

    from PIL import Image

    img_path = Path(image_dir) / f"{image_id}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scene.image * 255.0 + 0.5).astype(np.uint8)).save(img_path)

    if depth_cache is not None:
        inv = scene.inv_depth
        h2, w2 = (inv.shape[0] // 2) * 2, (inv.shape[1] // 2) * 2
        half = inv[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
        depth_cache.save(image_id, half.astype(np.float32))
    return str(img_path)
