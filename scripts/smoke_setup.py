"""Smoke environment setup — reproducible scaffold for the Faz 5 smoke campaign.

A Colab runtime reset wipes ``/content``, taking the smoke scaffold with it:
the opened SSL pool, the balanced image subset, and the Faz 5.1 SAPS
checkpoint that later phases consume as the SSL teacher. This script rebuilds
all three from their durable sources (the per-dataset tars on Drive + the
repo code) so a reset becomes a single-command recovery.

Idempotent: each piece is rebuilt only if missing. ``--force`` rebuilds all.

Usage (Colab):
    !python scripts/smoke_setup.py
    !python scripts/smoke_setup.py --force          # rebuild everything
    !python scripts/smoke_setup.py --subset-size 600

Pieces, in dependency order:
    1. SSL pool         — extract 4 part-tars from Drive into /content
    2. smoke subset     — balanced N-image draw from the pool manifest
    3. SAPS checkpoint  — short SAPS pretrain; the Faz 5.3 SSL teacher
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time


# ── default paths ───────────────────────────────────────────────────────
DRIVE_PARTS = "/content/drive/MyDrive/SSL_POOL_PARTS"
POOL_DIR = "/content/ssl_pool_v1"
SMOKE_DIR = "/content/saps_smoke_imgs"
SAPS_CKPT = "/content/saps_smoke_none.pt"

POOL_DATASETS = ("bdd100k", "cityscapes", "mapillary", "a2d2")
EXPECTED_POOL_SIZE = 181446


def setup_pool(drive_parts: str, pool_dir: str, force: bool) -> None:
    """Extract the 4 per-dataset part-tars from Drive into the pool dir."""
    manifest = os.path.join(pool_dir, "manifest.parquet")
    if os.path.exists(manifest) and not force:
        print(f"  [pool] zaten acik: {pool_dir}")
        return

    if force and os.path.exists(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir, exist_ok=True)

    if not os.path.isdir(drive_parts):
        raise FileNotFoundError(
            f"Drive parts dizini yok: {drive_parts} -- Drive mount'lu mu?"
        )

    print(f"  [pool] aciliyor: {drive_parts} -> {pool_dir}")
    t0 = time.time()
    for ds in POOL_DATASETS:
        tar = os.path.join(drive_parts, f"{ds}.tar")
        if not os.path.exists(tar):
            raise FileNotFoundError(f"part-tar yok: {tar}")
        subprocess.run(["tar", "-xf", tar, "-C", pool_dir], check=True)
    shutil.copy(os.path.join(drive_parts, "manifest.parquet"), manifest)
    print(f"  [pool] acildi ({(time.time() - t0) / 60:.1f} dk)")


def setup_subset(pool_dir: str, smoke_dir: str, size: int, force: bool) -> None:
    """Draw a balanced N-image subset from the pool manifest."""
    import pandas as pd

    if (os.path.isdir(smoke_dir)
            and len(os.listdir(smoke_dir)) >= size and not force):
        print(f"  [subset] zaten var: {smoke_dir} "
              f"({len(os.listdir(smoke_dir))} goruntu)")
        return

    shutil.rmtree(smoke_dir, ignore_errors=True)
    os.makedirs(smoke_dir)

    df = pd.read_parquet(os.path.join(pool_dir, "manifest.parquet"))
    per_ds = max(1, size // df["dataset"].nunique())
    # deterministic draw (random_state=0) -> reproducible smoke subset
    sample = df.groupby("dataset", group_keys=False).apply(
        lambda g: g.sample(min(per_ds, len(g)), random_state=0)
    )
    n = 0
    for _, row in sample.iterrows():
        src = os.path.join(pool_dir, row["materialized_path"])
        if os.path.exists(src):
            shutil.copy(src, os.path.join(smoke_dir, f"img_{n:04d}.jpg"))
            n += 1
    print(f"  [subset] olusturuldu: {smoke_dir} ({n} goruntu)")


def setup_saps_checkpoint(
    smoke_dir: str, ckpt: str, epochs: int, force: bool
) -> None:
    """Short SAPS pretrain — the Faz 5.3 SSL teacher checkpoint."""
    if os.path.exists(ckpt) and not force:
        print(f"  [saps] checkpoint zaten var: {ckpt}")
        return

    print(f"  [saps] checkpoint uretiliyor ({epochs} epoch)...")
    from yolo_contrastive.pretrain import DenseSSLPretrainer

    trainer = DenseSSLPretrainer(
        model="yolov8n.pt", saps_mode="none",
        out_dim=128, queue_size=4096, n_query=64,
        imgsz=320, device="cuda",
    )
    try:
        trainer.train(
            images_dir=smoke_dir, epochs=epochs, batch_size=16,
            warmup_epochs=2, num_workers=2, output=ckpt,
            save_every=0, print_every=4,
        )
    finally:
        trainer.cleanup()
    print(f"  [saps] checkpoint hazir: {ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faz 5 smoke ortamini kur (havuz + alt-kume + SAPS ckpt)."
    )
    parser.add_argument("--drive-parts", default=DRIVE_PARTS,
                        help="Drive'daki per-dataset part-tar dizini")
    parser.add_argument("--pool-dir", default=POOL_DIR,
                        help="havuzun acilacagi dizin")
    parser.add_argument("--smoke-dir", default=SMOKE_DIR,
                        help="smoke alt-kume goruntu dizini")
    parser.add_argument("--saps-ckpt", default=SAPS_CKPT,
                        help="SAPS checkpoint cikti yolu")
    parser.add_argument("--subset-size", type=int, default=600,
                        help="smoke alt-kume goruntu sayisi")
    parser.add_argument("--saps-epochs", type=int, default=8,
                        help="SAPS checkpoint pretrain epoch sayisi")
    parser.add_argument("--force", action="store_true",
                        help="her parcayi mevcut olsa da yeniden uret")
    parser.add_argument("--skip-saps", action="store_true",
                        help="SAPS checkpoint adimini atla (GPU gerektirir)")
    args = parser.parse_args()

    print("=" * 60)
    print("SMOKE ORTAMI KURULUMU")
    print("=" * 60)

    setup_pool(args.drive_parts, args.pool_dir, args.force)
    setup_subset(args.pool_dir, args.smoke_dir, args.subset_size, args.force)
    if args.skip_saps:
        print("  [saps] atlandi (--skip-saps)")
    else:
        setup_saps_checkpoint(
            args.smoke_dir, args.saps_ckpt, args.saps_epochs, args.force
        )

    # ── summary ──
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    pool_ok = os.path.exists(os.path.join(args.pool_dir, "manifest.parquet"))
    subset_n = (len(os.listdir(args.smoke_dir))
                if os.path.isdir(args.smoke_dir) else 0)
    saps_ok = os.path.exists(args.saps_ckpt)
    print(f"  havuz      : {'OK' if pool_ok else 'YOK'}  {args.pool_dir}")
    print(f"  smoke imgs : {subset_n} goruntu  {args.smoke_dir}")
    print(f"  SAPS ckpt  : {'OK' if saps_ok else 'YOK/atlandi'}  {args.saps_ckpt}")
    if pool_ok and subset_n > 0 and (saps_ok or args.skip_saps):
        print("\n  smoke ortami HAZIR")
    else:
        print("\n  ! eksik parca var -- yukaridaki ciktilari incele")


if __name__ == "__main__":
    main()
