"""Unified pipeline — dataset yapısına göre otomatik eğitim.

Kullanım:
    from yolo_contrastive.pipeline import auto_train

    # Otomatik mod algılama
    results = auto_train(data_yaml="path/to/data.yaml")

    # Veya manuel pipeline
    from yolo_contrastive.pipeline import SSLFinetunePipeline
    pipeline = SSLFinetunePipeline(config=PipelineConfig(...))
    results = pipeline.run(data_yaml="...", images_dir="...")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .discovery import discover, DatasetInfo, TrainMode
from .exceptions import ConfigError


def _log(msg: str):
    try:
        from ultralytics.utils import LOGGER
        LOGGER.info(msg)
    except Exception:
        print(msg)


@dataclass
class PipelineConfig:
    """Tüm pipeline konfigürasyonu."""

    # Model
    model: str = "yolov8n.pt"
    imgsz: int = 640
    device: Optional[str] = None

    # SSL Pretraining
    ssl_epochs: int = 100
    ssl_batch: int = 32
    ssl_lr: float = 1e-3
    ssl_aug_preset: str = "simclr_v2"
    ssl_lambda_cl: float = 1.0
    ssl_lambda_rot: float = 0.5
    ssl_temperature: float = 0.2
    ssl_warmup_epochs: int = 5
    ssl_num_workers: int = 4
    ssl_save_every: int = 25
    ssl_print_every: int = 10

    # Fine-tuning / Detection
    ft_epochs: int = 50
    ft_batch: int = 16
    ft_freeze_layers: int = 10
    ft_unfreeze_epoch: int = 0
    ft_backbone_lr_scale: float = 0.1

    # Contrastive (detection modunda opsiyonel)
    cl_lambda: float = 0.0
    cl_temperature: float = 0.2
    cl_two_view: bool = False
    cl_aug_preset: str = ""
    cl_lambda_rot: float = 0.0

    # Output
    backbone_path: str = "pretrained_backbone.pt"
    project: str = "runs/pipeline"
    name: str = "exp"

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


class SSLFinetunePipeline:
    """Uçtan uca eğitim pipeline'ı.

    Dataset yapısına göre otomatik mod:
        SSL_FINETUNE: unlabeled + labeled → pretrain → finetune
        DETECTION:    labeled only → detection training
        SSL_ONLY:     unlabeled only → backbone pretrain

    Kullanım:
        pipeline = SSLFinetunePipeline(config=PipelineConfig(ssl_epochs=10))
        results = pipeline.run(data_yaml="data.yaml")
    """

    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = PipelineConfig.from_dict(kwargs)

        self.dataset_info: Optional[DatasetInfo] = None
        self.backbone_path: Optional[str] = None
        self.ssl_time: float = 0.0
        self.ft_time: float = 0.0
        self.results = None

    def discover_dataset(
        self,
        data_yaml: Optional[str] = None,
        unlabeled_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ) -> DatasetInfo:
        """Dataset yapısını algıla."""
        self.dataset_info = discover(
            data_yaml=data_yaml,
            unlabeled_dir=unlabeled_dir,
            dataset_dir=dataset_dir,
        )
        _log("\n📂 Dataset Discovery:")
        _log(self.dataset_info.summary())
        return self.dataset_info

    def run_ssl(
        self,
        images_dir: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        """Aşama: SSL Pretraining."""
        images_dir = images_dir or (self.dataset_info.unlabeled_dir if self.dataset_info else None)
        if not images_dir:
            raise ConfigError("Etiketsiz görüntü klasörü bulunamadı")

        output = output or self.cfg.backbone_path

        _log("\n" + "=" * 60)
        _log("🔬 SSL Pretraining")
        _log("=" * 60)

        from .pretrain import SSLPretrainer

        pretrainer = SSLPretrainer(
            model=self.cfg.model,
            aug_preset=self.cfg.ssl_aug_preset,
            lambda_cl=self.cfg.ssl_lambda_cl,
            lambda_rot=self.cfg.ssl_lambda_rot,
            temperature=self.cfg.ssl_temperature,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
        )

        t0 = time.time()
        self.backbone_path = pretrainer.train(
            images_dir=images_dir,
            epochs=self.cfg.ssl_epochs,
            batch_size=self.cfg.ssl_batch,
            lr=self.cfg.ssl_lr,
            warmup_epochs=self.cfg.ssl_warmup_epochs,
            num_workers=self.cfg.ssl_num_workers,
            output=output,
            save_every=self.cfg.ssl_save_every,
            print_every=self.cfg.ssl_print_every,
        )
        self.ssl_time = time.time() - t0

        _log(f"✅ SSL tamamlandı: {self.ssl_time:.1f}s → {self.backbone_path}")
        return self.backbone_path

    def run_finetune(
        self,
        data_yaml: Optional[str] = None,
        backbone_path: Optional[str] = None,
    ):
        """Aşama: Fine-tune with pretrained backbone."""
        data_yaml = data_yaml or (self.dataset_info.data_yaml if self.dataset_info else None)
        if not data_yaml:
            raise ConfigError("data.yaml bulunamadı")

        backbone_path = backbone_path or self.backbone_path
        if not backbone_path or not os.path.exists(backbone_path):
            raise FileNotFoundError(
                f"Pretrained backbone bulunamadı: {backbone_path}. "
                f"Önce run_ssl() çalıştırın."
            )

        _log("\n" + "=" * 60)
        _log("🎯 Fine-tuning (pretrained backbone)")
        _log("=" * 60)

        os.environ["YCL_PRETRAINED"] = str(backbone_path)
        os.environ["YCL_FREEZE_BACKBONE"] = str(self.cfg.ft_freeze_layers)
        os.environ["YCL_UNFREEZE_EPOCH"] = str(self.cfg.ft_unfreeze_epoch)
        os.environ["YCL_BACKBONE_LR_SCALE"] = str(self.cfg.ft_backbone_lr_scale)

        from ultralytics import YOLO
        from .finetune import FinetuneDetectionTrainer

        device = self.cfg.device
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"

        model = YOLO(self.cfg.model)
        t0 = time.time()
        self.results = model.train(
            data=data_yaml,
            epochs=self.cfg.ft_epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.ft_batch,
            device=device,
            trainer=FinetuneDetectionTrainer,
            project=self.cfg.project,
            name=self.cfg.name + "_finetune",
            exist_ok=True,
        )
        self.ft_time = time.time() - t0

        _log(f"✅ Fine-tune tamamlandı: {self.ft_time:.1f}s")
        return self.results

    def run_detection(
        self,
        data_yaml: Optional[str] = None,
        use_contrastive: bool = False,
    ):
        """Aşama: Direkt detection eğitimi (SSL yok)."""
        data_yaml = data_yaml or (self.dataset_info.data_yaml if self.dataset_info else None)
        if not data_yaml:
            raise ConfigError("data.yaml bulunamadı")

        _log("\n" + "=" * 60)
        trainer_name = "Contrastive Detection" if use_contrastive else "Base Detection"
        _log(f"🎯 {trainer_name} Training")
        _log("=" * 60)

        from ultralytics import YOLO

        device = self.cfg.device
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"

        train_kwargs = dict(
            data=data_yaml,
            epochs=self.cfg.ft_epochs,
            imgsz=self.cfg.imgsz,
            batch=self.cfg.ft_batch,
            device=device,
            project=self.cfg.project,
            name=self.cfg.name + "_detection",
            exist_ok=True,
        )

        if use_contrastive:
            os.environ["YCL_LAMBDA"] = str(self.cfg.cl_lambda)
            os.environ["YCL_TEMP"] = str(self.cfg.cl_temperature)
            os.environ["YCL_TWO_VIEW"] = "1" if self.cfg.cl_two_view else "0"
            os.environ["YCL_AUG_PRESET"] = self.cfg.cl_aug_preset
            os.environ["YCL_LAMBDA_ROT"] = str(self.cfg.cl_lambda_rot)

            from .trainer import ContrastiveDetectionTrainer
            train_kwargs["trainer"] = ContrastiveDetectionTrainer

        model = YOLO(self.cfg.model)
        t0 = time.time()
        self.results = model.train(**train_kwargs)
        self.ft_time = time.time() - t0

        _log(f"✅ {trainer_name} tamamlandı: {self.ft_time:.1f}s")
        return self.results

    def run(
        self,
        data_yaml: Optional[str] = None,
        unlabeled_dir: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ):
        """Otomatik pipeline — dataset yapısına göre mod seçer.

        Args:
            data_yaml: YOLO data.yaml yolu
            unlabeled_dir: Etiketsiz görüntü klasörü
            dataset_dir: Üst dataset klasörü

        Returns:
            Training results
        """
        # 1) Dataset discovery
        info = self.discover_dataset(
            data_yaml=data_yaml,
            unlabeled_dir=unlabeled_dir,
            dataset_dir=dataset_dir,
        )

        _log(f"\n🚀 Seçilen mod: {info.mode.value}")

        # 2) Moda göre çalıştır
        if info.mode == TrainMode.SSL_FINETUNE:
            self.run_ssl(images_dir=info.unlabeled_dir)
            return self.run_finetune(data_yaml=info.data_yaml)

        elif info.mode == TrainMode.DETECTION:
            use_cl = self.cfg.cl_lambda > 0
            return self.run_detection(data_yaml=info.data_yaml, use_contrastive=use_cl)

        elif info.mode == TrainMode.SSL_ONLY:
            self.run_ssl(images_dir=info.unlabeled_dir)
            _log("\n📦 Sadece backbone pretrain yapıldı (etiketli veri yok).")
            _log(f"💾 Backbone: {self.backbone_path}")
            return self.backbone_path

    def summary(self) -> dict:
        """Pipeline sonuç özeti."""
        return {
            "mode": self.dataset_info.mode.value if self.dataset_info else None,
            "backbone_path": self.backbone_path,
            "ssl_time_sec": self.ssl_time,
            "ft_time_sec": self.ft_time,
            "total_time_sec": self.ssl_time + self.ft_time,
            "dataset": self.dataset_info.summary() if self.dataset_info else None,
            "results": self.results,
        }


def auto_train(
    data_yaml: Optional[str] = None,
    unlabeled_dir: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    **kwargs,
):
    """Tek fonksiyonla otomatik eğitim.

    Dataset yapısını algılar ve uygun pipeline'ı çalıştırır:
        - unlabeled + labeled → SSL Pretrain → Fine-tune
        - labeled only → Detection training
        - unlabeled only → Backbone pretrain

    Args:
        data_yaml: YOLO data.yaml yolu
        unlabeled_dir: Etiketsiz görüntü klasörü
        dataset_dir: Üst dataset klasörü
        **kwargs: PipelineConfig parametreleri

    Returns:
        Training results

    Kullanım:
        # En basit — otomatik algılama
        results = auto_train(dataset_dir="path/to/dataset")

        # Açık yollarla
        results = auto_train(
            data_yaml="path/to/data.yaml",
            unlabeled_dir="path/to/unlabeled",
            ssl_epochs=50,
            ft_epochs=30,
        )
    """
    config = PipelineConfig.from_dict(kwargs)
    pipeline = SSLFinetunePipeline(config=config)

    results = pipeline.run(
        data_yaml=data_yaml,
        unlabeled_dir=unlabeled_dir,
        dataset_dir=dataset_dir,
    )

    # Özet yazdır
    s = pipeline.summary()
    _log("\n" + "=" * 60)
    _log("📊 Pipeline Sonuç")
    _log("=" * 60)
    _log(f"Mod:     {s['mode']}")
    _log(f"SSL:     {s['ssl_time_sec']:.1f}s")
    _log(f"Det/FT:  {s['ft_time_sec']:.1f}s")
    _log(f"Toplam:  {s['total_time_sec']:.1f}s")
    _log("=" * 60)

    return results
