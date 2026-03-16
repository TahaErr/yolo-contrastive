"""Test SSLPretrainer init modes."""

import pytest
from yolo_contrastive.pretrain import SSLPretrainer
from yolo_contrastive.pretext import CompositeTask


class TestSSLPretrainerInit:
    def test_composite(self):
        pt = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                           lambda_cl=1.0, pretext_tasks=["freq_band", "solarization"],
                           pretext_weights=[1.0, 0.8], lambda_pretext=0.5, imgsz=64)
        assert isinstance(pt.pretext_task, CompositeTask)
        assert pt._has_pretext
        pt.cleanup()

    def test_legacy_rotation(self):
        pt = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                           lambda_cl=1.0, lambda_rot=0.3, imgsz=64)
        assert pt.rot_task is not None
        assert pt.pretext_task is None
        pt.cleanup()

    def test_cl_only(self):
        pt = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                           lambda_cl=1.0, lambda_rot=0.0, imgsz=64)
        assert not pt._has_pretext
        pt.cleanup()

    def test_freq_gated_adapter(self):
        pt = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                           lambda_cl=1.0, pretext_tasks=["freq_band"],
                           lambda_pretext=0.5, adapter="freq_gated",
                           adapter_rank=4, imgsz=64)
        assert pt._adapter_info is not None
        assert pt._adapter_info["injected"] > 0
        pt.cleanup()

    def test_task_routed_adapter(self):
        pt = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                           lambda_cl=1.0,
                           pretext_tasks=["freq_band", "solarization", "patch_shuffle"],
                           pretext_weights=[1.0, 0.8, 0.5], lambda_pretext=0.5,
                           adapter="task_routed", adapter_rank=4, imgsz=64)
        assert pt._adapter_info is not None
        assert pt._task_router is not None
        assert pt._task_router.num_tasks == 3
        pt.cleanup()


@pytest.mark.slow
def test_train_1epoch(dummy_images):
    pt = SSLPretrainer(model="yolov8n.pt", aug_preset="simclr_v2",
                       lambda_cl=1.0, pretext_tasks=["solarization", "blur"],
                       pretext_weights=[1.0, 0.5], lambda_pretext=0.3, imgsz=64)
    import os
    out = os.path.join(dummy_images, "backbone.pt")
    result = pt.train(images_dir=dummy_images, epochs=1, batch_size=4,
                      lr=1e-3, warmup_epochs=0, num_workers=0,
                      output=out, save_every=0, print_every=1)
    assert result == out and os.path.exists(out)
    pt.cleanup()
