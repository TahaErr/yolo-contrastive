"""Basic import tests."""


def test_import_version():
    import yolo_contrastive
    assert hasattr(yolo_contrastive, "__version__")
    assert isinstance(yolo_contrastive.__version__, str)


def test_convenience_imports():
    """Audit §4.1: public API should be importable from top-level."""
    from yolo_contrastive import NTXentLoss, build_contrastive_loss, FeatureTap

    assert NTXentLoss is not None
    assert callable(build_contrastive_loss)
    assert FeatureTap is not None
