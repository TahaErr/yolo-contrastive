"""Tests for SSL-pool common helpers.

Download tests use ``unittest.mock`` to fake ``requests.get`` — we never make
real HTTP calls in the test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from yolo_contrastive.data.ssl_pool.common import (
    download_with_resume,
    is_readable_image,
    resize_and_save,
    resize_long_side,
    to_rgb,
)


def _make_image(path: Path, size, mode: str = "RGB", color=(128, 64, 32)) -> Path:
    if mode != "RGB":
        # Pillow rejects RGB tuples for non-RGB modes
        color = 128 if mode == "L" else color
    img = Image.new(mode, size, color=color)
    img.save(path)
    return path


class TestResizeLongSide:
    def test_downsizes_landscape(self):
        img = Image.new("RGB", (1280, 720))
        out = resize_long_side(img, 640)
        assert out.size == (640, 360)

    def test_downsizes_portrait(self):
        img = Image.new("RGB", (720, 1280))
        out = resize_long_side(img, 640)
        assert out.size == (360, 640)

    def test_no_upscale_when_smaller(self):
        img = Image.new("RGB", (300, 200))
        out = resize_long_side(img, 640)
        assert out.size == (300, 200)

    def test_exact_long_side_passthrough(self):
        img = Image.new("RGB", (640, 480))
        out = resize_long_side(img, 640)
        assert out.size == (640, 480)

    def test_extreme_aspect_ratio(self):
        img = Image.new("RGB", (3200, 100))
        out = resize_long_side(img, 640)
        assert out.size[0] == 640
        # Short side must remain >= 1
        assert out.size[1] >= 1


class TestToRgb:
    def test_rgb_passthrough_returns_same_object(self):
        img = Image.new("RGB", (10, 10))
        assert to_rgb(img) is img

    def test_grayscale_converted(self):
        img = Image.new("L", (10, 10), color=128)
        out = to_rgb(img)
        assert out.mode == "RGB"
        # All channels equal for a uniform grayscale source
        assert out.getpixel((0, 0)) == (128, 128, 128)

    def test_rgba_composited_over_white(self):
        # Fully transparent red — should composite to white
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
        out = to_rgb(img)
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_cmyk_converted(self):
        img = Image.new("CMYK", (10, 10))
        assert to_rgb(img).mode == "RGB"


class TestResizeAndSave:
    def test_basic_landscape_resize(self, tmp_path):
        src = _make_image(tmp_path / "src.png", (1280, 720))
        dest = tmp_path / "out" / "dest.jpg"
        orig, mat, sha = resize_and_save(src, dest)
        assert orig == (1280, 720)
        assert mat == (640, 360)
        assert dest.exists()
        assert len(sha) == 64
        # Confirm the saved file is readable and matches reported dims
        with Image.open(dest) as out:
            assert out.size == (640, 360)
            assert out.mode == "RGB"

    def test_no_upscale_for_small_input(self, tmp_path):
        src = _make_image(tmp_path / "src.png", (300, 200))
        dest = tmp_path / "dest.jpg"
        orig, mat, _ = resize_and_save(src, dest)
        assert orig == (300, 200)
        assert mat == (300, 200)

    def test_grayscale_input_saved_as_rgb(self, tmp_path):
        src = _make_image(tmp_path / "src.png", (800, 600), mode="L")
        dest = tmp_path / "dest.jpg"
        resize_and_save(src, dest)
        with Image.open(dest) as out:
            assert out.mode == "RGB"

    def test_creates_nested_destination_dir(self, tmp_path):
        src = _make_image(tmp_path / "src.png", (640, 480))
        dest = tmp_path / "a" / "b" / "c" / "dest.jpg"
        resize_and_save(src, dest)
        assert dest.exists()

    def test_hash_deterministic_across_runs(self, tmp_path):
        src = _make_image(tmp_path / "src.png", (640, 480))
        _, _, h1 = resize_and_save(src, tmp_path / "d1.jpg")
        _, _, h2 = resize_and_save(src, tmp_path / "d2.jpg")
        assert h1 == h2

    def test_hash_differs_for_different_inputs(self, tmp_path):
        # Two images with distinct content must produce distinct hashes
        a = _make_image(tmp_path / "a.png", (640, 480), color=(10, 20, 30))
        b = _make_image(tmp_path / "b.png", (640, 480), color=(200, 100, 50))
        _, _, ha = resize_and_save(a, tmp_path / "da.jpg")
        _, _, hb = resize_and_save(b, tmp_path / "db.jpg")
        assert ha != hb

    def test_custom_long_side(self, tmp_path):
        src = _make_image(tmp_path / "src.png", (1024, 1024))
        dest = tmp_path / "dest.jpg"
        _, mat, _ = resize_and_save(src, dest, long_side=256)
        assert mat == (256, 256)

    def test_unreadable_source_raises(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"definitely not a jpeg")
        with pytest.raises(Exception):
            resize_and_save(bad, tmp_path / "dest.jpg")

    def test_accepts_bytesio_source(self, tmp_path):
        # Adapters that stream archive entries pass io.BytesIO instead of a path
        import io as _io

        src_img = Image.new("RGB", (800, 600), color=(40, 80, 120))
        buf = _io.BytesIO()
        src_img.save(buf, format="JPEG", quality=92)
        buf.seek(0)

        dest = tmp_path / "dest.jpg"
        orig, mat, sha = resize_and_save(buf, dest)
        assert orig == (800, 600)
        assert mat == (640, 480)
        assert dest.exists()
        assert len(sha) == 64


class TestIsReadableImage:
    def test_valid_png(self, tmp_path):
        p = _make_image(tmp_path / "x.png", (10, 10))
        assert is_readable_image(p) is True

    def test_valid_jpeg(self, tmp_path):
        p = tmp_path / "x.jpg"
        Image.new("RGB", (10, 10)).save(p, "JPEG")
        assert is_readable_image(p) is True

    def test_corrupted_bytes(self, tmp_path):
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"not an image at all")
        assert is_readable_image(p) is False

    def test_missing_file(self, tmp_path):
        assert is_readable_image(tmp_path / "nope.jpg") is False

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jpg"
        p.write_bytes(b"")
        assert is_readable_image(p) is False


class TestDownloadWithResume:
    def _mock_response(self, status_code: int = 200, chunks=(b"hello world",)):
        resp = MagicMock()
        resp.status_code = status_code
        resp.iter_content = MagicMock(return_value=iter(chunks))
        resp.raise_for_status = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_fresh_download(self, tmp_path):
        dest = tmp_path / "out.bin"
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(chunks=(b"abc", b"def"))
            n = download_with_resume("http://example.com/file", dest)
        assert n == 6
        assert dest.read_bytes() == b"abcdef"
        # No Range header on a fresh download
        _, kwargs = mock_get.call_args
        assert "Range" not in kwargs.get("headers", {})

    def test_resume_sends_range_header(self, tmp_path):
        dest = tmp_path / "out.bin"
        dest.write_bytes(b"already-")  # 8 bytes already on disk
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(chunks=(b"there",))
            n = download_with_resume("http://example.com/file", dest)
        assert n == 13  # 8 + 5
        assert dest.read_bytes() == b"already-there"
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Range"] == "bytes=8-"

    def test_416_treated_as_complete(self, tmp_path):
        dest = tmp_path / "out.bin"
        dest.write_bytes(b"complete")
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(status_code=416)
            n = download_with_resume("http://example.com/file", dest)
        assert n == 8
        assert dest.read_bytes() == b"complete"

    def test_size_mismatch_raises(self, tmp_path):
        dest = tmp_path / "out.bin"
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(chunks=(b"abc",))
            with pytest.raises(IOError, match="size mismatch"):
                download_with_resume(
                    "http://example.com/file", dest, expected_size=999
                )

    def test_size_match_passes(self, tmp_path):
        dest = tmp_path / "out.bin"
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(chunks=(b"hello",))
            n = download_with_resume(
                "http://example.com/file", dest, expected_size=5
            )
        assert n == 5

    def test_creates_parent_dir(self, tmp_path):
        dest = tmp_path / "deep" / "nested" / "out.bin"
        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_response(chunks=(b"x",))
            download_with_resume("http://example.com/file", dest)
        assert dest.exists()
