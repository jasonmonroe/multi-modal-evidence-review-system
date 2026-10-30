# code/tests/src/test_image_handler.py

import base64

import pytest
from PIL import Image

from src.image_handler import ImageHandler


@pytest.fixture
def img_dir(tmp_path):
    case = tmp_path / "images" / "case_001"
    case.mkdir(parents=True)
    Image.new("RGB", (8, 4), "red").save(case / "img_1.jpg")
    Image.new("RGBA", (5, 5), "blue").save(case / "img_2.png")
    Image.new("L", (3, 7), 10).save(case / "gray.png")
    (case / "blob.unknownext").write_bytes(b"\x00\x01\x02")
    return tmp_path


@pytest.fixture
def handler(img_dir):
    h = ImageHandler(eda_flag=False)
    h._base_dir = str(img_dir) + "/"
    return h


class TestInit:
    def test_defaults(self):
        h = ImageHandler(True)
        assert h.filenames == [] and h.images == []
        assert h._eda_flag is True
        assert h._base_dir.endswith("/")

    def test_eda_flag_false(self):
        assert ImageHandler(False)._eda_flag is False


class TestEncode:
    def test_returns_parallel_lists(self, handler):
        images, names = handler.encode(
            ["images/case_001/img_1.jpg", "images/case_001/img_2.png"]
        )
        assert isinstance(images, list) and isinstance(names, list)
        assert names == ["img_1.jpg", "img_2.png"]
        assert len(images) == len(names) == 2

    def test_data_url_mime_and_roundtrip(self, handler, img_dir):
        images, _ = handler.encode(["images/case_001/img_1.jpg"])
        prefix = "data:image/jpeg;base64,"
        assert images[0].startswith(prefix)
        raw = base64.b64decode(images[0][len(prefix):])
        assert raw == (img_dir / "images/case_001/img_1.jpg").read_bytes()

    def test_png_mime(self, handler):
        images, _ = handler.encode(["images/case_001/img_2.png"])
        assert images[0].startswith("data:image/png;base64,")

    def test_unknown_extension_defaults_to_jpeg(self, handler):
        images, names = handler.encode(["images/case_001/blob.unknownext"])
        assert images[0].startswith("data:image/jpeg;base64,")
        assert names == ["blob.unknownext"]

    def test_missing_file_skipped_with_warning(self, handler, capsys):
        images, names = handler.encode(
            ["images/case_001/nope.jpg", "images/case_001/img_1.jpg"]
        )
        assert names == ["img_1.jpg"]
        assert len(images) == 1
        assert "not found" in capsys.readouterr().out

    def test_all_missing_returns_empty(self, handler):
        assert handler.encode(["x.jpg", "y.jpg"]) == ([], [])

    def test_empty_input(self, handler):
        assert handler.encode([]) == ([], [])

    def test_path_is_normalised(self, handler):
        _, names = handler.encode(["images/../images/case_001//img_1.jpg"])
        assert names == ["img_1.jpg"]

    def test_eda_flag_triggers_describe_for_each_path(self, handler, monkeypatch):
        handler._eda_flag = True
        seen = []
        monkeypatch.setattr(handler, "describe", lambda p: seen.append(p))
        handler.encode(["images/case_001/img_1.jpg", "missing.jpg"])
        assert seen == ["images/case_001/img_1.jpg", "missing.jpg"]

    def test_no_describe_when_eda_off(self, handler, monkeypatch):
        seen = []
        monkeypatch.setattr(handler, "describe", lambda p: seen.append(p))
        handler.encode(["images/case_001/img_1.jpg"])
        assert seen == []


class TestLoad:
    def test_sets_images_and_filenames(self, handler):
        handler.load(["images/case_001/img_1.jpg"])
        assert handler.filenames == ["img_1.jpg"]
        assert len(handler.images) == 1

    def test_load_replaces_previous_state(self, handler):
        handler.load(["images/case_001/img_1.jpg", "images/case_001/img_2.png"])
        handler.load(["images/case_001/img_2.png"])
        assert handler.filenames == ["img_2.png"]

    @pytest.mark.parametrize("empty", [[], None])
    def test_empty_input_keeps_previous_state(self, handler, empty):
        handler.load(["images/case_001/img_1.jpg"])
        handler.load(empty)
        assert handler.filenames == ["img_1.jpg"]


class TestDescribe:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("images/case_001/img_1.jpg", (8, 4, 3)),
            ("images/case_001/img_2.png", (5, 5, 4)),
            ("images/case_001/gray.png", (3, 7, 1)),
        ],
    )
    def test_dimensions_and_channels(self, handler, path, expected):
        result = handler.describe(path)
        assert result == expected
        assert all(isinstance(v, int) for v in result)

    def test_prints_summary(self, handler, capsys):
        handler.describe("images/case_001/img_1.jpg")
        out = capsys.readouterr().out
        assert "Width: 8, Height: 4, Channels: 3" in out
        assert "img_1.jpg" in out

    def test_missing_file_returns_zeros(self, handler, capsys):
        assert handler.describe("nope.jpg") == (0, 0, 0)
        assert "not found" in capsys.readouterr().out

    def test_corrupt_image_raises(self, handler):
        from PIL import UnidentifiedImageError

        with pytest.raises(UnidentifiedImageError):
            handler.describe("images/case_001/blob.unknownext")
