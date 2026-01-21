"""
Tests for the napari-ome-arrow reader.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import napari_ome_arrow._reader as reader_mod
from napari_ome_arrow import napari_get_reader

DATA_ROOT = Path("tests/data").resolve()


def _p(*parts: str) -> str:
    """Build a test data path under tests/data."""
    return str(DATA_ROOT.joinpath(*parts))


# --------------------------------------------------------------------- #
#  Small helper: temporary env var, no monkeypatch
# --------------------------------------------------------------------- #


@contextlib.contextmanager
def temporary_env_var(key: str, value: str | None):
    """
    Temporarily set an environment variable.

    No pytest monkeypatch; we mutate os.environ directly and restore it.
    """
    old = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# --------------------------------------------------------------------- #
#  Basic reader dispatch
# --------------------------------------------------------------------- #


def test_get_reader_returns_none_for_unsupported_extension():
    """Reader should decline unsupported paths."""
    reader = napari_get_reader("fake.file")
    assert reader is None


# --------------------------------------------------------------------- #
#  Image mode: OME-Arrow sources
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path, expect_multi_channel",
    [
        # 2D single-channel
        (_p("ome-artificial-5d-datasets", "single-channel.ome.tiff"), False),
        # 2D multi-channel
        (_p("ome-artificial-5d-datasets", "multi-channel.ome.tiff"), True),
        # 3D z-stack
        (_p("ome-artificial-5d-datasets", "z-series.ome.tiff"), False),
        # ExampleHuman TIFF
        (_p("examplehuman", "AS_09125_050116030001_D03f00d0.tif"), False),
    ],
)
def test_reader_image_mode_ome_sources(path: str, expect_multi_channel: bool):
    """
    In image mode, OME-Arrow-backed sources should yield image layers
    with appropriate channel_axis settings.
    """
    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "image"):
        reader = napari_get_reader(path)
        assert callable(reader), (
            f"napari_get_reader did not return callable for {path}"
        )

        layer_data_list = reader(path)
        assert isinstance(layer_data_list, list)
        assert len(layer_data_list) == 1

        data, add_kwargs, layer_type = layer_data_list[0]

    assert layer_type == "image"
    assert isinstance(data, np.ndarray)
    assert data.ndim >= 2  # at least (Y, X)

    if expect_multi_channel:
        # Multi-channel inputs should expose a channel_axis
        assert "channel_axis" in add_kwargs
        axis = add_kwargs["channel_axis"]
        assert 0 <= axis < data.ndim
    else:
        # Single-channel: channel_axis may be absent or present, but if present, it must be valid
        if "channel_axis" in add_kwargs:
            axis = add_kwargs["channel_axis"]
            assert 0 <= axis < data.ndim


# --------------------------------------------------------------------- #
#  Labels mode: OME-Arrow sources
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        _p("ome-artificial-5d-datasets", "single-channel.ome.tiff"),
        _p("ome-artificial-5d-datasets", "multi-channel.ome.tiff"),
        _p("ome-artificial-5d-datasets", "z-series.ome.tiff"),
    ],
)
def test_reader_labels_mode_ome_sources(path: str):
    """
    In labels mode, OME-Arrow-backed sources should yield labels layers
    with integer dtype and no channel_axis.
    """
    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "labels"):
        reader = napari_get_reader(path)
        assert callable(reader)

        layer_data_list = reader(path)
        assert isinstance(layer_data_list, list)
        assert len(layer_data_list) == 1

        data, add_kwargs, layer_type = layer_data_list[0]

    assert layer_type == "labels"
    assert isinstance(data, np.ndarray)
    assert np.issubdtype(data.dtype, np.integer)
    assert "channel_axis" not in add_kwargs


# --------------------------------------------------------------------- #
#  .npy fallback behavior
# --------------------------------------------------------------------- #


def test_reader_npy_image_mode(tmp_path: Path):
    """
    .npy fallback should behave in image mode and preserve the data.
    """
    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "image"):
        my_test_file = tmp_path / "myfile.npy"
        original = np.random.rand(20, 20).astype(np.float32)
        np.save(my_test_file, original)

        reader = napari_get_reader(str(my_test_file))
        assert callable(reader)

        layer_data_list = reader(str(my_test_file))
        assert isinstance(layer_data_list, list)
        assert len(layer_data_list) == 1

        data, add_kwargs, layer_type = layer_data_list[0]

    assert layer_type == "image"
    np.testing.assert_allclose(original, data)


def test_reader_npy_labels_mode(tmp_path: Path):
    """
    .npy fallback should support labels mode, converting to integer labels.
    """
    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "labels"):
        my_test_file = tmp_path / "labels.npy"
        original = np.random.rand(20, 20).astype(np.float32)
        np.save(my_test_file, original)

        reader = napari_get_reader(str(my_test_file))
        assert callable(reader)

        layer_data_list = reader(str(my_test_file))
        assert isinstance(layer_data_list, list)
        assert len(layer_data_list) == 1

        data, add_kwargs, layer_type = layer_data_list[0]

    assert layer_type == "labels"
    assert np.issubdtype(data.dtype, np.integer)


# --------------------------------------------------------------------- #
#  OME-Parquet: multi-row grid behavior
# --------------------------------------------------------------------- #


def test_reader_prompts_for_stack_pattern_with_multiple_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Multiple files should trigger a stack pattern prompt and use the result.
    """
    paths = []
    for idx in range(3):
        arr = np.ones((4, 4), dtype=np.uint16) * idx
        p = tmp_path / f"img_{idx:03d}.npy"
        np.save(p, arr)
        paths.append(str(p))

    suggested = str(tmp_path / "img_<000-002>.npy")

    monkeypatch.setattr(
        reader_mod,
        "_get_layer_mode",
        lambda sample_path, image_type_hint=None: "image",
    )
    monkeypatch.setattr(
        reader_mod, "_prompt_stack_pattern", lambda files, folder: suggested
    )

    captured: dict[str, object] = {}

    def fake_read_one(
        src: str, mode: str, *, stack_default_dim: str | None = None
    ):
        captured["src"] = src
        captured["mode"] = mode
        captured["stack_default_dim"] = stack_default_dim
        data = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
        return data, {"name": "stack"}, "image"

    monkeypatch.setattr(reader_mod, "_read_one", fake_read_one)

    layers = reader_mod.reader_function(paths)
    assert len(layers) == 1
    assert captured["src"] == suggested
    assert captured["mode"] == "image"
    assert captured["stack_default_dim"] == "Z"


def test_reader_stack_pattern_nviz_dataset(monkeypatch: pytest.MonkeyPatch):
    """
    Use the nviz artificial dataset with a known pattern string.
    """
    pattern = (
        DATA_ROOT
        / "nviz-artificial-4d-dataset"
        / "E99_C<111,222>_ZS<000-021>.tif"
    )
    folder = pattern.parent
    files = sorted(folder.glob("E99_C111_ZS*.tif"))[:2]
    assert len(files) == 2

    monkeypatch.setattr(
        reader_mod,
        "_prompt_stack_pattern",
        lambda _files, _folder: str(pattern),
    )

    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "image"):
        layers = reader_mod.reader_function([str(p) for p in files])

    assert len(layers) == 2
    expected_labels = ["111", "222"]
    for idx, (data, add_kwargs, layer_type) in enumerate(layers):
        assert layer_type == "image"
        assert data.ndim >= 3
        assert data.shape[-3] == 22
        assert add_kwargs["name"].endswith(f"[{expected_labels[idx]}]")


def test_reader_stack_pattern_applies_scale_override(
    monkeypatch: pytest.MonkeyPatch,
):
    pattern = (
        DATA_ROOT
        / "nviz-artificial-4d-dataset"
        / "E99_C<111,222>_ZS<000-021>.tif"
    )
    folder = pattern.parent
    files = sorted(folder.glob("E99_C111_ZS*.tif"))[:2]
    assert len(files) == 2

    monkeypatch.setattr(
        reader_mod,
        "_prompt_stack_pattern",
        lambda _files, _folder: str(pattern),
    )

    with (
        temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "image"),
        temporary_env_var("NAPARI_OME_ARROW_STACK_SCALE", "1.0,0.5,0.2"),
    ):
        layers = reader_mod.reader_function([str(p) for p in files])

    assert len(layers) == 2
    for _data, add_kwargs, layer_type in layers:
        assert layer_type == "image"
        assert "scale" in add_kwargs
        assert add_kwargs["scale"] == pytest.approx((1.0, 1.0, 0.5, 0.2))


def test_infer_layer_mode_from_image_type():
    record = {"image_type": "labels"}
    assert reader_mod._infer_layer_mode_from_record(record) == "labels"

    record = {"pixels_meta": {"image_type": "image"}}
    assert reader_mod._infer_layer_mode_from_record(record) == "image"


def test_suggest_stack_pattern_nviz_dataset():
    folder = DATA_ROOT / "nviz-artificial-4d-dataset"
    files = sorted(folder.glob("E99_C*_ZS*.tif"))
    assert files
    suggested = reader_mod._suggest_stack_pattern(files, folder)
    assert suggested == str(folder / "E99_C<111,222>_ZS<000-021>.tif")


def test_suggest_stack_pattern_z_token_nonsequential_channels(tmp_path: Path):
    names = [
        "C10-1_405_ZS000_FOV-1.tif",
        "C10-1_405_ZS001_FOV-1.tif",
        "C10-1_488_ZS000_FOV-1.tif",
        "C10-1_555_ZS001_FOV-1.tif",
        "C10-1_Merge_ZS002_FOV-1.tif",
        "C10-1_TRANS_ZS003_FOV-1.tif",
    ]
    for name in names:
        (tmp_path / name).touch()
    files = [tmp_path / name for name in names]
    suggested = reader_mod._suggest_stack_pattern(files, tmp_path)
    assert suggested == str(
        tmp_path / "C10-1_<405,488,555,Merge,TRANS>_ZS<000-003>_FOV-1.tif"
    )


def _struct_columns(path: str) -> list[str]:
    table = pq.read_table(path)
    cols = []
    for name, col in zip(table.column_names, table.columns, strict=False):
        if pa.types.is_struct(col.type):
            cols.append(name)
    return cols


def test_reader_parquet_emits_one_layer_per_row():
    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "image"):
        path = _p("cytodataframe", "BR00117006.ome.parquet")
        struct_cols = _struct_columns(path)
        assert struct_cols, (
            "Expected at least one struct column in test parquet."
        )

        reader = napari_get_reader(path)
        assert callable(reader)

        layers = reader(path)

    table = pq.read_table(path)
    assert len(layers) == table.num_rows
    col = struct_cols[0]
    for idx, (data, add_kwargs, layer_type) in enumerate(layers):
        assert layer_type == "image"
        assert isinstance(data, np.ndarray)
        assert add_kwargs["name"].startswith(
            f"BR00117006.ome.parquet[{col}][row {idx}]"
        )


def test_reader_parquet_respects_column_override():
    target_col = "Image_FileName_OrigRNA_OMEArrow_COMP"
    with (
        temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "labels"),
        temporary_env_var("NAPARI_OME_ARROW_PARQUET_COLUMN", target_col),
    ):
        path = _p("cytodataframe", "BR00117006.ome.parquet")
        reader = napari_get_reader(path)
        assert callable(reader)

        layers = reader(path)

    assert layers, "Expected at least one layer from parquet override."
    for data, add_kwargs, layer_type in layers:
        assert layer_type == "labels"
        assert target_col in add_kwargs["name"]
        assert "channel_axis" not in add_kwargs
        assert np.issubdtype(data.dtype, np.integer)


def test_reader_vortex_emits_layers():
    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "labels"):
        path = _p("idr0062A", "6001240_labels.ome.vortex")
        reader = napari_get_reader(path)
        assert callable(reader)

        layers = reader(path)

    assert len(layers) == 1
    data, add_kwargs, layer_type = layers[0]
    assert layer_type == "labels"
    assert isinstance(data, np.ndarray)
    assert add_kwargs["name"] == "6001240_labels.ome.vortex"


# --------------------------------------------------------------------- #
#  Auto-3D behavior for Z-stacks (no monkeypatch)
# --------------------------------------------------------------------- #


def test_z_stack_switches_viewer_to_3d():
    """
    For Z-stacks, the reader should set viewer.dims.ndisplay = 3
    via _maybe_set_viewer_3d and napari.current_viewer().

    This test assumes:
      - napari is installed
      - napari.Viewer() registers itself as current_viewer()
    """
    napari = pytest.importorskip("napari")

    with temporary_env_var("NAPARI_OME_ARROW_LAYER_TYPE", "image"):
        try:
            viewer = napari.Viewer()
        except ImportError as e:
            pytest.skip(
                f"Skipping viewer construction due to dependency error: {e}"
            )
        try:
            # Start explicitly in 2D
            viewer.dims.ndisplay = 2

            path = _p("ome-artificial-5d-datasets", "z-series.ome.tiff")
            reader = napari_get_reader(path)
            assert callable(reader)

            # Running the reader should trigger _maybe_set_viewer_3d(...)
            _ = reader(path)

            assert viewer.dims.ndisplay == 3
        finally:
            # Clean up viewer so the test doesn't leak windows/resources
            viewer.close()
