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


# --------------------------------------------------------------------- #
#  Auto-3D behavior for Z-stacks (no monkeypatch)
# --------------------------------------------------------------------- #


@pytest.mark.skipif(
    "CI" in os.environ and os.environ["CI"].lower() == "true",
    reason="May require a functional Qt backend; skip in CI by default.",
)
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
