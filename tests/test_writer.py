"""
Tests for the napari-ome-arrow writer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from napari_ome_arrow import napari_write_image, napari_write_labels


def test_write_image_parquet_2d():
    """Test writing a 2D image to OME-Parquet format."""
    # Create simple 2D test data
    data = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    meta = {"name": "test_image"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_output.ome.parquet")

        # Write the data
        result = napari_write_image(output_path, data, meta)

        # Verify the file was created
        assert result == output_path
        assert Path(output_path).exists()

        # Verify the file can be read back as parquet
        table = pq.read_table(output_path)
        assert table.num_rows == 1
        assert table.num_columns == 1


def test_write_image_parquet_3d():
    """Test writing a 3D image to OME-Parquet format."""
    # Create 3D test data (Z, Y, X)
    data = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint16)
    meta = {"name": "test_3d_image"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_3d.ome.parquet")

        # Write the data
        result = napari_write_image(output_path, data, meta)

        # Verify the file was created
        assert result == output_path
        assert Path(output_path).exists()


def test_write_image_with_scale():
    """Test writing an image with scale metadata."""
    data = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    meta = {
        "name": "scaled_image",
        "scale": (0.5, 0.5),  # Physical pixel sizes
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "scaled.ome.parquet")

        # Write the data
        result = napari_write_image(output_path, data, meta)

        # Verify the file was created
        assert result == output_path
        assert Path(output_path).exists()


def test_write_labels_parquet():
    """Test writing labels data to OME-Parquet format."""
    # Create label data
    data = np.random.randint(0, 10, (100, 100), dtype=np.uint32)
    meta = {"name": "test_labels"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "labels.ome.parquet")

        # Write the data
        result = napari_write_labels(output_path, data, meta)

        # Verify the file was created
        assert result == output_path
        assert Path(output_path).exists()


def test_write_vortex():
    """Test writing to vortex format."""
    data = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    meta = {"name": "test_vortex"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.ome.vortex")

        # Try to write to vortex format
        result = napari_write_image(output_path, data, meta)

        # If vortex-data is installed, file should be created
        # If not installed, result should be None
        if result is not None:
            assert result == output_path
            assert Path(output_path).exists()
        else:
            # vortex-data not installed
            assert not Path(output_path).exists()


def test_write_empty_data():
    """Test that writing empty data list returns None."""
    data = []
    meta = {"name": "empty"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "empty.ome.parquet")

        with pytest.warns(UserWarning, match="Empty layer list"):
            result = napari_write_image(output_path, data, meta)
            assert result is None


def test_write_unsupported_extension():
    """Test that unsupported file extensions return None."""
    data = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    meta = {"name": "test"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test.txt")

        with pytest.warns(UserWarning, match="Unsupported file extension"):
            result = napari_write_image(output_path, data, meta)
            assert result is None


def test_write_layer_tuple():
    """Test writing when data is provided as a layer tuple."""
    array_data = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    layer_tuple = (array_data, {"name": "layer"}, "image")
    data = [layer_tuple]
    meta = {"name": "test_tuple"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "tuple.ome.parquet")

        result = napari_write_image(output_path, data, meta)

        assert result == output_path
        assert Path(output_path).exists()


def test_write_5d_image():
    """Test writing a 5D image (TCZYX) to OME-Parquet."""
    # Create 5D test data (T, C, Z, Y, X)
    data = np.random.randint(0, 255, (2, 3, 5, 100, 100), dtype=np.uint16)
    meta = {"name": "test_5d"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_5d.ome.parquet")

        result = napari_write_image(output_path, data, meta)

        assert result == output_path
        assert Path(output_path).exists()


def test_roundtrip_parquet():
    """Test that data can be written and read back."""
    from napari_ome_arrow import napari_get_reader

    # Create test data
    original_data = np.random.randint(0, 255, (50, 50), dtype=np.uint16)
    meta = {"name": "roundtrip_test"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "roundtrip.ome.parquet")

        # Write the data
        result = napari_write_image(output_path, original_data, meta)
        assert result == output_path

        # Read the data back
        reader = napari_get_reader(output_path)
        assert reader is not None

        # Set environment variable to avoid GUI prompt
        import os

        old_env = os.environ.get("NAPARI_OME_ARROW_LAYER_TYPE")
        os.environ["NAPARI_OME_ARROW_LAYER_TYPE"] = "image"
        try:
            layers = reader(output_path)
            assert len(layers) > 0

            # Extract the array from the first layer
            read_data = layers[0][0]

            # Shapes should match (though dimension order might differ)
            assert read_data.shape[-2:] == original_data.shape[-2:]
        finally:
            if old_env is None:
                os.environ.pop("NAPARI_OME_ARROW_LAYER_TYPE", None)
            else:
                os.environ["NAPARI_OME_ARROW_LAYER_TYPE"] = old_env
