"""
Napari writer implementations for OME-Arrow formats.

Provides writers for:
- .ome.parquet
- .ome.vortex (requires vortex-data optional dependency)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from ome_arrow.core import from_numpy, to_ome_parquet, to_ome_vortex

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "napari_write_image",
    "napari_write_labels",
]


def _extract_metadata_from_napari(
    data: np.ndarray | Sequence[tuple[Any, dict, str]], meta: dict
) -> dict[str, Any]:
    """Extract metadata from napari layer for OME-Arrow export.

    Args:
        data: Image data (numpy array) or list of layer tuples.
        meta: Napari layer metadata dictionary.

    Returns:
        Dictionary of metadata for OME-Arrow conversion.
    """
    metadata: dict[str, Any] = {}

    # Extract name from metadata
    if "name" in meta:
        metadata["name"] = meta["name"]

    # Extract scale information for physical pixel sizes
    if "scale" in meta and meta["scale"]:
        scale = meta["scale"]
        # napari scale is in (Z, Y, X) or (Y, X) format typically
        if len(scale) >= 2:
            metadata["physical_size_y"] = float(scale[-2])
            metadata["physical_size_x"] = float(scale[-1])
        if len(scale) >= 3:
            metadata["physical_size_z"] = float(scale[-3])

    # Extract channel names if available
    if (
        "channel_axis" in meta
        and meta.get("channel_axis") is not None
        and hasattr(data, "shape")
    ):
        channel_axis = meta["channel_axis"]
        if channel_axis < len(data.shape):
            n_channels = data.shape[channel_axis]
            metadata["channel_names"] = [
                f"Channel {i}" for i in range(n_channels)
            ]

    return metadata


def _determine_dim_order(data: np.ndarray) -> str:
    """Determine dimension order from array shape.

    Args:
        data: Numpy array to analyze.

    Returns:
        Dimension order string (e.g., "YX", "ZYX", "TCZYX").
    """
    ndim = data.ndim

    if ndim == 2:
        return "YX"
    if ndim == 3:
        # Could be ZYX, CYX, or TYX - default to ZYX
        return "ZYX"
    if ndim == 4:
        # Could be CZYX or TZYX - default to CZYX
        return "CZYX"
    if ndim == 5:
        return "TCZYX"

    # For 1D or >5D, pad with Y/X at the end
    if ndim == 1:
        return "X"

    raise ValueError(f"Unsupported number of dimensions: {ndim}")


def napari_write_image(
    path: str, data: Any, meta: dict
) -> str | list[str] | None:
    """Write image data to OME-Parquet or OME-Vortex format.

    This function is called by napari when the user saves an image layer
    to a .ome.parquet or .ome.vortex file.

    Args:
        path: Output file path (must end with .ome.parquet or .ome.vortex).
        data: Image data as numpy array or list of layer tuples.
        meta: Layer metadata dictionary from napari.

    Returns:
        Path to the written file, or None if writing failed.
    """
    try:
        # Handle list of layers (multi-layer write)
        if isinstance(data, list):
            # For now, only write the first layer
            # Future enhancement: write multiple layers as multi-row table
            if not data:
                warnings.warn(
                    "Empty layer list provided for writing.",
                    stacklevel=2,
                )
                return None
            data = data[0]
            if isinstance(data, tuple):
                # Extract array from layer tuple (array, meta, layer_type)
                data = data[0]

        if not isinstance(data, np.ndarray):
            warnings.warn(
                f"Expected numpy array, got {type(data).__name__}",
                stacklevel=2,
            )
            return None

        # Extract metadata
        ome_meta = _extract_metadata_from_napari(data, meta)

        # Determine dimension order
        dim_order = _determine_dim_order(data)

        # Convert to OME-Arrow struct
        ome_struct = from_numpy(
            data,
            dim_order=dim_order,
            name=ome_meta.get("name"),
            image_type="image",
            physical_size_x=ome_meta.get("physical_size_x", 1.0),
            physical_size_y=ome_meta.get("physical_size_y", 1.0),
            physical_size_z=ome_meta.get("physical_size_z", 1.0),
            channel_names=ome_meta.get("channel_names"),
        )

        # Write to appropriate format based on extension
        path_lower = path.lower()
        if path_lower.endswith((".ome.parquet", ".parquet", ".pq")):
            to_ome_parquet(ome_struct, path)
        elif path_lower.endswith((".ome.vortex", ".vortex")):
            try:
                to_ome_vortex(ome_struct, path)
            except ImportError:
                warnings.warn(
                    "Writing to .ome.vortex requires the 'vortex-data' package. "
                    "Install with: pip install napari-ome-arrow[vortex]",
                    stacklevel=2,
                )
                return None
        else:
            warnings.warn(
                f"Unsupported file extension for path: {path}",
                stacklevel=2,
            )
            return None

        return path

    except Exception as e:
        warnings.warn(
            f"Failed to write OME-Arrow data to {path}: {e}",
            stacklevel=2,
        )
        return None


def napari_write_labels(
    path: str, data: Any, meta: dict
) -> str | list[str] | None:
    """Write labels data to OME-Parquet or OME-Vortex format.

    This function is called by napari when the user saves a labels layer
    to a .ome.parquet or .ome.vortex file.

    Args:
        path: Output file path (must end with .ome.parquet or .ome.vortex).
        data: Labels data as numpy array or list of layer tuples.
        meta: Layer metadata dictionary from napari.

    Returns:
        Path to the written file, or None if writing failed.
    """
    try:
        # Handle list of layers
        if isinstance(data, list):
            if not data:
                warnings.warn(
                    "Empty layer list provided for writing.",
                    stacklevel=2,
                )
                return None
            data = data[0]
            if isinstance(data, tuple):
                data = data[0]

        if not isinstance(data, np.ndarray):
            warnings.warn(
                f"Expected numpy array, got {type(data).__name__}",
                stacklevel=2,
            )
            return None

        # Extract metadata
        ome_meta = _extract_metadata_from_napari(data, meta)

        # Determine dimension order
        dim_order = _determine_dim_order(data)

        # Convert to OME-Arrow struct with image_type="label"
        ome_struct = from_numpy(
            data,
            dim_order=dim_order,
            name=ome_meta.get("name"),
            image_type="label",
            physical_size_x=ome_meta.get("physical_size_x", 1.0),
            physical_size_y=ome_meta.get("physical_size_y", 1.0),
            physical_size_z=ome_meta.get("physical_size_z", 1.0),
            clamp_to_uint16=False,  # Don't clamp labels
        )

        # Write to appropriate format
        path_lower = path.lower()
        if path_lower.endswith((".ome.parquet", ".parquet", ".pq")):
            to_ome_parquet(ome_struct, path)
        elif path_lower.endswith((".ome.vortex", ".vortex")):
            try:
                to_ome_vortex(ome_struct, path)
            except ImportError:
                warnings.warn(
                    "Writing to .ome.vortex requires the 'vortex-data' package. "
                    "Install with: pip install napari-ome-arrow[vortex]",
                    stacklevel=2,
                )
                return None
        else:
            warnings.warn(
                f"Unsupported file extension for path: {path}",
                stacklevel=2,
            )
            return None

        return path

    except Exception as e:
        warnings.warn(
            f"Failed to write OME-Arrow labels to {path}: {e}",
            stacklevel=2,
        )
        return None
