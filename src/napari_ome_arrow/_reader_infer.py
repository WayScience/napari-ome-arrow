"""Inference helpers for determining reader mode and source support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ome_arrow.core import OMEArrow
from ome_arrow.ingest import from_stack_pattern_path

from ._reader_omearrow import _read_vortex_scalar

LOGGER = logging.getLogger(__name__)


def _normalize_image_type(value: Any) -> str | None:
    """Normalize an image type hint to "image" or "labels".

    Args:
        value: Raw hint value from metadata or user input.

    Returns:
        "image", "labels", or None if it cannot be inferred.
    """
    if value is None:
        return None
    # Normalize arbitrary input to a comparable string token.
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"image", "intensity", "raw"} or "image" in text:
        return "image"
    if (
        text
        in {
            "labels",
            "label",
            "mask",
            "masks",
            "segmentation",
            "seg",
            "outlines",
            "outline",
        }
        or "label" in text
        or "mask" in text
        or "seg" in text
        or "outline" in text
    ):
        return "labels"
    return None


def _infer_layer_mode_from_record(record: dict[str, Any]) -> str | None:
    """Infer the layer mode from a parsed OME-Arrow record.

    Args:
        record: Parsed metadata record from OME-Arrow.

    Returns:
        "image", "labels", or None if no hint is present.
    """
    # Collect candidate metadata fields that may encode image type.
    candidates = []
    if "image_type" in record:
        candidates.append(record.get("image_type"))
    pixels_meta = record.get("pixels_meta")
    if isinstance(pixels_meta, dict) and "image_type" in pixels_meta:
        candidates.append(pixels_meta.get("image_type"))
    for value in candidates:
        inferred = _normalize_image_type(value)
        if inferred:
            return inferred
    return None


def _infer_layer_mode_from_ome_arrow(obj: OMEArrow) -> str | None:
    """Infer the layer mode from an OMEArrow object.

    Args:
        obj: OMEArrow instance to inspect.

    Returns:
        "image", "labels", or None if no hint is present.
    """
    try:
        record = obj.data.as_py()
    except Exception:
        return None
    if not isinstance(record, dict):
        return None
    return _infer_layer_mode_from_record(record)


def _infer_layer_mode_from_source(
    src: str, *, stack_default_dim: str | None = None
) -> str | None:
    """Infer the layer mode by inspecting a data source path.

    Args:
        src: Source path or stack pattern.
        stack_default_dim: Default dimension to use for stack patterns.

    Returns:
        "image", "labels", or None if inference fails.
    """
    # Quick path sniffing before attempting to instantiate OMEArrow.
    s = src.lower()
    p = Path(src)
    looks_stack = any(c in src for c in "<>*")
    looks_zarr = (
        s.endswith((".ome.zarr", ".zarr"))
        or ".zarr/" in s
        or p.exists()
        and p.is_dir()
        and p.suffix.lower() == ".zarr"
    )
    looks_parquet = s.endswith(
        (".ome.parquet", ".parquet", ".pq")
    ) or p.suffix.lower() in {
        ".parquet",
        ".pq",
    }
    looks_vortex = s.endswith(
        (".ome.vortex", ".vortex")
    ) or p.suffix.lower() in {
        ".vortex",
    }
    looks_tiff = s.endswith(
        (".ome.tif", ".ome.tiff", ".tif", ".tiff")
    ) or p.suffix.lower() in {
        ".tif",
        ".tiff",
    }
    if not (
        looks_stack
        or looks_zarr
        or looks_parquet
        or looks_tiff
        or looks_vortex
    ):
        LOGGER.debug(
            "Skipping layer-mode inference for non-OME source: %s", src
        )
        return None

    scalar = None
    try:
        if looks_vortex:
            scalar = _read_vortex_scalar(src)
        if looks_stack and stack_default_dim is not None:
            scalar = from_stack_pattern_path(
                src,
                default_dim_for_unspecified=stack_default_dim,
                map_series_to="T",
                clamp_to_uint16=True,
            )
        obj = OMEArrow(scalar if scalar is not None else src)
    except Exception:
        return None
    return _infer_layer_mode_from_ome_arrow(obj)


def _looks_like_ome_source(path_str: str) -> bool:
    """Check whether a path appears to be an OME-Arrow supported source.

    Args:
        path_str: Path or stack pattern string.

    Returns:
        True if the path looks like a supported source, otherwise False.
    """
    s = path_str.strip().lower()
    p = Path(path_str)

    # Bio-Formats-style stack patterns.
    looks_stack = any(c in path_str for c in "<>*")
    looks_zarr = (
        s.endswith((".ome.zarr", ".zarr"))
        or ".zarr/" in s
        or p.exists()
        and p.is_dir()
        and p.suffix.lower() == ".zarr"
    )
    looks_parquet = s.endswith(
        (".ome.parquet", ".parquet", ".pq")
    ) or p.suffix.lower() in {
        ".parquet",
        ".pq",
    }
    looks_vortex = s.endswith(
        (".ome.vortex", ".vortex")
    ) or p.suffix.lower() in {
        ".vortex",
    }
    looks_tiff = s.endswith(
        (".ome.tif", ".ome.tiff", ".tif", ".tiff")
    ) or p.suffix.lower() in {
        ".tif",
        ".tiff",
    }
    looks_npy = s.endswith(".npy")
    looks_dir_stack = False
    if p.exists() and p.is_dir() and not looks_zarr:
        try:
            # Heuristic: directory with multiple image-like files.
            stack_files = [
                f
                for f in p.iterdir()
                if f.is_file()
                and f.name.lower().endswith(
                    (
                        ".ome.tif",
                        ".ome.tiff",
                        ".tif",
                        ".tiff",
                        ".npy",
                    )
                )
            ]
            looks_dir_stack = len(stack_files) > 1
        except OSError:
            looks_dir_stack = False
    return (
        looks_stack
        or looks_zarr
        or looks_parquet
        or looks_vortex
        or looks_tiff
        or looks_npy
        or looks_dir_stack
    )
