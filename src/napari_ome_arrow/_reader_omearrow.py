"""OME-Arrow backed reading helpers."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from ome_arrow.core import OMEArrow
from ome_arrow.ingest import from_stack_pattern_path
from ome_arrow.meta import OME_ARROW_STRUCT

from ._reader_napari import _as_labels, _enable_grid, _maybe_set_viewer_3d
from ._reader_stack import (
    _parse_stack_scale,
    _prompt_stack_scale,
    _scale_for_array,
    _scale_from_ome_arrow,
)
from ._reader_types import LayerData


def _find_ome_parquet_columns(table: pa.Table) -> list[str]:
    """Find Parquet columns matching the OME-Arrow schema.

    Args:
        table: Parquet table to scan.

    Returns:
        Column names that match the OME-Arrow struct schema.
    """
    import pyarrow as pa

    # Match struct columns that exactly mirror the OME-Arrow schema.
    expected_fields = {f.name for f in OME_ARROW_STRUCT}
    names: list[str] = []
    for name, col in zip(table.column_names, table.columns, strict=False):
        if (
            pa.types.is_struct(col.type)
            and {f.name for f in col.type} == expected_fields
        ):
            names.append(name)
    return names


def _read_vortex_scalar(src: str) -> pa.StructScalar:
    """Read a single OME-Vortex row as a struct scalar.

    Args:
        src: OME-Vortex file path.

    Returns:
        Struct scalar for the selected row and column.

    Raises:
        ImportError: If OME-Vortex support is not available.
    """
    # Delegate Vortex parsing to ome-arrow, which handles the file format details.
    try:
        from ome_arrow.ingest import from_ome_vortex
    except Exception as exc:
        raise ImportError(
            "OME-Vortex support requires ome-arrow with vortex support and the "
            "optional 'vortex' extra (vortex-data). Install with "
            "'pip install \"napari-ome-arrow[vortex]\"'."
        ) from exc

    # Allow column override for non-default schema names.
    override = os.environ.get(
        "NAPARI_OME_ARROW_VORTEX_COLUMN"
    ) or os.environ.get("NAPARI_OME_ARROW_PARQUET_COLUMN")
    # Use a single row from the requested/auto-detected column for OMEArrow.
    return from_ome_vortex(
        src,
        column_name=override or "ome_arrow",
        row_index=0,
        strict_schema=False,
    )


def _vortex_row_out_of_range(exc: Exception) -> bool:
    """Check whether a Vortex row exception indicates an out-of-range index.

    Args:
        exc: Exception raised by the Vortex reader.

    Returns:
        True if the error indicates an out-of-range row, otherwise False.
    """
    if isinstance(exc, IndexError):
        return True
    msg = str(exc).lower()
    return "out of range" in msg or ("row_index" in msg and "range" in msg)


def _read_vortex_rows(src: str, mode: str) -> list[LayerData] | None:
    """Read multi-row OME-Vortex data as a layer grid.

    Args:
        src: OME-Vortex file path.
        mode: "image" or "labels".

    Returns:
        A list of layers if multi-row data is detected, otherwise None.
    """
    s = src.lower()
    if not (s.endswith((".ome.vortex", ".vortex"))):
        return None

    try:
        from ome_arrow.ingest import from_ome_vortex
    except Exception:
        return None

    # Reuse the parquet override env var for consistency.
    override = os.environ.get(
        "NAPARI_OME_ARROW_VORTEX_COLUMN"
    ) or os.environ.get("NAPARI_OME_ARROW_PARQUET_COLUMN")
    selected = override or "ome_arrow"

    try:
        first = from_ome_vortex(
            src,
            column_name=selected,
            row_index=0,
            strict_schema=False,
        )
    except Exception:
        return None

    try:
        second = from_ome_vortex(
            src,
            column_name=selected,
            row_index=1,
            strict_schema=False,
        )
    except Exception:
        return None

    layers: list[LayerData] = []

    def _append_layer(idx: int, scalar: pa.StructScalar) -> None:
        # Per-row conversion with best-effort error handling.
        try:
            arr = OMEArrow(scalar).export(
                how="numpy", dtype=np.uint16, strict=False
            )
        except Exception as e:  # pragma: no cover - warn and skip bad rows
            warnings.warn(
                f"Skipping row {idx} in column '{selected}': {e}",
                stacklevel=2,
            )
            return

        # Layer metadata uses row index for unique naming.
        add_kwargs: dict[str, Any] = {
            "name": f"{Path(src).name}[{selected}][row {idx}]"
        }
        if mode == "image":
            if arr.ndim >= 5:
                add_kwargs["channel_axis"] = 1  # TCZYX
            elif arr.ndim == 4:
                add_kwargs["channel_axis"] = 0  # CZYX
            layer_type = "image"
        else:
            if arr.ndim == 5:
                arr = arr[:, 0, ...]
            elif arr.ndim == 4:
                arr = arr[0, ...]
            arr = _as_labels(arr)
            add_kwargs.setdefault("opacity", 0.7)
            layer_type = "labels"

        _maybe_set_viewer_3d(arr)
        layers.append((arr, add_kwargs, layer_type))

    _append_layer(0, first)
    _append_layer(1, second)

    max_rows = 10000
    for idx in range(2, max_rows):
        try:
            scalar = from_ome_vortex(
                src,
                column_name=selected,
                row_index=idx,
                strict_schema=False,
            )
        except Exception as e:
            if _vortex_row_out_of_range(e):
                break
            warnings.warn(
                f"Skipping row {idx} in column '{selected}': {e}",
                stacklevel=2,
            )
            continue
        _append_layer(idx, scalar)

    if layers:
        _enable_grid(len(layers))
    return layers or None


def _read_parquet_rows(src: str, mode: str) -> list[LayerData] | None:
    """Read multi-row OME-Parquet data as a layer grid.

    Args:
        src: OME-Parquet file path.
        mode: "image" or "labels".

    Returns:
        A list of layers if multi-row data is detected, otherwise None.
    """
    s = src.lower()
    if not (s.endswith((".ome.parquet", ".parquet", ".pq"))):
        return None

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return None

    # Read all rows; per-row layers are assembled below.
    table = pq.read_table(src)
    ome_cols = _find_ome_parquet_columns(table)
    if not ome_cols or table.num_rows <= 1:
        return None

    # Column override for multi-struct tables.
    override = os.environ.get("NAPARI_OME_ARROW_PARQUET_COLUMN")
    if override:
        matched = [c for c in ome_cols if c.lower() == override.lower()]
        selected = matched[0] if matched else ome_cols[0]
        if not matched:
            warnings.warn(
                f"Column '{override}' not found in {Path(src).name}; using {selected!r}.",
                stacklevel=2,
            )
    else:
        selected = ome_cols[0]

    column = table[selected]
    layers: list[LayerData] = []

    for idx in range(table.num_rows):
        try:
            record = column.slice(idx, 1).to_pylist()[0]
            scalar = pa.scalar(record, type=OME_ARROW_STRUCT)
            arr = OMEArrow(scalar).export(
                how="numpy", dtype=np.uint16, strict=False
            )
        except Exception as e:  # pragma: no cover - warn and skip bad rows
            warnings.warn(
                f"Skipping row {idx} in column '{selected}': {e}",
                stacklevel=2,
            )
            continue

        add_kwargs: dict[str, Any] = {
            "name": f"{Path(src).name}[{selected}][row {idx}]"
        }
        if mode == "image":
            if arr.ndim >= 5:
                add_kwargs["channel_axis"] = 1  # TCZYX
            elif arr.ndim == 4:
                add_kwargs["channel_axis"] = 0  # CZYX
            layer_type = "image"
        else:
            if arr.ndim == 5:
                arr = arr[:, 0, ...]
            elif arr.ndim == 4:
                arr = arr[0, ...]
            arr = _as_labels(arr)
            add_kwargs.setdefault("opacity", 0.7)
            layer_type = "labels"

        _maybe_set_viewer_3d(arr)
        layers.append((arr, add_kwargs, layer_type))

    if layers:
        _enable_grid(len(layers))
    return layers or None


def _read_one(
    src: str,
    mode: str,
    *,
    stack_default_dim: str | None = None,
    stack_scale_override: Sequence[float] | None = None,
) -> LayerData:
    """Read a single source into a napari layer tuple.

    Args:
        src: Source path or stack pattern.
        mode: "image" or "labels".
        stack_default_dim: Default dimension to use for stack patterns.
        stack_scale_override: Optional scale override for stack patterns.

    Returns:
        Tuple of (data, add_kwargs, layer_type).

    Raises:
        ValueError: If the source cannot be parsed or inferred.
        ImportError: If optional dependencies are required but missing.
        FileNotFoundError: If referenced stack files are missing.
    """
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
    looks_npy = s.endswith(".npy")

    add_kwargs: dict[str, Any] = {"name": p.name}

    # ---- OME-Arrow-backed sources -----------------------------------
    if (
        looks_stack
        or looks_zarr
        or looks_parquet
        or looks_tiff
        or looks_vortex
    ):
        scalar = None
        if looks_vortex:
            # Vortex needs ome-arrow's ingest helper to produce a typed scalar.
            scalar = _read_vortex_scalar(src)
        if looks_stack and stack_default_dim is not None:
            scalar = from_stack_pattern_path(
                src,
                default_dim_for_unspecified=stack_default_dim,
                map_series_to="T",
                clamp_to_uint16=True,
            )
        obj = OMEArrow(scalar if scalar is not None else src)
        arr = obj.export(how="numpy", dtype=np.uint16)  # TCZYX
        info = obj.info()  # may contain 'shape': (T, C, Z, Y, X)
        scale_override: tuple[float, ...] | None = None
        inferred_scale = _scale_from_ome_arrow(obj)
        if looks_stack:
            # Stack scales are optional and can be provided via env or prompt.
            if stack_scale_override is not None:
                scale_override = tuple(stack_scale_override)
            else:
                env_scale = os.environ.get("NAPARI_OME_ARROW_STACK_SCALE")
                if env_scale:
                    try:
                        scale_override = _parse_stack_scale(env_scale)
                    except ValueError as exc:
                        warnings.warn(
                            f"Invalid NAPARI_OME_ARROW_STACK_SCALE '{env_scale}': {exc}.",
                            stacklevel=2,
                        )
                        scale_override = None

                if scale_override is None and inferred_scale is None:
                    scale_override = _prompt_stack_scale(
                        sample_path=src, default_scale=None
                    )

        # Recover from accidental 1D flatten by using metadata shape.
        if getattr(arr, "ndim", 0) == 1:
            T, C, Z, Y, X = info.get("shape", (1, 1, 1, 0, 0))
            if Y and X and arr.size == Y * X:
                arr = arr.reshape((1, 1, 1, Y, X))
            else:
                raise ValueError(
                    f"Flat array with unknown shape for {src}: size={arr.size}"
                )

        if mode == "image":
            # Image: preserve channels
            if arr.ndim >= 5:
                add_kwargs["channel_axis"] = 1  # TCZYX
            elif arr.ndim == 4:
                add_kwargs["channel_axis"] = 0  # CZYX
            layer_type = "image"
        else:
            # Labels: squash channels, ensure integer dtype
            if arr.ndim == 5:  # (T, C, Z, Y, X)
                arr = arr[:, 0, ...]
            elif arr.ndim == 4:  # (C, Z, Y, X)
                arr = arr[0, ...]
            arr = _as_labels(arr)
            add_kwargs.setdefault("opacity", 0.7)
            layer_type = "labels"

        # 🔹 Ask viewer to switch to 3D if there is a real Z-stack
        _maybe_set_viewer_3d(arr)

        if looks_stack:
            # Stack scale can be inferred or user-provided.
            scale_override = (
                scale_override
                if scale_override is not None
                else inferred_scale
            )
            scale = (
                _scale_for_array(arr, mode, add_kwargs, scale_override)
                if scale_override is not None
                else None
            )
        else:
            scale = (
                _scale_for_array(arr, mode, add_kwargs, inferred_scale)
                if inferred_scale is not None
                else None
            )
        if scale is not None:
            add_kwargs["scale"] = scale

        return arr, add_kwargs, layer_type

    # ---- bare .npy fallback -----------------------------------------
    if looks_npy:
        # Minimal fallback for plain numpy arrays.
        arr = np.load(src)
        if arr.ndim == 1:
            n = int(np.sqrt(arr.size))
            if n * n == arr.size:
                arr = arr.reshape(n, n)
            else:
                raise ValueError(
                    f".npy is 1D and not a square image: {arr.shape}"
                )

        if mode == "image":
            if arr.ndim == 3 and arr.shape[0] <= 6:
                add_kwargs["channel_axis"] = 0
            layer_type = "image"
        else:
            # labels
            if arr.ndim == 3:  # treat as (C, Y, X) → first channel
                arr = arr[0, ...]
            arr = _as_labels(arr)
            add_kwargs.setdefault("opacity", 0.7)
            layer_type = "labels"

        # 🔹 Same 3D toggle for npy-based data
        _maybe_set_viewer_3d(arr)

        return arr, add_kwargs, layer_type

    raise ValueError(f"Unrecognized path for napari-ome-arrow reader: {src}")
