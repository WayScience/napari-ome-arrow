"""
Minimal napari reader for OME-Arrow sources (stack patterns, OME-Zarr, OME-Parquet,
OME-Vortex, OME-TIFF) plus a fallback .npy example.

Behavior:
    * If NAPARI_OME_ARROW_LAYER_TYPE is set to "image" or "labels",
      that choice is used.
    * Otherwise, in a GUI/Qt context, the user is prompted with a modal
      dialog asking whether to load as image or labels.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pyarrow as pa
from ome_arrow.core import OMEArrow
from ome_arrow.meta import OME_ARROW_STRUCT

PathLike = Union[str, Path]
LayerData = tuple[np.ndarray, dict[str, Any], str]


def _maybe_set_viewer_3d(arr: np.ndarray) -> None:
    """
    If the array has a Z axis with size > 1, switch the current napari viewer
    to 3D (ndisplay = 3).

    Assumes OME-Arrow's TCZYX convention or a subset, i.e., Z is always
    the third-from-last axis. No-op if there's no active viewer.
    """
    # Need at least (Z, Y, X)
    if arr.ndim < 3:
        return

    z_size = arr.shape[-3]
    if z_size <= 1:
        return

    try:
        import napari

        viewer = napari.current_viewer()
    except Exception:
        # no viewer / not in GUI context → silently skip
        return

    if viewer is not None:
        viewer.dims.ndisplay = 3


# --------------------------------------------------------------------- #
#  Mode selection (env var + GUI prompt)
# --------------------------------------------------------------------- #


def _get_layer_mode(sample_path: str) -> str:
    """
    Decide whether to load as 'image' or 'labels'.

    Priority:
      1. NAPARI_OME_ARROW_LAYER_TYPE env var (image/labels)
      2. If in a Qt GUI context, show a modal dialog asking the user
      3. Otherwise, default to 'image'
    """
    mode = os.environ.get("NAPARI_OME_ARROW_LAYER_TYPE")
    if mode is not None:
        mode = mode.lower()
        if mode in {"image", "labels"}:
            return mode
        raise RuntimeError(
            f"Invalid NAPARI_OME_ARROW_LAYER_TYPE={mode!r}; expected 'image' or 'labels'."
        )

    # No env var → try to prompt in GUI context
    try:
        from qtpy import QtWidgets
    except Exception:
        # no Qt, probably headless: default to image
        warnings.warn(
            "NAPARI_OME_ARROW_LAYER_TYPE not set and Qt not available; "
            "defaulting to 'image'.",
            stacklevel=2,
        )
        return "image"

    app = QtWidgets.QApplication.instance()
    if app is None:
        # Again, likely headless or non-Qt usage
        warnings.warn(
            "NAPARI_OME_ARROW_LAYER_TYPE not set and no QApplication instance; "
            "defaulting to 'image'.",
            stacklevel=2,
        )
        return "image"

    # Build a simple modal choice dialog
    msg = QtWidgets.QMessageBox()
    msg.setWindowTitle("napari-ome-arrow: choose layer type")
    msg.setText(
        f"<p align='left'>How should '{Path(sample_path).name}' be loaded?<br><br>"
        "You can also set NAPARI_OME_ARROW_LAYER_TYPE=image or labels to skip this prompt.</p>"
    )

    # Use ActionRole for ALL buttons so Qt does NOT reorder them
    image_button = msg.addButton("Image", QtWidgets.QMessageBox.ActionRole)
    labels_button = msg.addButton("Labels", QtWidgets.QMessageBox.ActionRole)
    cancel_button = msg.addButton("Cancel", QtWidgets.QMessageBox.ActionRole)

    # If you want Esc to behave as Cancel:
    msg.setEscapeButton(cancel_button)

    msg.exec_()
    clicked = msg.clickedButton()

    if clicked is labels_button:
        return "labels"
    if clicked is image_button:
        return "image"

    raise RuntimeError("User cancelled napari-ome-arrow load dialog.")


# --------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------- #


def _as_labels(arr: np.ndarray) -> np.ndarray:
    """Convert any array into an integer label array."""
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.round(arr).astype(np.int32, copy=False)
    elif arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int32, copy=False)
    return arr


def _looks_like_ome_source(path_str: str) -> bool:
    """Basic extension / pattern sniffing for OME-Arrow supported formats."""
    s = path_str.strip().lower()
    p = Path(path_str)

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
    return (
        looks_stack
        or looks_zarr
        or looks_parquet
        or looks_vortex
        or looks_tiff
        or looks_npy
    )


# --------------------------------------------------------------------- #
#  OME-Parquet helpers (multi-row grid)
# --------------------------------------------------------------------- #


def _find_ome_parquet_columns(table: pa.Table) -> list[str]:
    """Return struct columns matching the OME-Arrow schema."""
    import pyarrow as pa

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
    # Delegate Vortex parsing to ome-arrow, which handles the file format details.
    try:
        from ome_arrow.ingest import from_ome_vortex
    except Exception as exc:
        raise ImportError(
            "OME-Vortex support requires ome-arrow with vortex support and the "
            "optional 'vortex' extra (vortex-data). Install with "
            "'pip install \"napari-ome-arrow[vortex]\"'."
        ) from exc

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
    if isinstance(exc, IndexError):
        return True
    msg = str(exc).lower()
    return "out of range" in msg or ("row_index" in msg and "range" in msg)


def _read_vortex_rows(src: str, mode: str) -> list[LayerData] | None:
    """
    Specialized path for multi-row OME-Vortex:
    create one layer per row and enable napari grid view.
    """
    s = src.lower()
    if not (s.endswith((".ome.vortex", ".vortex"))):
        return None

    try:
        from ome_arrow.ingest import from_ome_vortex
    except Exception:
        return None

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


def _enable_grid(n_layers: int) -> None:
    """Switch current viewer into grid view when possible."""
    if n_layers <= 1:
        return
    try:
        import napari

        viewer = napari.current_viewer()
    except Exception:
        return
    if viewer is None:
        return

    cols = math.ceil(math.sqrt(n_layers))
    rows = math.ceil(n_layers / cols)
    try:
        viewer.grid.enabled = True
        viewer.grid.shape = (rows, cols)
    except Exception:
        # grid is best-effort; ignore if unavailable
        return


def _read_parquet_rows(src: str, mode: str) -> list[LayerData] | None:
    """
    Specialized path for multi-row OME-Parquet:
    create one layer per row and enable napari grid view.
    """
    s = src.lower()
    if not (s.endswith((".ome.parquet", ".parquet", ".pq"))):
        return None

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return None

    table = pq.read_table(src)
    ome_cols = _find_ome_parquet_columns(table)
    if not ome_cols or table.num_rows <= 1:
        return None

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


# --------------------------------------------------------------------- #
#  napari entry point: napari_get_reader
# --------------------------------------------------------------------- #


def napari_get_reader(path: Union[PathLike, Sequence[PathLike]]):
    """
    Napari plugin hook: return a reader callable if this plugin can read `path`.

    This MUST return a function object (e.g. `reader_function`) or None.
    """
    # napari may pass a list/tuple or a single path
    first = str(path[0] if isinstance(path, (list, tuple)) else path).strip()

    if _looks_like_ome_source(first):
        return reader_function
    return None


# --------------------------------------------------------------------- #
#  Reader implementation: reader_function
# --------------------------------------------------------------------- #


def _read_one(src: str, mode: str) -> LayerData:
    """
    Read a single source into (data, add_kwargs, layer_type),
    obeying `mode` = 'image' or 'labels'.
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
        or looks_vortex
        or looks_tiff
    ):
        if looks_vortex:
            # Vortex needs ome-arrow's ingest helper to produce a typed scalar.
            scalar = _read_vortex_scalar(src)
            obj = OMEArrow(scalar)
        else:
            obj = OMEArrow(src)
        arr = obj.export(how="numpy", dtype=np.uint16)  # TCZYX
        info = obj.info()  # may contain 'shape': (T, C, Z, Y, X)

        # Recover from accidental 1D flatten
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

        return arr, add_kwargs, layer_type

    # ---- bare .npy fallback -----------------------------------------
    if looks_npy:
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


def reader_function(
    path: Union[PathLike, Sequence[PathLike]],
) -> list[LayerData]:
    """
    The actual reader callable napari will use.

    It reads one or more paths, prompting the user (or using the env var)
    to decide 'image' vs 'labels', and returns a list of LayerData tuples.
    """
    paths: list[str] = [
        str(p) for p in (path if isinstance(path, (list, tuple)) else [path])
    ]
    layers: list[LayerData] = []

    # Use the first path as context for the dialog label
    try:
        mode = _get_layer_mode(sample_path=paths[0])  # 'image' or 'labels'
    except RuntimeError as e:
        # If user canceled the dialog, propagate a clean error for napari
        raise ValueError(str(e)) from e

    for src in paths:
        try:
            parquet_layers = _read_parquet_rows(src, mode)
            if parquet_layers is not None:
                layers.extend(parquet_layers)
                continue
            vortex_layers = _read_vortex_rows(src, mode)
            if vortex_layers is not None:
                layers.extend(vortex_layers)
                continue
            layers.append(_read_one(src, mode=mode))
        except Exception as e:
            warnings.warn(
                f"Failed to read '{src}' with napari-ome-arrow: {e}",
                stacklevel=2,
            )

    if not layers:
        raise ValueError("No readable inputs found for given path(s).")
    return layers
