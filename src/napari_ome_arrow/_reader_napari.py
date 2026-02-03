"""Napari viewer helpers for the OME-Arrow reader."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _maybe_set_viewer_3d(arr: np.ndarray) -> None:
    """Switch the active napari viewer to 3D if the array has a Z-stack.

    Assumes OME-Arrow's TCZYX convention or a subset, i.e. Z is always the
    third-from-last axis. No-op if there's no active viewer.

    Args:
        arr: Array to inspect for a Z dimension.
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


def _as_labels(arr: np.ndarray) -> np.ndarray:
    """Convert an array into an integer label array.

    Args:
        arr: Input array.

    Returns:
        Array converted to an integer dtype suitable for labels.
    """
    if arr.dtype.kind == "f":
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.round(arr).astype(np.int32, copy=False)
    elif arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int32, copy=False)
    return arr


def _enable_grid(n_layers: int) -> None:
    """Switch the current viewer into grid view when possible.

    Args:
        n_layers: Number of layers to arrange in the grid.
    """
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


def _strip_channel_axis(
    arr: np.ndarray, add_kwargs: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strip the channel axis from an array if configured.

    Args:
        arr: Input array.
        add_kwargs: Layer kwargs that may include "channel_axis".

    Returns:
        Tuple of (array without channel axis, updated kwargs).
    """
    # Only strip when channel_axis is present and valid.
    channel_axis = add_kwargs.pop("channel_axis", None)
    if channel_axis is None:
        return arr, add_kwargs
    try:
        axis = int(channel_axis)
    except Exception:
        return arr, add_kwargs
    if arr.ndim <= axis:
        return arr, add_kwargs
    arr = np.take(arr, 0, axis=axis)
    return arr, add_kwargs
