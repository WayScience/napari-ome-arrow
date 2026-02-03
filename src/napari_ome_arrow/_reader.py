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

import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from ._reader_infer import (
    _infer_layer_mode_from_source,
    _looks_like_ome_source,
)
from ._reader_napari import _maybe_set_viewer_3d, _strip_channel_axis
from ._reader_omearrow import _read_one, _read_parquet_rows, _read_vortex_rows
from ._reader_stack import (
    _channel_names_from_pattern,
    _collect_stack_files,
    _infer_stack_scale_from_pattern,
    _parse_stack_scale,
    _prompt_stack_pattern,
    _prompt_stack_scale,
    _read_rgb_stack_pattern,
    _replace_channel_placeholder,
    _stack_default_dim_for_pattern,
)
from ._reader_types import LayerData, PathLike

# --------------------------------------------------------------------- #
#  Mode selection (env var + GUI prompt)
# --------------------------------------------------------------------- #


def _get_layer_mode(
    sample_path: str, *, image_type_hint: str | None = None
) -> str:
    """Decide whether to load as "image" or "labels".

    Priority:
    1. `NAPARI_OME_ARROW_LAYER_TYPE` env var (image/labels).
    2. If in a Qt GUI context, show a modal dialog asking the user.
    3. Otherwise, default to "image".

    Args:
        sample_path: Path used for prompt labeling.
        image_type_hint: Optional inferred mode from metadata.

    Returns:
        The selected mode, "image" or "labels".

    Raises:
        RuntimeError: If the environment variable is invalid or the user
            cancels the dialog.
    """
    # Env var override (used in headless/batch workflows).
    mode = os.environ.get("NAPARI_OME_ARROW_LAYER_TYPE")
    if mode is not None:
        mode = mode.lower()
        if mode in {"image", "labels"}:
            return mode
        raise RuntimeError(
            f"Invalid NAPARI_OME_ARROW_LAYER_TYPE={mode!r}; expected 'image' or 'labels'."
        )

    # Metadata hint (best effort).
    if image_type_hint in {"image", "labels"}:
        return image_type_hint

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
#  napari entry point: napari_get_reader
# --------------------------------------------------------------------- #


def napari_get_reader(
    path: PathLike | Sequence[PathLike],
) -> Callable[[PathLike | Sequence[PathLike]], list[LayerData]] | None:
    """Return a reader callable for napari if the path is supported.

    Args:
        path: A single path or a sequence of paths provided by napari.

    Returns:
        Reader callable if the path is supported, otherwise None.
    """
    # napari may pass a list/tuple or a single path
    first = str(path[0] if isinstance(path, (list, tuple)) else path).strip()

    if _looks_like_ome_source(first):
        return reader_function
    return None


# --------------------------------------------------------------------- #
#  Reader implementation: reader_function
# --------------------------------------------------------------------- #


def reader_function(
    path: PathLike | Sequence[PathLike],
) -> list[LayerData]:
    """Read one or more paths into napari layer data.

    The user may be prompted (or an env var used) to decide "image" vs
    "labels".

    Args:
        path: A single path or a sequence of paths provided by napari.

    Returns:
        List of layer tuples for napari.

    Raises:
        ValueError: If no readable inputs are found.
    """
    paths: list[str] = [
        str(p) for p in (path if isinstance(path, (list, tuple)) else [path])
    ]
    layers: list[LayerData] = []

    # Offer a stack pattern prompt when multiple compatible files are present.
    stack_selection = _collect_stack_files(paths)
    stack_pattern = None
    if stack_selection is not None:
        stack_pattern = _prompt_stack_pattern(*stack_selection)

    # Use the original first path as context for the dialog label
    mode_hint = None
    if stack_pattern is not None:
        stack_default_dim = _stack_default_dim_for_pattern(stack_pattern)
        mode_hint = _infer_layer_mode_from_source(
            stack_pattern, stack_default_dim=stack_default_dim
        )
    elif paths:
        mode_hint = _infer_layer_mode_from_source(paths[0])
    try:
        mode = _get_layer_mode(
            sample_path=paths[0], image_type_hint=mode_hint
        )  # 'image' or 'labels'
    except RuntimeError as e:
        # If user canceled the dialog, propagate a clean error for napari
        raise ValueError(str(e)) from e

    if stack_pattern is not None:
        try:
            channel_names = _channel_names_from_pattern(
                stack_pattern, stack_default_dim
            )
            resolved_stack_scale = None
            if mode == "image" and channel_names and len(channel_names) > 1:
                # Split each channel into separate layers for stack patterns.
                inferred_stack_scale = _infer_stack_scale_from_pattern(
                    stack_pattern, stack_default_dim
                )
                env_scale = os.environ.get("NAPARI_OME_ARROW_STACK_SCALE")
                if env_scale:
                    try:
                        resolved_stack_scale = _parse_stack_scale(env_scale)
                    except ValueError as exc:
                        warnings.warn(
                            f"Invalid NAPARI_OME_ARROW_STACK_SCALE '{env_scale}': {exc}.",
                            stacklevel=2,
                        )
                        resolved_stack_scale = None
                if (
                    resolved_stack_scale is None
                    and inferred_stack_scale is None
                ):
                    resolved_stack_scale = _prompt_stack_scale(
                        sample_path=stack_pattern, default_scale=None
                    )

                for label in channel_names:
                    channel_pattern = _replace_channel_placeholder(
                        stack_pattern, label, stack_default_dim
                    )
                    channel_default_dim = _stack_default_dim_for_pattern(
                        channel_pattern
                    )
                    try:
                        arr, add_kwargs, layer_type = _read_one(
                            channel_pattern,
                            mode=mode,
                            stack_default_dim=channel_default_dim,
                            stack_scale_override=resolved_stack_scale,
                        )
                        arr, add_kwargs = _strip_channel_axis(arr, add_kwargs)
                    except Exception as exc:
                        try:
                            arr, is_rgb = _read_rgb_stack_pattern(
                                channel_pattern
                            )
                        except Exception:
                            warnings.warn(
                                f"Failed to read channel '{label}' from stack "
                                f"pattern '{stack_pattern}': {exc}",
                                stacklevel=2,
                            )
                            continue
                        add_kwargs = {"name": Path(channel_pattern).name}
                        if is_rgb:
                            add_kwargs["rgb"] = True
                        layer_type = "image"

                    add_kwargs["name"] = (
                        f"{add_kwargs.get('name', label)}[{label}]"
                    )
                    if not add_kwargs.get("rgb"):
                        _maybe_set_viewer_3d(arr)
                    layers.append((arr, add_kwargs, layer_type))
            else:
                # Single-layer stack read.
                arr, add_kwargs, layer_type = _read_one(
                    stack_pattern,
                    mode=mode,
                    stack_default_dim=stack_default_dim,
                )
                layers.append((arr, add_kwargs, layer_type))
        except Exception as e:
            warnings.warn(
                f"Failed to read stack pattern '{stack_pattern}': {e}. "
                "Loading files individually instead.",
                stacklevel=2,
            )
        else:
            return layers
    elif stack_selection is not None and len(paths) == 1:
        paths = [str(p) for p in stack_selection[0]]

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
