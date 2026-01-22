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
import re
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pyarrow as pa
from ome_arrow.core import OMEArrow
from ome_arrow.ingest import from_stack_pattern_path
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


def _normalize_image_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"image", "intensity", "raw"} or "image" in text:
        return "image"
    if (
        text in {"labels", "label", "mask", "masks", "segmentation", "seg"}
        or "label" in text
        or "mask" in text
        or "seg" in text
    ):
        return "labels"
    return None


def _infer_layer_mode_from_record(record: dict[str, Any]) -> str | None:
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


def _get_layer_mode(
    sample_path: str, *, image_type_hint: str | None = None
) -> str:
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
    looks_dir_stack = False
    if p.exists() and p.is_dir() and not looks_zarr:
        try:
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


def _collect_stack_files(
    paths: Sequence[str],
) -> tuple[list[Path], Path] | None:
    """Return stack candidate files and their shared folder, if any."""
    if any(any(c in p for c in "<>*") for p in paths):
        return None

    if len(paths) == 1:
        candidate = Path(paths[0])
        if (
            candidate.exists()
            and candidate.is_dir()
            and candidate.suffix.lower() not in {".zarr", ".ome.zarr"}
        ):
            files = sorted(p for p in candidate.iterdir() if p.is_file())
            if len(files) > 1:
                return files, candidate
        return None

    path_objs = [Path(p) for p in paths]
    if not all(p.exists() and p.is_file() for p in path_objs):
        return None

    parents = {p.parent for p in path_objs}
    if len(parents) != 1:
        return None

    return sorted(path_objs), parents.pop()


def _suggest_stack_pattern(files: Sequence[Path], folder: Path) -> str:
    """Suggest a Bio-Formats-style stack pattern string for files."""
    if not files:
        return str(folder / ".*")

    suffix_counts = Counter(p.suffix.lower() for p in files)
    preferred_suffix = (
        suffix_counts.most_common(1)[0][0] if suffix_counts else ""
    )
    candidates = [
        p for p in files if p.suffix.lower() == preferred_suffix
    ] or list(files)
    names = [p.name for p in candidates]

    def _unique_in_order(values: Sequence[str]) -> list[str]:
        seen = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _suggest_from_z_token(values: Sequence[str]) -> str | None:
        # Look for a Z token like "Z000" or "ZS000" and build a range.
        z_re = re.compile(
            r"^(?P<pre>.*?)(?P<zprefix>Z[S]?)(?P<znum>\d+)(?P<post>.*)$"
        )
        matches = [z_re.match(v) for v in values]
        if not all(m is not None for m in matches):
            return None

        zprefixes = {m.group("zprefix") for m in matches}  # type: ignore[union-attr]
        if len(zprefixes) != 1:
            return None
        zprefix = next(iter(zprefixes))

        pre_parts = [m.group("pre") for m in matches]  # type: ignore[union-attr]
        post_parts = [m.group("post") for m in matches]  # type: ignore[union-attr]
        if len(set(post_parts)) != 1:
            return None
        post = post_parts[0]

        z_values = [m.group("znum") for m in matches]  # type: ignore[union-attr]
        z_width = max(len(v) for v in z_values)
        z_nums = [int(v) for v in z_values]
        z_unique = sorted(set(z_nums))
        if len(z_unique) == (max(z_unique) - min(z_unique) + 1):
            z_token = f"<{str(min(z_unique)).zfill(z_width)}-{str(max(z_unique)).zfill(z_width)}>"
        else:
            z_token = (
                "<" + ",".join(str(v).zfill(z_width) for v in z_unique) + ">"
            )

        pre_prefix = os.path.commonprefix(pre_parts)
        pre_suffix = os.path.commonprefix([p[::-1] for p in pre_parts])[::-1]
        if pre_prefix:
            middle_parts = [p[len(pre_prefix) :] for p in pre_parts]
        else:
            middle_parts = pre_parts
        if pre_suffix:
            middle_parts = [
                p[: -len(pre_suffix)] if p.endswith(pre_suffix) else p
                for p in middle_parts
            ]
        channel_values = _unique_in_order(middle_parts)
        channel_token = ""
        if len(set(channel_values)) > 1:
            channel_token = "<" + ",".join(channel_values) + ">"
        elif channel_values and channel_values[0]:
            channel_token = channel_values[0]

        suggested = (
            f"{pre_prefix}{channel_token}{pre_suffix}{zprefix}{z_token}{post}"
        )
        return suggested

    suggestion = _suggest_from_z_token(names)
    if suggestion is not None:
        return str(folder / suggestion)

    split_names = [re.split(r"(\d+)", name) for name in names]
    if split_names and all(
        len(parts) == len(split_names[0]) for parts in split_names
    ):
        static_ok = True
        for idx in range(0, len(split_names[0]), 2):
            token = split_names[0][idx]
            if any(parts[idx] != token for parts in split_names):
                static_ok = False
                break
        if static_ok:
            pattern_parts: list[str] = []
            for idx, token in enumerate(split_names[0]):
                if idx % 2 == 0:
                    pattern_parts.append(token)
                    continue
                values = [parts[idx] for parts in split_names]
                unique = sorted(set(values), key=lambda v: int(v))
                if len(unique) == 1:
                    pattern_parts.append(unique[0])
                    continue
                width = max(len(v) for v in unique)
                nums = [int(v) for v in unique]
                if len(unique) == (max(nums) - min(nums) + 1):
                    start = str(min(nums)).zfill(width)
                    end = str(max(nums)).zfill(width)
                    pattern_parts.append(f"<{start}-{end}>")
                else:
                    values_str = ",".join(str(n).zfill(width) for n in nums)
                    pattern_parts.append(f"<{values_str}>")
            pattern_name = "".join(pattern_parts)
            return str(folder / pattern_name)

    matches = [re.search(r"(\d+)(?!.*\d)", name) for name in names]
    if all(m is not None for m in matches):
        prefix = names[0][: matches[0].start()]
        suffix = names[0][matches[0].end() :]
        if all(
            name[: m.start()] == prefix and name[m.end() :] == suffix
            for name, m in zip(names, matches, strict=False)
        ):
            nums = [int(m.group(1)) for m in matches if m is not None]
            width = max(len(m.group(1)) for m in matches if m is not None)
            start = str(min(nums)).zfill(width)
            end = str(max(nums)).zfill(width)
            pattern_name = f"{prefix}<{start}-{end}>{suffix}"
            return str(folder / pattern_name)

    common_prefix = os.path.commonprefix(names)
    common_suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
    prefix = re.escape(common_prefix)
    suffix = re.escape(common_suffix)
    pattern_name = f"{prefix}.*{suffix}"
    return str(folder / pattern_name)


def _stack_default_dim_for_pattern(pattern: str) -> str:
    dim_tokens = {"z", "zs", "sec", "fp", "focal", "focalplane"}
    for idx, ch in enumerate(pattern):
        if ch != "<":
            continue
        before = pattern[:idx]
        match = re.search(r"([A-Za-z]+)$", before)
        if not match:
            continue
        token = match.group(1).lower()
        if token in dim_tokens:
            return "C"
    return "Z"


def _detect_dim_token(before_text: str) -> str | None:
    match = re.search(r"([A-Za-z]+)$", before_text)
    if not match:
        return None
    token = match.group(1).lower()
    if token in {"c", "ch", "w", "wavelength"}:
        return "C"
    if token in {"t", "tl", "tp", "timepoint"}:
        return "T"
    if token in {"z", "zs", "sec", "fp", "focal", "focalplane"}:
        return "Z"
    if token in {"s", "sp", "series"}:
        return "S"
    return None


def _channel_names_from_pattern(
    pattern: str, default_dim_for_unspecified: str
) -> list[str] | None:
    num_range = re.compile(r"^(?P<a>\d+)\-(?P<b>\d+)(?::(?P<step>\d+))?$")

    def parse_choices(raw: str) -> list[str] | None:
        raw = raw.strip()
        if not raw:
            return None
        if "," in raw and not num_range.match(raw):
            return [p.strip() for p in raw.split(",") if p.strip()]
        match = num_range.match(raw)
        if match:
            a, b = match.group("a"), match.group("b")
            step = int(match.group("step") or "1")
            start, stop = int(a), int(b)
            if stop < start:
                return None
            width = max(len(a), len(b))
            return [str(v).zfill(width) for v in range(start, stop + 1, step)]
        return None

    i = 0
    while i < len(pattern):
        if pattern[i] != "<":
            i += 1
            continue
        j = pattern.find(">", i + 1)
        if j == -1:
            break
        raw_inside = pattern[i + 1 : j]
        dim = _detect_dim_token(pattern[:i]) or default_dim_for_unspecified
        if dim.upper() == "C":
            choices = parse_choices(raw_inside)
            if choices:
                return choices
        i = j + 1
    return None


def _replace_channel_placeholder(
    pattern: str, channel_value: str, default_dim_for_unspecified: str
) -> str:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(pattern):
        if pattern[i] != "<":
            out.append(pattern[i])
            i += 1
            continue
        j = pattern.find(">", i + 1)
        if j == -1:
            out.append(pattern[i:])
            break
        raw_inside = pattern[i + 1 : j]
        dim = _detect_dim_token(pattern[:i]) or default_dim_for_unspecified
        if not replaced and dim.upper() == "C":
            out.append(channel_value)
            replaced = True
        else:
            out.append(f"<{raw_inside}>")
        i = j + 1
    return "".join(out)


def _files_from_pattern(pattern: str) -> list[Path]:
    p = Path(pattern)
    folder = p.parent
    name = p.name
    regex = re.escape(name)
    regex = re.sub(r"\\<[^>]+\\>", r"[^/]+", regex)
    compiled = re.compile(f"^{regex}$")
    candidates = [f for f in folder.iterdir() if f.is_file()]
    matched = [f for f in candidates if compiled.match(f.name)]

    def z_key(path: Path) -> tuple[int, str]:
        match = re.search(r"Z[S]?(\d+)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return sorted(matched, key=z_key)


def _read_rgb_stack_pattern(pattern: str) -> tuple[np.ndarray, bool]:
    try:
        from bioio import BioImage
        from bioio_ome_tiff import Reader as OMEReader
        from bioio_tifffile import Reader as TiffReader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RGB stack fallback requires bioio and bioio_tifffile."
        ) from exc

    files = _files_from_pattern(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    frames: list[np.ndarray] = []
    rgb = False
    shape_ref: tuple[int, ...] | None = None

    for fpath in files:
        reader = (
            OMEReader
            if fpath.suffix.lower() in (".ome.tif", ".ome.tiff")
            else TiffReader
        )
        img = BioImage(image=str(fpath), reader=reader)
        arr = np.asarray(img.data)

        if arr.ndim == 2:
            frame = arr
        elif arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                frame = arr
                rgb = True
            elif arr.shape[0] in (3, 4):
                frame = np.moveaxis(arr, 0, -1)
                rgb = True
            else:
                raise ValueError(
                    f"Unsupported 3D frame shape {arr.shape} for {fpath.name}"
                )
        else:
            raise ValueError(
                f"Unsupported frame dimensions {arr.shape} for {fpath.name}"
            )

        if shape_ref is None:
            shape_ref = frame.shape
        elif frame.shape != shape_ref:
            raise ValueError(
                f"Shape mismatch for {fpath.name}: {frame.shape} vs {shape_ref}"
            )
        frames.append(frame)

    stack = np.stack(frames, axis=0)
    return stack, rgb


def _strip_channel_axis(
    arr: np.ndarray, add_kwargs: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
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


def _prompt_stack_pattern(files: Sequence[Path], folder: Path) -> str | None:
    """
    Prompt for a stack pattern when multiple files are detected.

    Returns a pattern string or None to skip stack parsing.
    """
    suggested = _suggest_stack_pattern(files, folder)

    try:
        from qtpy import QtWidgets
    except Exception:
        warnings.warn(
            "Multiple files detected but Qt is not available; "
            "loading files individually.",
            stacklevel=2,
        )
        return None

    app = QtWidgets.QApplication.instance()
    if app is None:
        warnings.warn(
            "Multiple files detected but no QApplication instance; "
            "loading files individually.",
            stacklevel=2,
        )
        return None

    label = (
        "Multiple files detected. Enter a stack pattern string to load as a 3D stack.\n"
        "Use <...> for indices (e.g. z<000-120> or c<111,222>), or a regex like .* for non-numbered files.\n"
        "If no dimension token (z/c/t) is present, Z is assumed for this stack.\n\n"
        "Edit if needed and press OK, or Cancel to load files individually."
    )
    text, ok = QtWidgets.QInputDialog.getText(
        None,
        "napari-ome-arrow: stack pattern",
        label,
        text=suggested,
    )
    if not ok:
        return None
    value = text.strip()
    return value or None


def _parse_stack_scale(text: str) -> tuple[float, ...]:
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if len(tokens) not in (3, 5):
        raise ValueError("Expected 3 values (Z,Y,X) or 5 values (T,C,Z,Y,X).")
    values = tuple(float(v) for v in tokens)
    if any(v <= 0 for v in values):
        raise ValueError("Scale values must be positive.")
    return values


def _format_stack_scale(values: Sequence[float]) -> str:
    return ",".join(f"{v:g}" for v in values)


def _scale_from_ome_arrow(
    obj: OMEArrow,
) -> tuple[float, float, float] | None:
    try:
        record = obj.data.as_py()
        pixels_meta = record.get("pixels_meta", {}) if record else {}
        if not any(
            k in pixels_meta
            for k in (
                "physical_size_z",
                "physical_size_y",
                "physical_size_x",
            )
        ):
            return None
        z = float(pixels_meta.get("physical_size_z") or 1.0)
        y = float(pixels_meta.get("physical_size_y") or 1.0)
        x = float(pixels_meta.get("physical_size_x") or 1.0)
        if z == 1.0 and y == 1.0 and x == 1.0:
            return None
        return z, y, x
    except Exception:
        return None


def _prompt_stack_scale(
    sample_path: str, default_scale: Sequence[float] | None
) -> tuple[float, ...] | None:
    try:
        from qtpy import QtWidgets
    except Exception:
        return default_scale

    app = QtWidgets.QApplication.instance()
    if app is None:
        return default_scale

    default_text = (
        _format_stack_scale(default_scale) if default_scale else "1.0,1.0,1.0"
    )
    detected_note = f"Detected: {default_text}\n" if default_scale else ""
    label = (
        f"Enter voxel spacing for '{Path(sample_path).name}' in microns.\n"
        "Format: Z,Y,X (or T,C,Z,Y,X for full axis order).\n"
        f"{detected_note}"
        "Leave blank to keep the metadata/default values.\n"
        "You can also set NAPARI_OME_ARROW_STACK_SCALE to prefill this."
    )
    text, ok = QtWidgets.QInputDialog.getText(
        None,
        "napari-ome-arrow: stack scale",
        label,
        text=default_text,
    )
    if not ok:
        return default_scale
    value = text.strip()
    if not value:
        return default_scale
    try:
        return _parse_stack_scale(value)
    except ValueError as exc:
        warnings.warn(
            f"Invalid stack scale '{value}': {exc}. Using defaults instead.",
            stacklevel=2,
        )
        return default_scale


def _normalize_stack_scale(scale: Sequence[float]) -> tuple[float, ...]:
    if len(scale) == 3:
        z, y, x = scale
        return (1.0, 1.0, z, y, x)
    if len(scale) == 5:
        return tuple(scale)
    raise ValueError("Stack scale must have 3 or 5 values.")


def _scale_for_array(
    arr: np.ndarray,
    mode: str,
    add_kwargs: dict[str, Any],
    scale: Sequence[float],
) -> tuple[float, ...] | None:
    scale_tczyx = _normalize_stack_scale(scale)
    channel_axis = add_kwargs.get("channel_axis")
    if arr.ndim == 5:
        if channel_axis is not None:
            # Drop the channel axis; napari splits channels before applying scale.
            return (
                scale_tczyx[0],
                scale_tczyx[2],
                scale_tczyx[3],
                scale_tczyx[4],
            )
        return scale_tczyx
    if arr.ndim == 4:
        if mode == "image" and channel_axis is not None:
            if channel_axis == 0:
                return (scale_tczyx[2], scale_tczyx[3], scale_tczyx[4])
            return (
                scale_tczyx[1],
                scale_tczyx[2],
                scale_tczyx[3],
                scale_tczyx[4],
            )
        return (
            scale_tczyx[0],
            scale_tczyx[2],
            scale_tczyx[3],
            scale_tczyx[4],
        )
    if arr.ndim == 3:
        return (scale_tczyx[2], scale_tczyx[3], scale_tczyx[4])
    if arr.ndim == 2:
        return (scale_tczyx[3], scale_tczyx[4])
    return None


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


def _read_one(
    src: str, mode: str, *, stack_default_dim: str | None = None
) -> LayerData:
    """
    Read a single source into (data, add_kwargs, layer_type),
    obeying `mode` = 'image' or 'labels'.

    If `stack_default_dim` is provided and `src` is a stack pattern,
    placeholders without explicit tokens will be treated as that dimension.
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

        if looks_stack:
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
            add_kwargs.setdefault("scale", scale)
            if scale is not None:
                add_kwargs["scale"] = scale

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
            if mode == "image" and channel_names and len(channel_names) > 1:
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
