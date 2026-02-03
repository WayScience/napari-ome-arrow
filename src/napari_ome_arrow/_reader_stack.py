"""Stack pattern and scale helpers for the OME-Arrow reader."""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ome_arrow.core import OMEArrow
from ome_arrow.ingest import from_stack_pattern_path


def _collect_stack_files(
    paths: Sequence[str],
) -> tuple[list[Path], Path] | None:
    """Collect stack candidates and their shared folder if applicable.

    Args:
        paths: Input paths passed by napari.

    Returns:
        A tuple of (files, folder) or None if stack detection fails.
    """
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
    """Suggest a stack pattern string for a set of files.

    Args:
        files: Files to analyze for a stack pattern.
        folder: Folder that contains the files.

    Returns:
        Suggested stack pattern string.
    """
    if not files:
        return str(folder / ".*")

    # Prefer the most common suffix to avoid mixing file types.
    suffix_counts = Counter(p.suffix.lower() for p in files)
    preferred_suffix = (
        suffix_counts.most_common(1)[0][0] if suffix_counts else ""
    )
    candidates = [
        p for p in files if p.suffix.lower() == preferred_suffix
    ] or list(files)
    names = [p.name for p in candidates]

    def _unique_in_order(values: Sequence[str]) -> list[str]:
        # Preserve original ordering for channel tokens.
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

        # Separate channel tokens from Z tokens when possible.
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

    # Generic numeric token scanning when no explicit Z token exists.
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
    """Infer the default dimension token for a stack pattern.

    Args:
        pattern: Stack pattern string.

    Returns:
        Default dimension token ("Z" or "C").
    """
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
    """Detect a dimension token from text preceding a placeholder.

    Args:
        before_text: Text before the placeholder in the pattern.

    Returns:
        Dimension token ("C", "T", "Z", "S") or None if not detected.
    """
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
    """Extract channel names from a stack pattern if available.

    Args:
        pattern: Stack pattern string.
        default_dim_for_unspecified: Default dimension token to use.

    Returns:
        Channel names if a channel placeholder is found, otherwise None.
    """
    # Matches numeric ranges like "000-015" or "1-10:2" inside <...>.
    num_range = re.compile(r"^(?P<a>\d+)\-(?P<b>\d+)(?::(?P<step>\d+))?$")

    def parse_choices(raw: str) -> list[str] | None:
        # Support comma lists or numeric ranges within angle brackets.
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
    """Replace the first channel placeholder in a pattern.

    Args:
        pattern: Stack pattern string.
        channel_value: Channel value to insert.
        default_dim_for_unspecified: Default dimension token to use.

    Returns:
        Pattern string with the channel placeholder replaced.
    """
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
    """List files that match a stack pattern.

    Args:
        pattern: Stack pattern string.

    Returns:
        Files matching the pattern, sorted by Z-like tokens when possible.
    """
    p = Path(pattern)
    folder = p.parent
    name = p.name
    # Convert <...> placeholders to a wildcard regex.
    regex = re.escape(name)
    regex = re.sub(r"\\<[^>]+\\>", r"[^/]+", regex)
    compiled = re.compile(f"^{regex}$")
    candidates = [f for f in folder.iterdir() if f.is_file()]
    matched = [f for f in candidates if compiled.match(f.name)]

    def z_key(path: Path) -> tuple[int, str]:
        # Prefer Z-like numeric ordering when present.
        match = re.search(r"Z[S]?(\d+)", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return sorted(matched, key=z_key)


def _read_rgb_stack_pattern(pattern: str) -> tuple[np.ndarray, bool]:
    """Read a stack pattern using RGB-aware fallbacks.

    Args:
        pattern: Stack pattern string.

    Returns:
        Tuple of (stack array, is_rgb).

    Raises:
        ImportError: If required optional dependencies are missing.
        FileNotFoundError: If no files match the pattern.
        ValueError: If frame shapes are unsupported or inconsistent.
    """
    try:
        from bioio import BioImage
        from bioio_ome_tiff import Reader as OMEReader
        from bioio_tifffile import Reader as TiffReader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RGB stack fallback requires bioio and bioio_tifffile."
        ) from exc

    # Try to load a stack using image readers that can report RGB frames.
    files = _files_from_pattern(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    shape_ref: tuple[int, ...] | None = None

    def _read_frame(fpath: Path) -> tuple[np.ndarray, bool]:
        # Normalize RGB channel placement to last axis.
        reader = (
            OMEReader
            if fpath.suffix.lower() in (".ome.tif", ".ome.tiff")
            else TiffReader
        )
        img = BioImage(image=str(fpath), reader=reader)
        arr = np.asarray(img.data)

        if arr.ndim == 2:
            return arr, False
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                return arr, True
            if arr.shape[0] in (3, 4):
                return np.moveaxis(arr, 0, -1), True
            raise ValueError(
                f"Unsupported 3D frame shape {arr.shape} for {fpath.name}"
            )
        raise ValueError(
            f"Unsupported frame dimensions {arr.shape} for {fpath.name}"
        )

    with ThreadPoolExecutor(
        max_workers=min(8, max(1, len(files)))
    ) as executor:
        results = list(executor.map(_read_frame, files))

    frames: list[np.ndarray] = []
    rgb_flags: list[bool] = []
    for frame, is_rgb in results:
        if shape_ref is None:
            shape_ref = frame.shape
        elif frame.shape != shape_ref:
            raise ValueError(
                f"Shape mismatch for stack: {frame.shape} vs {shape_ref}"
            )
        frames.append(frame)
        rgb_flags.append(is_rgb)

    rgb = any(rgb_flags)

    stack = np.stack(frames, axis=0)
    return stack, rgb


def _prompt_stack_pattern(files: Sequence[Path], folder: Path) -> str | None:
    """Prompt for a stack pattern when multiple files are detected.

    Args:
        files: Stack candidate files.
        folder: Folder containing the files.

    Returns:
        A pattern string or None to skip stack parsing.
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
    """Parse a stack scale string into numeric values.

    Args:
        text: Comma- or space-separated scale values.

    Returns:
        Tuple of scale values.

    Raises:
        ValueError: If the scale is invalid or has the wrong length.
    """
    # Accept comma- or space-delimited values.
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if len(tokens) not in (3, 5):
        raise ValueError("Expected 3 values (Z,Y,X) or 5 values (T,C,Z,Y,X).")
    values = tuple(float(v) for v in tokens)
    if any(v <= 0 for v in values):
        raise ValueError("Scale values must be positive.")
    return values


def _format_stack_scale(values: Sequence[float]) -> str:
    """Format scale values as a comma-separated string.

    Args:
        values: Scale values.

    Returns:
        String representation of the scale values.
    """
    return ",".join(f"{v:g}" for v in values)


def _scale_from_ome_arrow(
    obj: OMEArrow,
) -> tuple[float, float, float] | None:
    """Extract Z/Y/X scale from OME-Arrow metadata.

    Args:
        obj: OMEArrow instance to inspect.

    Returns:
        (z, y, x) scale tuple, or None if unavailable or defaulted to 1.0.
    """
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
    """Prompt for stack scale values when in a Qt context.

    Args:
        sample_path: Path used for prompt labeling.
        default_scale: Default scale to prefill or return on cancel.

    Returns:
        Scale tuple or None if no override is provided.
    """
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


@lru_cache(maxsize=128)
def _infer_stack_scale_from_pattern(
    pattern: str, stack_default_dim: str | None
) -> tuple[float, float, float] | None:
    """Infer scale from a stack pattern using OME-Arrow metadata.

    Args:
        pattern: Stack pattern string.
        stack_default_dim: Default dimension token for the stack.

    Returns:
        (z, y, x) scale tuple, or None if inference fails.
    """
    try:
        # Use ome-arrow ingestion to infer physical size from a stack.
        scalar = from_stack_pattern_path(
            pattern,
            default_dim_for_unspecified=stack_default_dim,
            map_series_to="T",
            clamp_to_uint16=True,
        )
        obj = OMEArrow(scalar)
    except Exception:
        return None
    return _scale_from_ome_arrow(obj)


def _normalize_stack_scale(scale: Sequence[float]) -> tuple[float, ...]:
    """Normalize a stack scale into TCZYX order.

    Args:
        scale: Scale values in Z,Y,X or T,C,Z,Y,X order.

    Returns:
        Normalized scale values in T,C,Z,Y,X order.

    Raises:
        ValueError: If the input has an unsupported length.
    """
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
    """Compute the scale tuple appropriate for a specific array and mode.

    Args:
        arr: Data array to be displayed.
        mode: "image" or "labels".
        add_kwargs: Layer kwargs, possibly including "channel_axis".
        scale: Scale values in Z,Y,X or T,C,Z,Y,X order.

    Returns:
        Scale tuple aligned to the array's dimensionality, or None.
    """
    # Normalize scale to TCZYX, then trim per array dimensionality.
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
