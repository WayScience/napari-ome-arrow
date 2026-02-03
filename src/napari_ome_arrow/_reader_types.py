"""Shared reader type aliases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, Path]
LayerData = tuple[np.ndarray, dict[str, Any], str]
