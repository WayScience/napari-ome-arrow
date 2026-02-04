"""
Init for napari_ome_arrow
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._writer import napari_write_image, napari_write_labels

__all__ = [
    "__version__",
    "napari_get_reader",
    "napari_write_image",
    "napari_write_labels",
]
