# napari-ome-arrow

[![License BSD-3](https://img.shields.io/pypi/l/napari-ome-arrow.svg?color=green)](https://github.com/wayscience/napari-ome-arrow/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-ome-arrow.svg?color=green)](https://pypi.org/project/napari-ome-arrow)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-ome-arrow.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-ome-arrow)](https://napari-hub.org/plugins/napari-ome-arrow)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.17613571.svg)](https://doi.org/10.5281/zenodo.17613571)

`napari-ome-arrow` opens OME image data in [napari] using the
[OME-Arrow](https://github.com/wayscience/ome-arrow) library.

## Supported inputs

- Typed OME-Arrow dataset directories (`*.ome-arrow`)
- Nested OME-Arrow tables in Parquet (`*.ome.parquet`, `*.parquet`, `*.pq`)
- OME-Vortex files (`*.ome.vortex`, `*.vortex`)
- OME-TIFF and TIFF files (`*.ome.tif`, `*.ome.tiff`, `*.tif`, `*.tiff`)
- OME-Zarr stores (`*.ome.zarr`, `*.zarr`)
- Numbered image stacks and stack patterns containing `<`, `>`, or `*`
- NumPy arrays (`*.npy`)

Multi-image OME-Arrow datasets and multi-row nested tables are loaded as
multiple napari layers.

## Installation

Python 3.11 or newer is required.

Install the plugin into an existing napari environment:

```bash
pip install napari-ome-arrow
```

Or install napari with a Qt backend at the same time:

```bash
pip install "napari-ome-arrow[pyqt6]"
```

OME-Vortex support requires one additional extra:

```bash
pip install "napari-ome-arrow[vortex]"
```

## Usage

Open napari, then drag a supported file or directory into the viewer. You can
also start napari with a path:

```bash
napari sample.ome.parquet
napari images.ome-arrow
```

The plugin loads data as either image or labels layers. It chooses the layer
type in this order:

1. `NAPARI_OME_ARROW_LAYER_TYPE`, when set to `image` or `labels`.
1. OME-Arrow `image_type` metadata, when available.
1. A napari prompt.
1. `image` when running without a Qt application.

Set the mode explicitly for scripts or repeatable commands:

```bash
NAPARI_OME_ARROW_LAYER_TYPE=labels napari segmentation.ome.parquet
```

For a numbered stack, select the files together or provide a pattern:

```bash
napari "stack/z<000-120>.tif"
```

## Configuration

| Environment variable              | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `NAPARI_OME_ARROW_LAYER_TYPE`     | Load as `image` or `labels`                  |
| `NAPARI_OME_ARROW_PARQUET_COLUMN` | Select an OME-Arrow struct column in Parquet |
| `NAPARI_OME_ARROW_VORTEX_COLUMN`  | Select an OME-Arrow struct column in Vortex  |
| `NAPARI_OME_ARROW_STACK_SCALE`    | Set stack spacing as `Z,Y,X` or `T,C,Z,Y,X`  |

Multiple rows are displayed in napari's grid view. Image stacks with a real Z
dimension open in 3D. If stack spacing is missing, the plugin can prompt for it
when a Qt application is available.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and test instructions.

## License

This project uses the BSD 3-Clause License. See [LICENSE](LICENSE).

Report problems through the [issue tracker].

[issue tracker]: https://github.com/wayscience/napari-ome-arrow/issues
[napari]: https://napari.org
