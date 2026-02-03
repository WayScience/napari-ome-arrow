# napari-ome-arrow

[![License BSD-3](https://img.shields.io/pypi/l/napari-ome-arrow.svg?color=green)](https://github.com/wayscience/napari-ome-arrow/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-ome-arrow.svg?color=green)](https://pypi.org/project/napari-ome-arrow)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-ome-arrow.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-ome-arrow)](https://napari-hub.org/plugins/napari-ome-arrow)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Software DOI badge](https://zenodo.org/badge/DOI/10.5281/zenodo.17613571.svg)](https://doi.org/10.5281/zenodo.17613571)

`napari-ome-arrow` is a minimal plugin for [napari](https://napari.org) that opens image data through the [OME-Arrow](https://github.com/wayscience/ome-arrow) toolkit.

It provides a single, explicit pathway for loading OME-style bioimage data:

- **OME-TIFF** (`.ome.tif`, `.ome.tiff`, `.tif`, `.tiff`)
- **OME-Zarr** (`.ome.zarr`, `.zarr` stores and URLs)
- **OME-Parquet** (`.ome.parquet`, `.parquet`, `.pq`)
- **OME-Vortex** (`.ome.vortex`, `.vortex`)
- **Bio-Formats–style stack patterns** (paths containing `<`, `>`, or `*`)
- A simple **`.npy` fallback** for quick testing / ad-hoc arrays

## Key features

- ✅ **Unified reader via OMEArrow**
  All supported formats are loaded through `OME-Arrow`, which normalizes data into a common **TCZYX**-like representation.

- ✅ **Explicit image vs labels mode**
  This plugin never guesses whether your data are intensities or segmentation masks. You must tell it:

  - via the GUI prompt when you drop/open a file in napari, or
  - via an environment variable for scripted/CLI usage.

- ✅ **Interactive choice in the GUI**
  When `NAPARI_OME_ARROW_LAYER_TYPE` is not set and you open a supported file, napari shows a small dialog:

  > *How should `my_data.ome.tif` be loaded?*
  > `[Image]   [Labels]   [Cancel]`

  This makes the “image vs labels” choice explicit at load time without relying on file naming conventions.

- ✅ **Image mode**

  - Returns a napari **image layer**
  - Preserves channels and sets `channel_axis` when appropriate
    (e.g. multi-channel OME-TIFF or stack patterns)
  - Works for 2D, 3D (Z-stacks), and higher-dimensional data (T, C, Z, Y, X)

- ✅ **Labels mode**

  - Returns a napari **labels layer**
  - Converts data to an integer dtype (suitable for labels)
  - Applies a reasonable default opacity for overlaying on images

- ✅ **Automatic 3D for Z-stacks**
  If the loaded data include a true Z dimension (`Z > 1`, assuming a TCZYX subset), the plugin asks the current viewer to switch to **3D** (`viewer.dims.ndisplay = 3`) so z-stacks open directly in volume mode.

- ✅ **Headless / scripted friendly**
  When Qt is not available (e.g., in headless or purely programmatic contexts), the reader:

  - respects `NAPARI_OME_ARROW_LAYER_TYPE`, and
  - defaults to `"image"` if the variable is not set.

- ✅ **Grid view for multi-row OME-Parquet / OME-Vortex**
  When a Parquet or Vortex file contains multiple OME-Arrow rows, each row is loaded as its own layer and the viewer is switched to napari’s grid mode. Set `NAPARI_OME_ARROW_PARQUET_COLUMN` or `NAPARI_OME_ARROW_VORTEX_COLUMN` to pick which image column to visualize.

- ✅ **Stack scale prompt + override**
  When loading image stacks (stack patterns), napari prompts for voxel spacing only if no scale metadata or `NAPARI_OME_ARROW_STACK_SCALE` override is present and a Qt UI is available (headless runs skip the prompt). To avoid the prompt, set `NAPARI_OME_ARROW_STACK_SCALE` using `Z,Y,X` or `T,C,Z,Y,X`.

______________________________________________________________________

This [napari] plugin was generated with [copier] using the [napari-plugin-template] (None).

## Installation

You can install `napari-ome-arrow` via [pip]:

```
pip install napari-ome-arrow
```

If napari is not already installed, you can install `napari-ome-arrow` with napari and Qt via:

```
pip install "napari-ome-arrow[all]"
```

To install latest development version :

```
pip install git+https://github.com/wayscience/napari-ome-arrow.git
```

To enable OME-Vortex support via OME-Arrow, install the optional extra:

```bash
pip install "napari-ome-arrow[vortex]"
```

## Usage

### From the napari GUI

1. Install the plugin (see above).
1. Start napari.
1. Drag and drop an OME-TIFF, OME-Zarr, OME-Parquet, OME-Vortex file, a stack pattern, or a multi-select stack (e.g. `img_000.tif` ... `img_123.tif`) into the viewer.
1. When prompted, choose **Image** or **Labels**.

The plugin will:

- load the data through `OMEArrow`,
- map channels and axes appropriately, and
- automatically switch to 3D if there is a Z-stack.
- When multiple files look like a numbered stack, treat them as a single stack rather than independent layers.

### From the command line

You can control the mode via an environment variable:

```bash
# Load as regular images
NAPARI_OME_ARROW_LAYER_TYPE=image napari my_data.ome.tif

# Load as labels (segmentation)
NAPARI_OME_ARROW_LAYER_TYPE=labels napari my_labels.ome.parquet

# Pick a specific column in a multi-row OME-Parquet and show in grid mode
NAPARI_OME_ARROW_LAYER_TYPE=image \\
NAPARI_OME_ARROW_PARQUET_COLUMN=Image_FileName_OrigDNA_OMEArrow_ORIG \\
napari tests/data/cytodataframe/BR00117006.ome.parquet

# Prefill stack voxel spacing for stack patterns (Z,Y,X or T,C,Z,Y,X)
NAPARI_OME_ARROW_STACK_SCALE=1.0,0.108,0.108 napari "stack/z<000-120>.tif"

# Prefill stack voxel spacing for multi-file stacks (use a pattern or glob on CLI)
NAPARI_OME_ARROW_STACK_SCALE=1.0,0.108,0.108 napari "stack/img_<000-120>.tif"
```

## Contributing

Contributions are very welcome.
Please reference our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## License

Please see the [LICENSE](LICENSE) file for more information.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[copier]: https://copier.readthedocs.io/en/stable/
[file an issue]: https://github.com/wayscience/napari-ome-arrow/issues
[napari]: https://github.com/napari/napari
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[pip]: https://pypi.org/project/pip/
