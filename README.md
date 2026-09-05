# trs-viewer

<p align="center">
  <img src="docs/logo.svg" alt="trs-viewer logo" width="320"/>
</p>

Interactive power-trace viewer and side-channel analysis toolkit for Riscure `.trs` files and NumPy trace sets.

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Qt6](https://img.shields.io/badge/Qt-6%20%7C%205-green)

![Trace browser and pipeline](docs/trace_browser.gif)

Open multi-gigabyte trace sets without loading them into RAM, build a processing
pipeline on top of them, and run the standard side-channel analyses over the
result — each in its own tab, several files at a time.

- **Viewing** — lazily memory-mapped browser, pan/zoom/measure, stacked lanes
- **Pipeline** — filters, resampling, FFT/STFT, applied live to every analysis
- **Alignment** — peak and cross-correlation, including per-tile alignment
- **Analyses** — Welch t-test (TVLA), CPA, SNR, cross-correlation, FFT
- **Export** — TRS, NPY, NPZ, plus a dataset export with labels
- **Chains** — save a whole workflow and replay it with one click

## Quick start

```bash
scripts/install-deps.sh                       # build dependencies for your distro
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/trs-viewer                            # file picker on startup
./build/trs-viewer path/to/file.trs           # open directly
```

`scripts/install-deps.sh --print` shows the package list without installing it.

## Documentation

Full documentation lives in [`docs/`](docs/index.md), and is published as a
site built with [MkDocs](https://www.mkdocs.org/) + Material:

| | |
|---|---|
| [Installation](docs/install.md) | Per-distribution packages, WSL, macOS, troubleshooting |
| [Usage](docs/usage.md) | Opening files, navigation, interaction modes, exporting |
| [Features](docs/index.md#what-it-does) | One page per analysis |
| [File formats](docs/formats.md) | TRS, NPY and NPZ layouts trs-viewer reads and writes |
| [Development](docs/development.md) | Project layout |

To build the docs site locally:

```bash
pip install mkdocs-material
mkdocs serve          # live preview on http://127.0.0.1:8000
mkdocs build          # static site into site/
```
