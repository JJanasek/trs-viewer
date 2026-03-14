# trs-viewer

Interactive power-trace viewer and side-channel analysis toolkit for Riscure `.trs` files and NumPy trace sets.

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Qt6](https://img.shields.io/badge/Qt-6%20%7C%205-green)

---

## Screenshots

![Trace browser and pipeline](docs/trace_browser.gif)

![T-test result and PDF export](docs/ttest_export.gif)

![Cross-correlation heatmap](docs/heatmap.gif)

---

## Features

### Trace Viewing

- Load `.trs` power-trace files (Riscure format) **or** NumPy `.npy` / `.npz` trace matrices
- Multi-trace overlay with 8 distinct colours
- Pan · box-zoom · distance-measurement interaction modes
- **Y-axis zoom** — Ctrl+scroll (or Shift+scroll) to compress/expand amplitude; ↑ Amp / ↓ Amp buttons
- Dark and light themes
- Crop ranges — mark sample regions and export as a new `.trs` file

### Signal Processing Pipeline

Transforms stack in order and are applied live on every render:

| Transform | Description |
|---|---|
| Absolute Value | `\|x\|` point-wise |
| Negate | `-x` point-wise |
| Moving Average | Causal sliding window |
| Window Resample | Block-average decimation |
| Stride Resample | Pick every N-th sample |
| Offset | Add constant |
| Scale | Multiply by constant |

### Welch T-Test

Computes the per-sample Welch t-statistic between two trace groups (labelled 0 / 1 via the first auxiliary data byte, or the `ttest` parameter map entry).

**Result dialog features:**
- Adjustable ±threshold line (orange dashed)
- **One-sided mode** — show only the positive threshold (for abs-preprocessed signals)
- **Calc TH…** — Bonferroni-corrected threshold calculator using the Welch-Satterthwaite degrees of freedom computed directly from the accumulated trace data
- **Style…** — set plot title, line width, trace colour, dark/light theme
- **Export PDF…** — A4 landscape vector PDF (ideal after switching to light theme)
- Export as `.npy` or `.trs`

### Cross-Correlation Matrix

Computes the M×M normalised Pearson correlation matrix `C[i,j] = Corr(sᵢ, sⱼ)` across all traces, or a rectangular search×ref matrix for template matching.

| Method | Description | Best for |
|---|---|---|
| **Baseline** | Streaming rank-1 outer-product updates (Welford + double DGEMM) | Any n, moderate M |
| **Dual Matrix** | Gram `G = Aᵀ A / M` → eigen-reconstruction | Large M, moderate n |
| **MP-Cleaned** | Same as Dual but zeroes eigenvalues ≤ λ₊ (Marchenko-Pastur edge) | Noise suppression |
| **Two-Window** | Rectangular search×ref cross-correlation | Template matching / alignment |

All methods support a **stride** to subsample before computing (`M = ⌈samples / stride⌉`).

Numerical accuracy: Welford online variance (no catastrophic cancellation for large DC offsets), double-precision DGEMM throughout, correct Eigen symmetric mirroring.

### Heatmap Viewer

The correlation matrix opens as an interactive false-colour heatmap:

- Pan (drag) · zoom (scroll wheel)
- Adjustable colour range with per-channel spin boxes
- **Colour schemes**: RdBu · Grayscale · Hot · Viridis · Plasma · Lukasz
- **Gaussian blur** (σ-controlled) for pattern smoothing
- **Abs value** mode and **binary threshold** on `|v|`
- Export as PNG (≤ 2400 px) or `.npy`

### NPY / NPZ Support

| Action | Menu / Button |
|---|---|
| Open `.npy` or `.npz` as traces | **File → Open NPY/NPZ as traces…** |
| Export traces (pipeline applied) to `.npy` | **Export → Export traces as NPY…** |
| Export traces + data bytes to `.npz` | **Export → Export traces as NPZ…** |
| Load pre-computed 1-D t-test vector | **SCA → Load T-Test NPY…** |
| Load pre-computed 2-D heatmap matrix | **SCA → Load Heatmap NPY…** |

**NPZ layout** (NumPy convention):
```python
import numpy as np
d = np.load("traces.npz")
traces = d["traces"]   # float32, shape (n_traces, n_samples)
data   = d["data"]     # uint8,   shape (n_traces, data_length)  — if TRS has aux bytes
```

Once loaded from NPY/NPZ, the in-memory trace set is fully compatible with the t-test, xcorr, and all pipeline transforms — no conversion step needed.

---

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.16 | Build system |
| C++ compiler | C++17 | GCC 9+, Clang 10+, MSVC 2019+ |
| Qt | 6.x (5.x fallback) | Core · Gui · Widgets · PrintSupport |
| OpenMP | any | Parallelises correlation computation |
| Eigen3 | ≥ 3.3 | Linear algebra; **auto-downloaded** if not found |

---

## Building

### 1 · Install dependencies

**Arch Linux**
```bash
sudo pacman -S base-devel cmake qt6-base eigen
```

**Ubuntu / Debian (22.04+)**
```bash
sudo apt install build-essential cmake qt6-base-dev libeigen3-dev
```

**Fedora**
```bash
sudo dnf install gcc-c++ cmake qt6-qtbase-devel eigen3-devel
```

**macOS (Homebrew)**
```bash
brew install cmake qt@6 eigen
export CMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
```

> Eigen3 is optional — CMake downloads version 3.4.0 automatically if not found.

### 2 · Clone and build

```bash
git clone <repo-url>
cd trs-viewer

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binary is at `build/trs-viewer`.

### 3 · Run

```bash
./build/trs-viewer                    # file picker on startup
./build/trs-viewer path/to/file.trs   # open directly
./build/trs-viewer traces.npz         # open NPZ directly
```

---

## Usage

### Opening a file

| Method | How |
|---|---|
| TRS file | **File → Open TRS file…** (Ctrl+O) or command-line argument |
| NPY (2-D trace matrix) | **File → Open NPY/NPZ as traces…** |
| NPZ archive | same — looks for `traces.npy` + optional `data.npy` inside |

The side panel shows file metadata: trace count, samples/trace, sample type, data bytes/trace.

### Navigating traces

1. Set **First trace** and **Count** in the side panel, click **Load**.
2. Scroll to zoom, drag to pan, **R** / **Reset** for full view.
3. **Ctrl+scroll** or **Shift+scroll** — Y-axis amplitude zoom.

### Interaction modes

| Button | Key | Action |
|---|---|---|
| Pan | — | Drag to pan, scroll to zoom |
| Measure | P | Click two points — reads sample index, value, and delta |
| Box Zoom | Z | Drag a rectangle to zoom into it |
| Crop Select | — | Drag to add a sample range to the crop list |

### Processing pipeline

1. Pick a transform from the drop-down.
2. Click **+** — a parameter dialog appears for configurable transforms.
3. Reorder with **↑ / ↓**, remove with **−**.
4. Pipeline is applied live on every render and on SCA computations.

### T-Test

**SCA → Run Welch T-Test…**

Each trace's first auxiliary data byte is used as the group label (0 or 1). If the file has a `ttest` parameter map entry that byte offset is used automatically.

- Adjust ±threshold with the spin box or use **Calc TH…** for a statistically grounded Bonferroni-corrected value.
- Tick **One-sided (+)** if the signal was preprocessed with `|·|` (abs value) — hides the negative threshold line.
- **Style…** → set a title, pick trace colour, choose dark/light theme → **Export PDF…** for publication-ready output.

**SCA → Load T-Test NPY…** — load a pre-computed 1-D `float32` t-statistic vector.

### Cross-Correlation

**SCA → Cross-Correlation…**

1. Set trace range, sample range, and **stride** (increase to subsample large traces).
2. Choose a method:
   - **Baseline** — safe default for any dataset size
   - **Dual Matrix / MP-Cleaned** — better for large M (many samples), moderate n
   - **Two-Window** — opens a second window range picker; outputs a rectangular search×ref matrix
3. Computation runs in the background with a progress bar; cancel at any time.
4. The heatmap opens when done.

**Memory guidance** (Dual / MP methods):

| n traces | stride → M | A matrix | G matrix |
|---|---|---|---|
| 500 | 1 → 1 000 | 4 MB | 2 MB |
| 2 000 | 4 → 2 500 | 40 MB | 32 MB |
| 5 000 | 10 → 3 000 | 120 MB | 200 MB |

### Exporting

| What | Where |
|---|---|
| Transformed traces → TRS | **Export → Export TRS…** |
| Trace matrix → NPY | **Export → Export traces as NPY…** |
| Trace matrix + data → NPZ | **Export → Export traces as NPZ…** |
| Plot → PNG | **Export → Export plot as PNG…** (Ctrl+Shift+S) |
| Plot → PDF | **Export → Export plot as PDF…** or **Export PDF…** in result dialogs |
| Heatmap → PNG | "Export PNG…" in heatmap dialog |
| Heatmap → NPY | "Export .npy…" in heatmap dialog |
| T-test vector → NPY | "Export .npy…" in t-test dialog |

---

## Performance Notes

The cross-correlation engine uses Welford's online algorithm for numerically stable variance, double-precision DGEMM (Eigen `MatrixXd`), and OpenMP parallelism:

| Operation | Implementation |
|---|---|
| Per-sample mean + variance | Welford online (no catastrophic cancellation) |
| Normalised trace matrix A | Eigen `MatrixXd`, double precision |
| Gram matrix G = AᵀA / M | Eigen DGEMM |
| Eigendecomposition | `SelfAdjointEigenSolver` (O(n²√n)) |

Approximate wall time on a modern 8-core machine:

| n | M | Method | Time |
|---|---|---|---|
| 500 | 2 000 | Baseline | ~2 s |
| 2 000 | 1 000 | MP-Cleaned | ~5 s |
| 5 000 | 500 | MP-Cleaned | ~30 s |

---

## File Format

The reader supports standard Riscure TRS v1 files:

| Type tag | C type |
|---|---|
| `0x01` | `int8_t` |
| `0x02` | `int16_t` |
| `0x04` | `int32_t` |
| `0x14` | `float32` |

Traces are memory-mapped — the full file is never loaded into RAM.

For NPY/NPZ: only little-endian `float32` (`<f4`) and `uint8` (`\|u1`) dtypes are supported. NPZ must use STORE compression (no deflate).

---

## Project Layout

```
trs-viewer/
├── CMakeLists.txt
├── inc/
│   ├── mainwindow.h
│   ├── trs_file.h          # TRS + in-memory trace source
│   ├── plot_widget.h       # Interactive trace plot (title, trace width, Y-zoom, PDF export)
│   ├── heatmap_widget.h    # Interactive correlation heatmap
│   ├── processing.h        # ITransform pipeline
│   ├── ttest.h             # Welch t-test accumulator + Welch-Satterthwaite df
│   └── xcorr.h             # Cross-correlation methods
└── src/
    ├── main.cpp
    ├── mainwindow.cpp       # Main window, all dialogs, NPY/NPZ I/O, ZIP writer
    ├── trs_file.cpp         # Memory-mapped TRS reader + openFromArray()
    ├── plot_widget.cpp      # Rendering, interaction, PDF render
    ├── heatmap_widget.cpp
    ├── processing.cpp
    ├── ttest.cpp
    └── xcorr.cpp            # Baseline · Dual · MP-Cleaned · Two-Window
```
