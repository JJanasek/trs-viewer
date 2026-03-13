# trs-viewer

Interactive power-trace viewer and side-channel analysis toolkit for Riscure `.trs` files.

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![Qt6](https://img.shields.io/badge/Qt-6%20%7C%205-green)

---

## Features

### Trace Viewing
- Load and browse `.trs` power-trace files (Riscure format)
- Multi-trace overlay with 8 distinct colours
- Pan, box-zoom, and distance-measurement interaction modes
- Dark and light themes
- Crop ranges — mark sample regions and export them as a new `.trs` file

### Signal Processing Pipeline
Transforms are stacked in order and applied to every rendered trace:

| Transform | Description |
|---|---|
| Absolute Value | `|x|` point-wise |
| Negate | `-x` point-wise |
| Moving Average | Causal sliding window (smoothing) |
| Window Resample | Block-average decimation — `floor(N/W)` output samples |
| Stride Resample | Pick every Nth sample — `ceil(N/stride)` output samples (identical to XCorr stride) |
| Offset | Add constant |
| Scale | Multiply by constant |

### Side-Channel Analysis (SCA)

#### Welch T-Test
Computes the per-sample Welch t-statistic between two trace groups (labelled 0 / 1).
Result is displayed as a 1-D plot with a configurable significance threshold.

#### Cross-Correlation Matrix
Computes the M×M normalised correlation matrix `C[i,j] = Corr(s_i, s_j)` across all traces.

Three computation methods:

| Method | How | Best for |
|---|---|---|
| **Baseline** | Streaming outer-product `Xnorm^T Xnorm / n` | Any n, moderate M |
| **Dual Matrix** | Gram matrix `G = A^T A / M` → eigen-reconstruction | Large M, moderate n |
| **MP-Cleaned** | Same as Dual but zeroes eigenvalues ≤ λ+ (Marchenko-Pastur edge) | Noise suppression |

All methods support a **stride** parameter to subsample the trace before computing (`M = ceil(samples / stride)`).

#### Heatmap Viewer
The correlation matrix is shown as an interactive false-colour heatmap:
- Pan (drag) and zoom (scroll wheel)
- Adjustable colour range
- **Colour schemes**: RdBu · Grayscale · Hot · Viridis · Plasma
- **Gaussian blur** (σ-controlled) for pattern smoothing
- **Binary threshold** on `|v|` to binarise leakage regions
- Export as scaled PNG (≤ 2400 px) or `.npy`

#### NPY Import
Pre-computed t-test vectors and heatmap matrices (saved as NumPy `.npy`, dtype `float32`) can be loaded directly for visualisation without re-running the analysis.

### Export
- **TRS** — export cropped / transformed traces as a new `.trs` file
- **PNG** — plot or heatmap (capped at ~300 DPI / 2400 px)
- **PDF** — vector plot export

---

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.16 | Build system |
| C++ compiler | C++17 | GCC 9+, Clang 10+, MSVC 2019+ |
| Qt | 6.x (5.x fallback) | Core · Gui · Widgets |
| OpenMP | any | Parallelises correlation computation |
| Eigen3 | ≥ 3.3 | Linear algebra for eigendecomposition; **auto-downloaded** if not found |

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

> **Note:** Eigen3 is optional — if not found CMake will automatically download version 3.4.0 during the configure step (requires internet access).

### 2 · Clone and build

```bash
git clone <repo-url>
cd trs-viewer

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The binary is at `build/trs-viewer`.

### 3 · Run

```bash
./build/trs-viewer                   # open file picker on startup
./build/trs-viewer path/to/file.trs  # open file directly
```

---

## Usage

### Opening a file
Use **File → Open** (Ctrl+O) or pass the path as a command-line argument.
The side panel shows file metadata (trace count, samples per trace, sample type).

### Navigating traces
- Set **First trace** and **Count** in the side panel, then click **Load**.
- Scroll the plot to zoom, drag to pan.
- Press **R** or click **Reset** to return to full view.

### Interaction modes (toolbar)
| Button | Key | Action |
|---|---|---|
| Pan | — | Drag to pan, scroll to zoom |
| Measure | P | Click two points to read distance and value difference |
| Box Zoom | Z | Drag a rectangle to zoom into it |

### Processing pipeline
1. Select a transform from the drop-down.
2. Click **+** — a parameter dialog appears for configurable transforms.
3. Reorder with **↑ / ↓**, remove with **−**.
4. Pipeline is applied live on every render.

### T-Test
**SCA → Run T-Test**
Traces must be loaded first. Each trace's first byte of auxiliary data is used as the group label (0 or 1). The result plot shows the t-statistic with a ±4.5 significance threshold.

### Cross-Correlation
**SCA → Run Cross-Correlation**

1. Set the trace range, sample range, and **stride** (increase to subsample large traces).
2. Choose a method (Baseline · Dual Matrix · MP-Cleaned).
3. The matrix is computed in the background with a progress bar; cancel at any time.
4. The heatmap opens when computation finishes.

**Memory guidance for Dual / MP methods:**

| n (traces) | stride → M | A matrix | G matrix |
|---|---|---|---|
| 500 | 1 → 1 000 | 4 MB | 2 MB |
| 2 000 | 4 → 2 500 | 40 MB | 32 MB |
| 5 000 | 10 → 3 000 | 120 MB | 200 MB |

### Loading pre-computed results
- **SCA → Load T-Test NPY…** — 1-D `float32` array
- **SCA → Load Heatmap NPY…** — 2-D square `float32` array

Both formats must be standard NumPy `.npy` v1 or v2, dtype `<f4` (little-endian float32).

### Exporting
| Action | Menu |
|---|---|
| Save transformed traces | Export → Export Cropped Traces to TRS |
| Save plot image | Export → Export Plot to PNG / PDF |
| Save heatmap image | "Export PNG…" button in heatmap dialog |
| Save matrix | "Export .npy…" button in heatmap dialog |

---

## Performance Notes

The cross-correlation engine is parallelised with OpenMP and uses Eigen's `SelfAdjointEigenSolver` (divide-and-conquer, SIMD-optimised) in place of a custom Jacobi solver:

| Operation | Implementation |
|---|---|
| Rank-1 outer-product updates | `#pragma omp parallel for` over rows |
| Gram matrix G = A^T A / M | `#pragma omp parallel for` with dynamic scheduling |
| Eigendecomposition of G (n×n) | Eigen SelfAdjointEigenSolver (O(n²√n)) |
| A·vₖ matrix-vector products | `#pragma omp parallel for` over rows |

Approximate wall time on a modern 8-core machine:

| n | M | Method | Time |
|---|---|---|---|
| 500 | 2 000 | Baseline | ~2 s |
| 2 000 | 1 000 | MP-Cleaned | ~5 s |
| 5 000 | 500 | MP-Cleaned | ~30 s |

---

## File Format

The reader supports standard Riscure TRS v1 files with sample types:

| Type tag | C type | Notes |
|---|---|---|
| `0x01` | `int8_t` | Scaled by `yscale` |
| `0x02` | `int16_t` | Scaled by `yscale` |
| `0x04` | `int32_t` | Scaled by `yscale` |
| `0x14` | `float32` | Used directly |

Traces are memory-mapped and read on demand — the full file is never loaded into RAM.

---

## Project Layout

```
trs-viewer/
├── CMakeLists.txt
└── src/
    ├── main.cpp            # Entry point
    ├── mainwindow.cpp/h    # Main window and all dialogs
    ├── trs_file.cpp/h      # TRS file reader (memory-mapped)
    ├── processing.cpp/h    # ITransform pipeline
    ├── plot_widget.cpp/h   # Interactive trace plot
    ├── heatmap_widget.cpp/h # Interactive correlation heatmap
    ├── ttest.cpp/h         # Welch t-test accumulator
    └── xcorr.cpp/h         # Cross-correlation (OpenMP + Eigen)
```
