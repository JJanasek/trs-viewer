# Development

```
trs-viewer/
├── CMakeLists.txt
├── scripts/
│   └── install-deps.sh         # Installs build deps for the detected distro
├── inc/
│   ├── mainwindow.h
│   ├── trs_file.h              # TRS + in-memory trace source
│   ├── plot_widget.h           # Interactive trace plot
│   ├── heatmap_widget.h        # Interactive correlation heatmap
│   ├── processing.h            # ITransform pipeline
│   ├── align.h                 # Trace alignment (xcorr + peak methods)
│   ├── chain.h                 # Saveable/replayable operation chains
│   ├── job_manager.h           # Background analysis jobs (worker pool)
│   ├── flow_layout.h           # Wrapping toolbar layout
│   ├── ttest.h                 # Welch t-test accumulator
│   ├── snr.h                   # Online SNR accumulator
│   ├── xcorr.h                 # Cross-correlation methods
│   ├── cpa.h                   # Correlation Power Analysis
│   ├── leakage_model.h         # Python leakage model wrapper
│   └── leakage_model_dialog.h  # CPA model editor dialog
└── src/
    ├── main.cpp
    ├── mainwindow.cpp          # Main window, dialogs, NPY/NPZ I/O
    ├── trs_file.cpp            # TRS reader + openFromArray()
    ├── plot_widget.cpp         # Rendering, interaction, PDF export
    ├── heatmap_widget.cpp
    ├── processing.cpp
    ├── align.cpp
    ├── chain.cpp               # Chain JSON persistence
    ├── job_manager.cpp
    ├── flow_layout.cpp
    ├── ttest.cpp
    ├── snr.cpp
    ├── xcorr.cpp               # Baseline · Dual · MP-Cleaned · Two-Window
    ├── cpa.cpp                 # Online-accumulator CPA engine
    ├── leakage_model.cpp       # Python C API wrapper
    ├── leakage_model_dialog.cpp
    └── cli/                    # Shared NPY I/O for the trs-cli front-end
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The binary is at `build/trs-viewer`. See [Installation](install.md) for the
dependencies and per-distribution package names.

## Running the tests

The test suite covers the non-GUI cores (trace I/O, processing, alignment,
t-test, SNR, CPA, cross-correlation), the heatmap widget, and the whole app
through the GUI soak test. It is off by default because it fetches
GoogleTest at configure time:

```bash
cmake -B build-tests -DBUILD_TESTS=ON
cmake --build build-tests -j$(nproc)
cd build-tests && ctest --output-on-failure
```

### GUI soak test

`tests/gui_soak.cpp` runs the real main window offscreen and drives it the
way a user would: dialogs filled in and buttons clicked, on trace files it
generates itself. Each scenario loops:

| Scenario | What it does |
|---|---|
| `ttest` | 1st order, 2nd order and dcor t-tests, then closes the results |
| `average` | Average Trace |
| `fft` | FFT of the average trace and of each trace, then Remove frequencies and the inverse FFT |
| `cpa` | CPA with a Python leakage model |
| `snapshots` | 70 view changes past the undo limit, then undo |
| `combined` | opens and closes a `.trslist` |
| `quit_dialogs` | quits with non-modal dialogs still open |

```bash
QT_QPA_PLATFORM=offscreen build-tests/tests/gui_soak --scenario fft --loops 10
```

A scenario fails on a crash, an unexpected message box, a timeout, or memory
that keeps growing after the first loop.

### Memory checks

Three scripts in `tools/` run the tests under the memory tools:

| Script | Tool | Finds |
|---|---|---|
| `tools/run_asan.sh` | AddressSanitizer, LeakSanitizer, UBSan (`-DTRS_SANITIZE=address,undefined`, in `build-asan/`) | use after free, out-of-bounds access, leaks, undefined behaviour; every test and soak scenario |
| `tools/run_valgrind.sh [--soak <scenario>]` | valgrind memcheck | the same plus reads of uninitialised memory; no special build, but slow |
| `tools/run_heaptrack.sh <scenario> <loops>` | heaptrack | what is still allocated at exit, and memory that grows with the number of loops |

Suppressions go in `tools/lsan.supp`, `tools/valgrind.supp` and
`tools/heaptrack.supp`. Each entry says why it is there. Today the only one is
the Python interpreter's own memory, which it keeps after it is shut down.

## Building the documentation

The site you are reading is generated from the Markdown in `docs/` with
[MkDocs](https://www.mkdocs.org/) and the Material theme — there is no
hand-written HTML to keep in sync:

```bash
pip install mkdocs-material
mkdocs serve     # live preview on http://127.0.0.1:8000
mkdocs build     # static site into site/
```
