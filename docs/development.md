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
t-test, SNR, CPA, cross-correlation). It is off by default because it fetches
GoogleTest at configure time:

```bash
cmake -B build-tests -DBUILD_TESTS=ON
cmake --build build-tests -j$(nproc)
cd build-tests && ctest --output-on-failure
```

Add `-DENABLE_SANITIZERS=ON` to build them with AddressSanitizer and
LeakSanitizer.

## Building the documentation

The site you are reading is generated from the Markdown in `docs/` with
[MkDocs](https://www.mkdocs.org/) and the Material theme — there is no
hand-written HTML to keep in sync:

```bash
pip install mkdocs-material
mkdocs serve     # live preview on http://127.0.0.1:8000
mkdocs build     # static site into site/
```
