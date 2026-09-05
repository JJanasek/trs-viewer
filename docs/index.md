# trs-viewer

<p align="center">
  <img src="logo.svg" alt="trs-viewer logo" width="320"/>
</p>

Interactive power-trace viewer and side-channel analysis toolkit for Riscure
`.trs` files and NumPy trace sets.

![Trace browser and pipeline](trace_browser.gif)

## What it does

Open multi-gigabyte trace sets without loading them into RAM, build a
processing pipeline on top of them, and run the standard side-channel
analyses over the result — each in its own tab, several files at a time.

| | |
|---|---|
| **[Viewing](features/viewing.md)** | Lazily memory-mapped trace browser, pan/zoom/measure, stacked lanes |
| **[Pipeline](features/pipeline.md)** | Filters, resampling, FFT/STFT and more, applied live to every analysis |
| **[Alignment](features/alignment.md)** | Peak and cross-correlation alignment, including per-tile alignment |
| **[Welch t-test](features/ttest.md)** | TVLA leakage assessment with thresholds and export |
| **[CPA](features/cpa.md)** | Correlation power analysis with a Python leakage model |
| **[SNR](features/snr.md)** · **[Static SNR](features/static-snr.md)** | Per-sample signal-to-noise |
| **[Cross-correlation](features/xcorr.md)** · **[Heatmap](features/heatmap.md)** | Trace similarity matrices |
| **[FFT](features/fft.md)** | Averaged spectra over a selected region |
| **[Export](features/export.md)** · **[Dataset export](features/export-dataset.md)** | TRS, NPY and NPZ out |
| **[Chain editor](features/chain.md)** | Save a whole workflow and replay it with one click |

## Getting started

```bash
scripts/install-deps.sh                       # installs build dependencies
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/trs-viewer path/to/file.trs
```

Full details, per-distribution package lists and troubleshooting are in
**[Installation](install.md)**; the day-to-day controls are in
**[Usage](usage.md)**.
