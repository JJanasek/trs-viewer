# Usage

## Opening a file


| Method | How |
|---|---|
| TRS file | **File → Open TRS File…** (Ctrl+O) or command-line argument |
| Several TRS files as one set | **File → Open Several Files as One…**; reopen with the saved `.trslist` ([details](features/combined.md)) |
| NPY (2-D trace matrix) | **File → Open NPY/NPZ as Traces…** |
| NPY with labels | same — `labels.npz` next to it is read as the per-trace data ([details](features/npy-npz.md)) |
| NPZ archive | same — looks for a `traces` array; the other per-trace arrays become labels |
| Any of the above | Drag the file onto the window |
| A file you opened before | **File → Open Recent**, or click it on the start page |

With no file open, the window shows a start page with the open buttons,
your recent files and a short outline of the usual workflow.


## Navigating traces


1. Set **First trace** and **Count** in the side panel, click **Reload Traces**.
2. Scroll to zoom, drag to pan, **R** / **Reset** for full view (crop ranges from the Range Editor are kept; clear them there).
3. **Ctrl+scroll** or **Shift+scroll** — Y-axis amplitude zoom.


## Theme

**View › Theme** offers *Follow System* (the default: it switches with your
desktop's light/dark setting), *Dark* and *Light*. The theme covers the whole
window, including the plots, result tabs and heatmaps. Trace colours come from
the colour-vision-safe Okabe–Ito set, adjusted for each theme. In light mode the
plot area stays pure white, so exported PNG/PDF figures sit cleanly on a page.


## Remembered settings

The analysis dialogs remember their *method* settings from the last run: t-test
grouping byte and |t|, SNR class mode and bytes, CPA hypothesis count, FFT
window/output options, Leakage vs Frequency STFT settings, and the
cross-correlation method. Trace and sample ranges and "Apply last alignment
shifts" are never remembered; they always start from the file that is open, so
a range from one file is never silently applied to another.


## Status bar

The status bar always shows the active tab's state: which traces are drawn
out of how many, how many pipeline steps are applied, whether an alignment
is stored, and **Baked** when the view is a frozen "Apply to Main View" copy
(click **Back to Live View** in the side panel to return to the live view). Baked tabs are also
marked `[baked]` in the tab bar, and result tabs (t-test, SNR, …) are drawn
in a different colour.


## Alignment & View panel

Below the pipeline, the side panel shows the alignment stored for the
active file, which Reload, export and the analyses all reuse, and whether
the plot is live or baked.

- **Align…** opens Process › Align Traces.
- **Discard** forgets the stored alignment, so everything goes back to
  unshifted traces. Ctrl+Z undoes it.
- **Back to Live View** appears only while the view is baked. It replaces the
  frozen copy with live, file-backed traces and keeps the alignment.


## Interaction modes


| Button | Key | Action |
|---|---|---|
| Pan | — | Drag to pan, scroll to zoom |
| Measure | P | Click two points — reads sample index, value, and delta |
| Box Zoom | Z | Drag a rectangle to zoom into it |
| Drag-shift | — | Drag a trace left/right to shift it by hand (changes the alignment; Ctrl+Z undoes) |
| Crop Select | — | Drag to add a sample range to the crop list (enable from the Range Editor) |
| Stack | — | Toggle — draw each trace in its own lane rather than overlaid. Combines with any mode above |

Result windows (t-test, SNR, FFT, loaded NPY) carry their own smaller toolbar: **Pan · Measure · ⬚ Zoom · ✂ Cut · Reset**.


## Processing pipeline


1. Pick a transform from the drop-down and click **Add**.
2. Reorder with **↑ / ↓**, remove with **Remove**.
3. Applied live on every render and on all SCA computations.

Some steps need a whole trace in one piece: Spectral Excise, FFT and STFT
Magnitude, zero-phase filters, Periodic Subtract and Complex Demodulate. The
plot applies them only when the view holds at most 4M samples. Zoomed out
further, the traces are drawn without them and a note on the plot names the
steps left out. Analyses always apply the whole pipeline.

## Memory

Before a large analysis or export, trs-viewer compares what it will need
with the memory free at that moment:

- below half of the free memory it just runs;
- up to 90 % it asks first;
- above that it refuses and says what needs less: a narrower sample range,
  fewer traces, or a **Window Resample** step in the pipeline.

Running out of memory would close the whole program, along with the work in
every other tab.


## Exporting


| What | Where |
|---|---|
| Transformed traces → TRS | **File → Export → TRS…** |
| Trace matrix → NPY | **File → Export → Traces as NPY…** |
| Trace matrix + data → NPZ | **File → Export → Traces as NPZ…** |
| Trace matrix + named labels → NPZ | **File → Export → Dataset (NPZ)…** |
| Plot → figure | **File → Export → Figure (PDF, SVG, PNG)…** (Ctrl+Shift+S), or right-click a plot — see [Figures](features/figures.md) |
| Heatmap → PNG | "Export PNG…" in heatmap dialog |
| Heatmap → NPY | "Export .npy…" in heatmap dialog |
| T-test vector → NPY | "Export .npy…" in t-test dialog |

## When it crashes

If trs-viewer crashes, it writes a report to
`~/.local/state/trs-viewer/crash-<pid>.log` (and to the terminal it was
started from) before it ends. The report has:

- the signal, and the process that sent it if it came from outside. For
  example, `kwin_killer_helper` means the window manager closed a window that
  had stopped responding.
- the backtrace;
- the last events: menu actions, files opened and tabs closed, analyses
  started and finished, and Qt warnings.

The system still records the core dump. For every thread's stack:

```bash
coredumpctl list trs-viewer
coredumpctl info <pid>        # stack of every thread
coredumpctl gdb <pid>         # debugger on the core dump
```

Please attach the crash log when reporting a crash. When the process ends,
the system frees all its memory, so a crash can't leave memory allocated.

Ctrl+C in the terminal, or `kill <pid>`, quits like File › Quit: running
analyses are cancelled and the settings saved. A second one ends the program
at once.

---
