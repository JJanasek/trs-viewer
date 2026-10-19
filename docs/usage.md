# Usage

## Opening a file


| Method | How |
|---|---|
| TRS file | **File → Open TRS File…** (Ctrl+O) or command-line argument |
| NPY (2-D trace matrix) | **File → Open NPY/NPZ as Traces…** |
| NPZ archive | same — looks for `traces` array + optional `data` array |
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

---
