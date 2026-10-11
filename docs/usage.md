# Usage

## Opening a file


| Method | How |
|---|---|
| TRS file | **File → Open TRS File…** (Ctrl+O) or command-line argument |
| NPY (2-D trace matrix) | **File → Open NPY/NPZ as Traces…** |
| NPZ archive | same — looks for `traces` array + optional `data` array |


## Navigating traces


1. Set **First trace** and **Count** in the side panel, click **Reload Traces**.
2. Scroll to zoom, drag to pan, **R** / **Reset** for full view.
3. **Ctrl+scroll** or **Shift+scroll** — Y-axis amplitude zoom.


## Status bar

The status bar always shows the active tab's state: which traces are drawn
out of how many, how many pipeline steps are applied, whether an alignment
is stored, and **Baked** when the view is a frozen "Apply to Main View" copy
(click **Reload Traces** to go back to the live view). Baked tabs are also
marked `[baked]` in the tab bar, and result tabs (t-test, SNR, …) are drawn
in a different colour.


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
| Plot → PNG | **File → Export → Plot as PNG…** (Ctrl+Shift+S) |
| Plot → PDF | **File → Export → Plot as PDF…** or **Export PDF…** in result dialogs |
| Heatmap → PNG | "Export PNG…" in heatmap dialog |
| Heatmap → NPY | "Export .npy…" in heatmap dialog |
| T-test vector → NPY | "Export .npy…" in t-test dialog |

---
