# Usage

## Opening a file


| Method | How |
|---|---|
| TRS file | **File → Open TRS file…** (Ctrl+O) or command-line argument |
| NPY (2-D trace matrix) | **File → Open NPY/NPZ as traces…** |
| NPZ archive | same — looks for `traces` array + optional `data` array |


## Navigating traces


1. Set **First trace** and **Count** in the side panel, click **Load**.
2. Scroll to zoom, drag to pan, **R** / **Reset** for full view.
3. **Ctrl+scroll** or **Shift+scroll** — Y-axis amplitude zoom.


## Interaction modes


| Button | Key | Action |
|---|---|---|
| Pan | — | Drag to pan, scroll to zoom |
| Measure | P | Click two points — reads sample index, value, and delta |
| Box Zoom | Z | Drag a rectangle to zoom into it |
| Align | — | Drag a trace left/right to manually shift it |
| Crop Select | — | Drag to add a sample range to the crop list (enable from the Range Editor) |
| Stack | — | Toggle — draw each trace in its own lane rather than overlaid. Combines with any mode above |

Result windows (t-test, SNR, FFT, loaded NPY) carry their own smaller toolbar: **Pan · Measure · ⬚ Zoom · ✂ Cut · Reset**.


## Processing pipeline


1. Pick a transform from the drop-down and click **+**.
2. Reorder with **↑ / ↓**, remove with **−**.
3. Applied live on every render and on all SCA computations.


## Exporting


| What | Where |
|---|---|
| Transformed traces → TRS | **Export → Export TRS…** |
| Trace matrix → NPY | **Export → Export traces as NPY…** |
| Trace matrix + data → NPZ | **Export → Export traces as NPZ…** |
| Trace matrix + named labels → NPZ | **Export → Export Dataset (NPZ)…** |
| Plot → PNG | **Export → Export plot as PNG…** (Ctrl+Shift+S) |
| Plot → PDF | **Export → Export plot as PDF…** or **Export PDF…** in result dialogs |
| Heatmap → PNG | "Export PNG…" in heatmap dialog |
| Heatmap → NPY | "Export .npy…" in heatmap dialog |
| T-test vector → NPY | "Export .npy…" in t-test dialog |

---
