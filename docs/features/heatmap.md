# Heatmap Viewer

Interactive false-colour heatmap for correlation matrices:

- Pan (drag) · zoom (scroll wheel)
- Adjustable colour range
- **Colour schemes**: RdBu · Grayscale · Hot · Viridis · Plasma · Lukasz
- **Gaussian blur** for pattern smoothing
- **Abs value** mode and **binary threshold** on `|v|`
- Export as PNG or `.npy`

The window keeps the heatmap as large as possible: a slim bar on top holds the
value readout, **Reset View**, **Export ▾** and the **Display** toggle. All the
other settings sit in a panel on the right, in sections: **Colour** (scheme,
range, auto-clip, abs value, blur, power, binary threshold), **Tone** (contrast,
brightness, region boost), **Navigate** (zoomed-out mode, hotspots, go to a
cell) and **Annotate**. **Display** hides the panel, and the app remembers your
choice.

## Contrast and brightness

The two are coupled, and that is the point. Together they map a normalised
value `t` to

```
t' = (t - 0.5) · contrast + 0.5 + brightness
```

and `t'` is clamped to `[0,1]`. So the pair selects a **window of the data
range of width `1/contrast`, centred at `0.5 − brightness/contrast`** —
everything outside it saturates flat.

That gives the useful recipe for picking out the strongest cells: **raise
contrast first, then push brightness negative.**

| Contrast | Brightness | Visible slice of the range |
|---|---|---|
| 1 | 0 | all of it |
| 1 | −1 | nothing — already fully saturated |
| 10 | 0 | middle 10 % (0.45–0.55) |
| 10 | −1 | 0.55–0.65 |
| 10 | **−4.5** | **top 10 % (0.90–1.00)** |

Brightness therefore runs to **±10**, not ±1: at contrast 1 anything past ±1 is
already off the scale, but at contrast 10 the top decile needs −4.5. Over-driving
is harmless — past roughly `±(0.5 × contrast + 0.5)` the whole map simply goes
flat.

## PNG export

**Export ▾ → PNG image…** writes the *current view* at up to 2400 px on the longer side —
the zoom, colour range, contrast, brightness, colour scheme, region boost and
any annotation boxes, exactly as shown.

Two details make it match the screen rather than merely resemble it:

- The **downsample level follows the on-screen plot size**, not the export
  bitmap. Otherwise a larger PNG pools the matrix more finely, and a single-cell
  peak that *Peak* downsampling had inflated into a visible block collapses back
  to one pixel — which on a 2400 px image is invisible, and is exactly the mark
  the export was made to show.
- The **aspect ratio follows the plot rectangle**, not the view's data
  proportions. On screen the view is stretched to fill that rectangle, so taking
  the data aspect would export differently-shaped output than you are looking at.

## Annotating peaks

Red boxes for marking the cells a figure is about — a correlation peak, a
region worth a second look, the spot a report refers to.

| Control | What it does |
|---|---|
| **Annotate** (toggle) | Drag on the heatmap to draw a box. It is labelled with the strongest value it covers. |
| **Box top peaks** | Finds the *N* strongest cells — at most one per tile, so they spread out instead of clustering on one blob — and boxes each. On a square matrix the main diagonal is skipped, being trivially 1.0. |
| **± cells** | Half-size of each auto-placed box: a peak is one cell, this is how much context to draw around it. |
| **Box region** | Promotes the current shift-drag selection to a permanent box. |
| **Clear boxes** | Removes them all. |

Right-click a box (in Annotate mode) deletes it. Where boxes overlap, the most
recently added one is removed first — the one you see on top.

Two things worth knowing:

- Boxes are stored in **data coordinates**, so they stay on their cells through
  pan and zoom, and survive every colour-scheme, blur, gamma and contrast
  change. A box zoomed out to sub-pixel size is still drawn at a minimum size,
  so marking a single cell in a 30k×30k matrix is useful rather than invisible.
- Boxes are drawn into the **exported PNG**, at the export's own scale. The
  figure carries its own markings instead of needing them added afterwards in
  an image editor.

This is deliberately separate from the region selection (shift-drag). That
selection is a *tool* — it drives local contrast and zoom, and there is only
ever one. These are a *record*: many at once, and they do not move.

Available on the cross-correlation matrix, the trace-similarity heatmap, and
the [Leakage vs Frequency](interference.md#leakage-vs-frequency) time-frequency
map.
