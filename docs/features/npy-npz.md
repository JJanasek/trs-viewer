# NPY / NPZ Support

| Action | Menu |
|---|---|
| Open `.npy` or `.npz` as traces | **File → Open NPY/NPZ as Traces…** |
| Export traces (pipeline applied) to `.npy` | **File → Export → Traces as NPY…** |
| Export traces + data bytes to `.npz` | **File → Export → Traces as NPZ…** |
| Export traces + derived labels to `.npz` | **File → Export → Dataset (NPZ)…** |
| Load pre-computed 1-D t-test vector | **File → Load Result → t-test Curve (NPY)…** |
| Load pre-computed 2-D heatmap matrix | **File → Load Result → Heatmap (NPY)…** |

---

## Labels next to a `.npy`

A capture is often a plain `traces.npy` with its labels in an archive beside
it. When you open `<name>.npy`, the first of these that exists is read:

`<name>_labels.npz`, `<name>-labels.npz`, `<name>.labels.npz`, `<name>.npz`, `labels.npz`

Every array whose first dimension is the number of traces becomes a named
label: `lam` of shape (1000, 32) is 32 bytes per trace, `retval` of shape
(1000,) is one number per trace. They appear in the Trace Data panel, and the
status bar and file details say what was found. The samples stay memory-mapped
and only the labels are held in memory, so the matrix can be larger than RAM.

Then every analysis that reads per-trace data works on them:

- **t-test, Average per group, Leakage vs Frequency:** *Group by* picks the
  label, the byte and the bit (see [the t-test](ttest.md)).
- **SNR:** picks the label and byte the classes come from.
- **CPA:** the model gets all labels in `plaintexts`; the editor lists where
  each one is ([details](cpa.md)).
- **Export TRS** keeps the labels and their names.

Left out: scalars and arrays of another length, which are not per-trace
labels. Arrays with a row per trace that can't be used (strings, big-endian,
Fortran order, more than 32767 bytes per trace in total) are named in the note.
If the archive has no usable array at all, a warning says so and the traces
open without labels.

The same rules apply to the other arrays inside an `.npz` opened as traces. A
matrix named `data`, `labels`, `textin` or `plaintext` comes first, with one
byte per element, so its columns stay data bytes 0, 1, 2, ...
