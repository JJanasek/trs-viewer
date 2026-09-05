# Trace Alignment

**SCA → Align Traces…**

Align a set of traces to a reference using one of two methods:

| Method | Description |
|---|---|
| **Cross-correlation** | Finds the shift that maximises the normalised cross-correlation with the reference trace |
| **Peak alignment** | Aligns the highest-amplitude peak of each trace to the reference peak |

Configure:

| Setting | Meaning |
|---|---|
| **First trace / Count** | The set of traces to align |
| **Reference trace** | Absolute trace index used as the alignment template |
| **Reference region first / length** | The window *within* the reference that is matched against, in pipeline-processed samples |
| **Search half-window ±** | How far each trace may be shifted while searching |
| **Peak mode** | For peak alignment: absolute max `\|v\|` or signed max |
| **Discard below correlation** | Optionally drop traces whose best correlation falls under a threshold |
| **Output mode** | Full trace padded with the trace average · full trace padded with zeros · crop to the common range |

Results are shown in a table and previewed in a separate plot. Click **Apply to Main View** to bake the aligned traces into the main display, or **Un-apply Shifts** to return to the live file-backed view — the computed shifts are kept for later reuse either way.

Alternatively, use the **↔ Align** drag mode in the main toolbar — drag any trace left or right to manually shift it one sample at a time.

Once alignment is applied (either method), the stored shifts are available to all SCA tools. Every SCA dialog shows an **"Apply last alignment shifts"** checkbox (pre-ticked when shifts are available) that locks the trace/sample range to the aligned values and applies per-trace shifts during analysis. Alignment state is automatically cleared when a new file is loaded or the trace set is changed.
