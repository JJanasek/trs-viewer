# Chain Editor

**Chain → Chain Editor…**

A **chain** is a saved, ordered sequence of operations — build it once, then run it with one click instead of redoing the same dialogs by hand every time. Chains are stored as JSON and can be loaded back later, so a whole workflow (add helper transforms → align → strip the helper transforms → reload the raw traces with the shift still applied → export) becomes one button.

| Step kind | Effect |
|---|---|
| **Add Transform** | Append a configured transform to the pipeline. Opens the same per-type parameter dialog as adding one normally |
| **Clear Pipeline** | Empty the transform pipeline |
| **Align** | Run alignment and store the result. Opens the real, interactive Align Traces dialog (drag-on-plot region, **Run**, results table) — its **Apply to Main View + Add to Chain** button bakes the alignment as usual *and* captures the exact parameters that just worked into the step |
| **Reload** | Re-read traces live from the file, over the stored alignment's own trace range if one exists — this is what makes "align on a processed view, then export the raw traces with the same shift" possible |
| **Export** | Write the current traces (TRS / NPY / NPZ), with the same range and **Apply last alignment shifts** option as the standalone export dialogs. Give it a fixed output path or leave it blank to be prompted each run |

The Align step is deliberately *not* a blind parameter form — remembering a good reference region or correlation threshold without seeing it applied is hard, so building the step means actually running alignment against the current view first, then capturing what worked.

Steps run against whichever dataset is active when you click **Run** — a chain has no "open file" step of its own (yet).
