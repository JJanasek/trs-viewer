# Average Trace

**Analyze → Average Trace…**

The per-sample **mean** of a range of traces, read exactly as the t-test
reads them: with the last alignment's shifts (if ticked), through the
pipeline, in parallel, as a background job. The result opens as a tab in the
main window, so every plot tool, **Tile Tabs** (e.g. next to a t-test or an
FFT) and **Export Figure** work on it.

| Setting | |
|---|---|
| **First trace / Count** | the traces to average |
| **First sample / Sample count** | the sample range (0 = all) |
| **Apply last alignment shifts** | average the aligned traces, so a jittered feature stays sharp instead of smearing out (one tile's shifts, when the alignment is tiled) |
| **Average** | *All traces together*, or *Each t-test group separately* (group from the data byte, as in the t-test; the file's `ttest` field picks it automatically) |
| **Show ± 1 standard deviation** | faint curves at mean ± σ (sample standard deviation) |
| **Also the difference of the group means** | per group: a second tab with G1 − G0 |

Per group, the tab shows *Group 0 mean* and *Group 1 mean*. The difference
of the means shows where the groups' average power differs; the t-test is the
same difference divided by its noise, so the two side by side (Tile Tabs) show
whether a t-test peak is a large difference or a quiet sample.

Files without per-trace data bytes can only be averaged as one set.
