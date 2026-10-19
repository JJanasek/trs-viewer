# Welch T-Test

**Analyze → Welch t-test (TVLA)…**

Computes the per-sample Welch t-statistic between two trace groups. No special header fields are required — group assignment is configured in the dialog by choosing which data byte to use as the group label (0 = group 0, non-zero = group 1). If the file's parameter map contains a `ttest` entry, its `offset` field is used automatically and the byte selector is hidden.

## Order, jitter window and sanity check

The **Test** group in the dialog:

| Setting | Effect |
|---|---|
| **Order: 1st** | Compares the groups' means at each sample (classic TVLA). |
| **Order: 2nd** | Compares their **variances**, for masked implementations whose first-order leakage is hidden. |
| **Jitter window** | 2nd order: adds up the centred squares over this many samples, so leftover misalignment smaller than the window still counts. Use about 2–4× the remaining jitter. |
| **Show the 1st-order curve alongside** | Drawn faded behind a 2nd-order result. For a masked design, expect a flat 1st order and a peaked 2nd order. |
| **Random-split sanity check** | Uses only group-0 traces, split into two random halves. Nothing should leak; if it does, the alignment or preprocessing created it. |

### Second order

The second-order test is the univariate test of Schneider & Moradi
(*Leakage Assessment Methodology*, CHES 2015). It runs in two passes:

1. A first-order pass gives each group's per-sample mean (and the 1st-order
   curve shown alongside).
2. Each trace is transformed into `y = (x − mean of its group)²`, optionally
   summed over the jitter window, and the Welch t-test runs on `y`.

Second order is much more sensitive to misalignment than first order. Align
first. When different parts of the trace need different shifts, use
[pattern alignment](patterns.md). Live progress shows the second pass.

The test is *univariate*: it finds leakage where the shares are processed at
the same moment (typical of hardware masking). Software masking that handles
the shares at different times needs a bivariate test, which is not available
yet.

## Using patterns

When the file has aligned [patterns](patterns.md), the dialog shows **Use
patterns** with a checklist. Each selected pattern is tested on the raw traces
with its own shifts, and the results are joined side by side in one tab with
labelled bands. Calc TH uses all samples of all patterns.

## Live progress

With **Show intermediate results while it runs** ticked (the default), a
*Live t-test* tab opens as soon as you click **Run t-test**. While the job runs
it shows:

- **The t-curve so far,** recomputed every *N* traces (**Update every**, and at
  most a few times per second). The previous three snapshots are drawn faded
  behind the current one, so you can watch real leakage peaks grow while noise
  stays put. Overlays switch off automatically for traces longer than 500,000
  samples, to save memory.
- **max |t| vs traces processed,** a small plot under the controls. It is the
  classic "traces to disclosure" view, and the status line notes when |t| first
  went over the threshold.
- **Progress:** traces done, traces per second, time remaining, the number of
  traces per group, and how many samples are currently over the threshold.

The controls:

| Button | Effect |
|---|---|
| **Pause / Resume** | Hold the run between batches. |
| **Stop & Keep Result** | End now and open the full result tab for the traces processed so far. |
| **Cancel** | Abort and discard. |

When the run ends, the normal result tab described below opens next to the
live tab, which keeps the history. The live tab can be closed at any time; the
job carries on.

!!! note "Random trace order"
    **Random trace order** (on by default) processes the traces in a fixed
    pseudo-random order. If a file stores all of one group first and then the
    other, file order would make early snapshots lopsided, or impossible until
    the second group starts. The final result is the same in either order,
    apart from floating-point rounding in the last digits.

!!! warning "The first crossing is only indicative"
    Looking at the curve after every update and stopping when it crosses 4.5
    is repeated testing, which raises the false-positive rate. Treat the first
    crossing as a hint, and judge the final result with **Calc TH…**.

## Result tab

The result tab has a single row of controls; pan, zoom, measure and amplitude
come from the main toolbar, as for any tab.

- **Threshold ±**: adjustable threshold line (orange dashed)
- **One-sided**: show only the positive threshold (for abs-preprocessed signals)
- **Calc TH…** — significance-threshold calculator. The per-sample level for an overall type-I error rate α over `n_L` samples uses the **Šidák** correction `α_TH = 1 − (1 − α)^(1/n_L)`, and the threshold is `TH = CDF_t⁻¹(1 − α_TH/2, ν̂)` with Welch–Satterthwaite degrees of freedom computed from the trace data. The dialog reports both the median and approximate ν̂. Follows Zhang, Ding, Durvaux, Standaert & Fei, *Towards Sound and Optimal Leakage Detection Procedure* (IACR ePrint 2017/287, EuroS&P 2018)
- **Cut**: drag across the plot to cut a region out of the exported result
- **Trim ▾**: exclude a number of samples at the start or end; the exported length is shown next to it while trimmed
- **Style…**: plot title, line width, trace colour, dark/light theme
- **Export ▾**: figure (PDF / SVG / PNG — see [Figures](figures.md)), copy figure, curve as `.npz`, or curve as TRS

**File → Load Result → t-test Curve (NPY)…** — load a pre-computed 1-D `float32` t-statistic vector.

![T-test result and PDF export](../ttest_export.gif)
