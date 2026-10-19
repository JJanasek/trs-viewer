# Pattern Alignment

**Process → Patterns…**

When the misalignment differs from one part of the trace to another (for
example, random delays between operations), a single shift per trace is not
enough. Pattern alignment lets you mark several regions ("patterns"), align
each one **on its own**, and keep only the shifts. The t-test then reads the
**raw** traces pattern by pattern, each with its own shifts.

## Workflow

1. **Make the structure visible.** Add a *Window Resample* (and perhaps
   *Absolute Value → Moving Average*) to the pipeline so the operations stand
   out. These stages only help you see and align; they are removed again
   later.
2. **Mark the patterns.** In the Patterns window:
    - **Draw on Plot**: drag across the plot; every drag adds a pattern. It
      stays on until you click it again.
    - **Add Current View**: the visible range becomes a pattern.
    - **Repeat…**: copy the selected pattern N times at a fixed period (cipher
      rounds, loop iterations).

    In the table, edit the name, start, end, **length** (editing it moves the
    end), search width (±) and method (cross-correlation or peak).
    **Positions in** switches these between *pipeline samples* — the
    positions of the resampled signal on the plot, the same units as the
    Align Traces dialog — and *raw samples*; patterns are always stored in
    raw samples. The search width defaults to ±¼ of the pattern; a width of
    less than three steps of the resampled signal is shown in amber, since it
    can't catch real misalignment.
    For cross-correlation, **Min corr.** excludes traces whose best
    correlation with the template is lower than this. It defaults to 0.7 for
    new patterns; *off* keeps every trace. Patterns are drawn on the plot as coloured, labelled bands, dashed
    until they are aligned.
3. **Align All**, or select rows (Ctrl/Shift-click) and **Align Selected**.
   Each pattern is aligned in two stages:
    - **coarse**, on the current pipeline (fast on the resampled signal, but
      only accurate to about one resampling step);
    - **refine**, by normalised cross-correlation at full raw resolution within
      one resampling step of the coarse result. The pipeline is used with its
      resamplers removed, so filters still shape the signal.

    The traces default to the ones loaded in the view (first trace onward),
    with that first trace as the template (an absolute trace number, as in
    Align Traces). The resulting shifts are exact to the raw sample. The Result column
    shows, per pattern, how many traces were kept (e.g. *kept 60 / 300 (20%)*,
    amber below 50%, red below 25%) and the median correlation over all
    traces.

    If too few traces are kept, the region or the template is off: select the
    pattern, click **Redraw Selected**, drag the new region (the name and
    settings stay), and **Align Selected** again. By default the pipeline stages are
    removed **once every pattern is aligned**, so you can align them one at a
    time on the resampled view and the analyses still run on raw traces.
    Ctrl+Z undoes this.

    The pipeline the patterns were aligned on is **remembered**: re-aligning
    a pattern later (after Redraw Selected, say) still runs on the resampled
    signal even though the view shows raw traces again. The window shows what
    it aligns on. The final refinement works coarse-to-fine (e.g. 64× → 8× →
    1×), so even long patterns refine to the exact sample quickly.
4. **Run the t-test** (**Analyze → Welch t-test**). With aligned patterns the
   dialog shows **Use patterns** with a checklist. The selected patterns are
   tested on the raw traces with their own shifts, and shown side by side in
   one result tab with labelled bands. First or second order: see
   [Welch t-test](ttest.md#second-order).

**Save Shifts…** / **Load Shifts…** store the patterns and their shifts (see
[file format](#shift-file-format)). The *Alignment & View* box in the side panel
shows how many patterns are aligned; **Discard** removes them.

## Align on something that does not depend on the secret

The shifts are chosen from the traces' own content. If a pattern covers the
very samples where the secret leaks, and nothing else in it lines the traces
up, the alignment can follow the leakage itself. That makes the shift depend
on the data, which can **create** leakage in the test (first or second order)
or hide it.

- Prefer patterns with clear, data-independent structure: the start of an
  operation, a load, a fixed sequence of instructions.
- Align on a smoothed envelope (Absolute Value → Moving Average) rather than
  on raw samples where possible.
- Run the **random-split sanity check** in the t-test dialog. It must show no
  leakage; if it does, the alignment or preprocessing created it.

## In a chain

The Chain Editor's **Add step → Align Patterns (current patterns)** records
the current patterns' definitions. When the chain runs, they are re-aligned
(so the step works on new captures too), optionally followed by clearing the
pipeline. A **Run T-test** step can then use the patterns:

```
Add transform: Window Resample (w=64)
Align Patterns: 11 patterns, then clear pipeline
Run T-test: 2nd order (window 7), patterns
```

## Shift file format

```
# trs-viewer pattern shifts v1
# pattern 0 1000 5000 key load
# pattern 1 9000 12000 round 1
pattern,trace,offset
0,0,3
0,1,-7
1,0,12
…
```

Each `# pattern` line gives the index, raw start, raw end and name. Rows are
in raw samples. Excluded traces are omitted. An older single-alignment file
(`trace,offset` rows) loads as one pattern covering the whole trace.
