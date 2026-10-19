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

    Edit the name, region, search width (± raw samples), method
    (cross-correlation or peak) and an optional minimum correlation in the
    table. Patterns are drawn on the plot as coloured, labelled bands, dashed
    until they are aligned.
3. **Align All.** Each pattern is aligned in two stages:
    - **coarse**, on the current pipeline (fast on the resampled signal, but
      only accurate to about one resampling step);
    - **refine**, by normalised cross-correlation at full raw resolution within
      one resampling step of the coarse result. The pipeline is used with its
      resamplers removed, so filters still shape the signal.

    The resulting shifts are exact to the raw sample. The table shows, per
    pattern, how many traces were kept, the median correlation, and how many
    were excluded by the minimum correlation. By default the pipeline stages are
    then removed, so analyses run on raw traces. Ctrl+Z undoes this.
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
