# Several files as one set

**File → Open Several Files as One…** (or **Open Several as One…** on the
start page)

Captures are often split into several files, e.g. 2,000 traces as four files
of 500, so a crash loses only the last part. This opens the files as **one
trace set**: the traces of the first file, then of the second, and so on.
Nothing is copied: each file is memory-mapped as usual, and reads go to the
file that holds the trace.

Trace numbers run on from file to file (with four files of 500, trace 750 is
trace 250 of the second file). Everything works on the whole set: alignment,
patterns and saved shift files, the t-test, SNR, CPA, export and chains.

The files can be TRS, `.npy` (with the [labels](npy-npz.md) next to it) or
`.npz`. Several runs of a capture, each a folder with `traces.npy` and
`labels.npz`, become one set with their labels, so a t-test on a bit of a key
uses the traces of all of them. Only a `.npz` is loaded into memory, as it is
when opened alone.

## The dialog

The dialog lists the files sorted by name, with numbers in numeric order
(`capture_2` before `capture_10`). For each file it shows its traces and which
trace numbers they get in the set. A `.npy` or `.npz` is shown with its folder
(`run_1/traces.npy`), since captures tend to share the file name.

- **Add Folders…** takes capture folders; select several with Ctrl or Shift. A
  folder gives its `traces.npy` or `traces.npz` if it has one, otherwise its
  `.trs`, `.npy` and `.npz` files, leaving out archives of labels. A folder
  with none of those, such as the one holding all your runs, gives the
  `traces.npy` or `traces.npz` of each folder directly inside it.
- **Add Files…** takes single files.
- **Move Up / Move Down / Sort by Name** change the order; **Remove** takes
  files out.
- A file shown **in red** can't be read together with the first one: it has a
  different number of samples per trace, sample coding, data bytes per trace
  or trace parameters (for a `.npy`, its labels must have the same names, types
  and shapes). Remove it to open the rest.
- A file whose capture stopped early (**amber**) is used up to its last
  complete trace. Its header may promise more traces than were written, or
  none; either way only the complete ones are read.
- A different time or value scale than the first file is noted; the first
  file's scale is used.

## Reopening: the file list (.trslist)

With **Save the file list** ticked (the default), the dialog writes a small
text file in the folder that holds them all, named after what the files share
(`capture_0.trs` … `capture_15.trs` → `capture.trslist`; `run_1/traces.npy`,
`run_2/traces.npy` → `run.trslist`). It appears in
**Open Recent**, opens with **File → Open File…**, by drag and drop, or
as a command-line argument, like a `.trs` file. The tab is named after it.

```
# trs-viewer trace list: these files are opened as one trace set, in this order.
# Paths are relative to this file's folder (or absolute).
capture_0.trs
capture_1.trs
capture_10.trs
capture_15.trs
```

One file per line, in the order to read them; relative paths are relative to
the list's folder, so the folder can be moved as a whole. Lines starting with
`#` are comments. You can write or edit the list by hand, e.g. to add the
next part of a capture.

## In the side panel

*File* shows **Files: N (one set)**; hovering the file name lists each file
with its trace numbers. The trace data navigator shows which file and which
trace within it the current trace is.

## One real file

To get a single `.trs` (for another tool, or to share), use
**[Export TRS](export.md)** on the combined set: it writes all traces, or a
range, into one file.
