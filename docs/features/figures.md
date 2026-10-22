# Figures for papers and slides

**Right-click any plot → Export Figure…** (or **File → Export → Figure**,
Ctrl+Shift+S). **Copy Figure** puts it on the clipboard.

Every plot in the app (traces, t-test and SNR results, live t-test, FFT,
loaded curves) exports through the same renderer. It draws the **visible
range** as a vector figure at its final printed size, so it can go straight
into LaTeX without scaling, and its text stays at the size you chose.

## What the figure is for

**For:** picks the size and fonts for where the figure goes, so it fits
without scaling. The line under the list says what each choice means.

| For | Size | Font |
|---|---|---|
| **Paper** — Springer conference paper (LNCS: CARDIS, COSADE, CT-RSA, older CHES), page / half width | 12.2 × 4.5 / 5.9 × 3.0 cm | serif 8 / 7 pt |
| **Paper** — CHES / TCHES paper (IACR journals), page / half width | 13.6 × 4.8 / 6.6 × 3.2 cm | serif 8 / 7 pt |
| **Paper** — IEEE paper, one column / across both columns | 8.8 × 4.0 / 18.1 × 5.0 cm | Times 8 pt |
| **Thesis** on A4 — page width | 15 × 5.5 cm | serif 9 pt |
| **Slides** — LaTeX (Beamer) 16:9 full / half, 4:3 full | 14 × 6 / 6.8 × 5 / 10.8 × 5.5 cm | sans 9 / 8 pt |
| **Slides** — PowerPoint 16:9 full / half | 30 × 12 / 15 × 10 cm | sans 16 pt |

Changing the size or font by hand switches to **Custom size**. In LaTeX,
`\includegraphics[width=\textwidth]{figure.pdf}` (or `\columnwidth`; the
dialog shows which) places it at exactly its exported size.

## Axis units

- **X axis: Sample index or Time.** Time comes from the TRS file's sample
  interval (the `SCALE_X` tag, e.g. 2 ns per sample at 500 MS/s), shown with a
  fitting unit: *Time [ms]*, *Time [µs]*. If the file has none, type the
  sampling rate. t-test results remember where their window started, so their
  time axis is absolute (a window from sample 10,000,000 at 2 ns starts at
  20 ms). Joined pattern results stay in samples.
- **Y axis (raw traces): Raw values or Voltage**, from the file's value scale
  (`SCALE_Y`, e.g. 0.39 mV per ADC step): *Voltage [mV]*.

Choosing a unit fills in the matching label, which you can still edit.

## Style

The defaults follow how side-channel figures look in published papers:

- **Data drawn thin and black**; colour is kept for the threshold. Several
  overlaid traces stay black, and their legend is left to the caption.
  Untick *Data in black* for slides to keep the plot's colours.
- **TVLA threshold** as a red dashed line, labelled on the y axis (4.5).
- **Full frame, inward ticks, no grid** (a light grid is optional).
- Plain tick numbers; a common power of ten moves to the end of the axis
  (×10⁵) instead of k/M suffixes.
- **No title**: figures get their caption in LaTeX. The X and Y labels can be
  edited in the dialog, e.g. *Time samples* and *t-value*.
- Dense data (millions of samples) is drawn as a min/max envelope at figure
  resolution, so PDFs stay small.

**Fonts**: *Serif — like LaTeX papers* uses Latin Modern when it is installed
(an exact match for LaTeX's default font), otherwise a Times-like serif.
*Times — like IEEE papers and Word* suits IEEE, and *Sans* suits slides. The dialog shows which font is used.

## Formats

| Format | Use |
|---|---|
| **PDF** (vector, fonts embedded) | LaTeX papers, theses, Beamer |
| **SVG** (vector) | PowerPoint, LibreOffice, Inkscape |
| **PNG** (300–600 dpi, resolution stored in the file) | Word, web, anywhere else |

The clipboard copy carries both SVG and a 300 dpi PNG.
