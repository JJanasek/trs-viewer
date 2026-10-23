# FFT Spectrum

**Analyze → FFT Spectrum…**

Averaged one-sided spectrum over a range of traces.

| Setting | Options |
|---|---|
| **Window function** | None (rectangular) · Hann · Hamming · Blackman |
| **Output** | Magnitude · Magnitude (dB) · Phase (rad) |
| **Min/max envelope** | Overlay the per-bin minimum and maximum alongside the average |

The spectrum opens as a tab in the main window, like the t-test and other
results: every plot tool works on it, **Tile Tabs** puts it next to another
result (e.g. *Leakage vs Frequency*, to see which lines carry leakage), and
its toolbar has **Detect spectral lines…**, **Export .npy…** and **Export
Figure…**. The line detector's window belongs to the tab and closes with it.

Unlike the *FFT Magnitude* pipeline transform — which replaces the traces themselves so downstream SCA runs in the frequency domain — this is a read-only measurement view.
