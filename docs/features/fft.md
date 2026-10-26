# FFT Spectrum

**Analyze → FFT Spectrum…**

The one-sided spectrum of a range of traces and samples.

| Setting | Options |
|---|---|
| **Spectrum of** | The average trace (one FFT) · Each trace, spectra averaged |
| **Apply last alignment shifts** | Average the traces as aligned (average trace only) |
| **Window function** | None (rectangular) · Hann · Hamming · Blackman |
| **Output** | Magnitude · Magnitude (dB), the default · Phase (rad) |
| **Min/max envelope** | Overlay the per-bin minimum and maximum (each trace only) |

**The average trace** averages the traces first and takes one FFT, so it
is fast even on long traces. Noise that differs from trace to trace averages
out, and what is left is what every trace has in common: the operation
itself and lines locked to the trigger. Align the traces first, or a
jittered operation smears out.

**Each trace** takes one FFT per trace and averages the spectra. It is
slower, but it also shows lines that are not locked to the trigger, such as
a radio carrier or a switching converter, which average out of the average
trace. Use it to look for interference.

The spectrum opens as a tab in the main window, like the t-test and other
results: every plot tool works on it, **Tile Tabs** puts it next to another
result (e.g. *Leakage vs Frequency*, to see which lines carry leakage), and
its toolbar has **Detect spectral lines…**, **Remove frequencies…**,
**Export .npy…** and **Export Figure…**. The tools' windows belong to the
tab and close with it.

## Remove frequencies

**Remove frequencies…** on the spectrum's toolbar takes bands out of the
signal and shows the result in time:

1. Pick the bands. Press **Draw on spectrum →** and drag across each line
   (zoom in first for a narrow one), or type a centre and a width and press
   **Add**. The bands are marked on the spectrum.
2. Choose what to show: **the average trace** the spectrum was made from,
   or **one trace** by number.
3. **Show inverse FFT** opens a tab with the signal before (grey) and
   without the bands (blue).
4. **Add to pipeline…** adds a *Spectral Excise* step with the same bands to
   the trace file's pipeline. You can check them first, and then the main
   view and every analysis use the traces without them.

A removed band is not set to zero: it is filled with the level of the
spectrum either side of it, as the *Spectral Excise* step does. The noise
floor stays where it was and the result doesn't ring. The offset (0 Hz) is
never touched.

Unlike the *FFT Magnitude* pipeline transform — which replaces the traces themselves so downstream SCA runs in the frequency domain — this is a read-only measurement view.
