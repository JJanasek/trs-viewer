# Narrowband interference removal

Radio carriers, DC-DC switching converters, reference clocks and RFID reader
fields all deposit **narrowband** energy in a trace. That is a different problem
from broadband noise, and it wants different tools: the energy sits in a handful
of frequency bins, so it can be taken out almost surgically — or, if it is
carrying the leakage rather than burying it, demodulated instead.

## Detect before you remove

!!! warning "A comb of notches is not a free action"
    Side-channel leakage is **not** confined to clock harmonics. Notching every
    multiple of an assumed `f0` can delete signal as easily as interference.
    Measure SNR or run a t-test both ways rather than assuming it helped.

    See *Leakage vs Frequency* below for the tool that tells the two apart.

**Analyze → FFT Spectrum… → Detect spectral lines…**

Runs over the averaged spectrum and reports every narrowband line standing above
the **local** noise floor.

| Setting | Meaning |
|---|---|
| **Baseline window** | Width of the running-median floor estimate, in bins. Must be wider than any line you want to find, narrower than real structure in the floor. |
| **Min excess (dB)** | How far above that floor a peak must stand. |
| **Max width (bins)** | Peaks wider than this are treated as broadband and ignored. |
| **Ignore bins below** | Skips DC and drift, which are not lines in any useful sense. |

The width limit is what separates interference from signal: a carrier is a few
bins wide standing on a smooth floor, whereas real leakage is broadband and
moves the floor itself. Both can be tall, so height alone decides nothing.

Detected lines are listed strongest-first, marked on the spectrum, and can be
handed straight to the excision stage with **Excise checked…**.

## The four removal stages

All four are ordinary pipeline stages (**Processing Pipeline → Add**), so they
apply to every analysis downstream and are saved in a [chain](chain.md).

| Stage | Use when |
|---|---|
| **Spectral Line Excision** | You have the whole trace and know where the lines are. The best general choice. |
| **Harmonic Notch** | You know `f0` and want a comb, or the trace is too long to transform whole. |
| **Periodic Subtract** | The interferer is periodic but *not* sinusoidal — a switching converter, a reader field. |
| **Demodulate** | The carrier is the *transport* of the leakage, not interference. |

### Spectral Line Excision

Replaces the magnitude inside each band with the floor either side of it,
leaving phase untouched. Compared with a notch filter this costs **no phase
distortion and no time-axis smearing**, can be as narrow as a single bin, and
handles any number of lines in one pass. It needs the whole trace and `O(N log N)`
work per trace.

Interpolating the magnitude rather than zeroing the bins is deliberate: a hole
punched in the noise floor is itself a spectral feature and rings in the time
domain.

!!! tip "Widen the band for an off-bin interferer"
    A carrier that does not land exactly on an FFT bin has its energy spread
    across neighbours by the window's sinc skirts. Too narrow a band leaves a
    visible residue — the fix is a wider band, not a different tool.

### Harmonic Notch

A cascade of notches at `f0, 2·f0, 3·f0, …`. **Zero-phase by default**, and an
**odd-harmonics-only** mode, because a square-ish clock or switching converter
puts most of its energy into odd multiples and notching the even ones spends
passband for very little return. Multiples at or past Nyquist are dropped.

### Periodic Subtract

Folds the trace at the interferer's period, averages to recover its waveform, and
subtracts that. Removes the fundamental **and every harmonic**, including
whatever was never sinusoidal, while leaving every bin of the spectrum intact —
no notches, no holes.

This works because the cryptographic operation is triggered and so repeats at the
*trace* rate, while a free-running interferer repeats at its own. If the two are
locked together — a converter clocked by the device — this will remove signal, so
check both ways.

The period may be given in samples, derived from a frequency, or detected
automatically. Fractional periods are supported and usually necessary: a
switcher's period rarely lands on a whole number of samples.

### Demodulate

Selects the band around a centre frequency, shifts it to baseband and reports its
envelope. This is the stage for the case where switching noise has been coupled
into an RF front end, amplified, modulated onto the carrier and broadcast — where
notching the carrier would delete the signal. The same applies to clock-harmonic
sidebands, where activity appears at `f_clk ± n/T` rather than at `f_clk`.

## Zero-phase filtering

The existing **Filter** stage gained a **Zero-phase** option, and it is worth
understanding even if you never touch the stages above.

A causal filter does not merely attenuate — it *delays different frequencies by
different amounts*, and a high-Q notch's delay spikes sharply right at the notch.
Every analysis here is indexed by sample, so that smears leakage along the time
axis and can look like a genuine change in where a device leaks.

Filtering forwards and then backwards cancels the phase exactly. Two things
change:

- the stop band becomes **twice as deep in dB**, so design for half
- the whole trace is needed at once, so a chunked export switches to whole-trace reads

## Leakage vs Frequency

**Analyze → Leakage vs Frequency…**

Runs a Welch t-test at every *(time window, frequency bin)* of an STFT, then
reports the largest `|t|` each frequency ever reached.

This is the counterpart to line detection, and the reason detection can afford to
be conservative:

- a tall spectral line with **no** t-statistic is interference — excise it freely
- a band that **lights up** here is the leakage itself — filtering it away is the expensive mistake

The **Time-frequency map…** button shows the full plane, so you can also see
*when* a band leaks. A band lit in one window is a specific operation; one lit
throughout is more likely a property of the whole capture.

The reduction takes a maximum rather than a mean over time, because leakage is
usually confined to the few windows where the operation of interest happens, and
averaging over a long quiet capture would bury it.

### Misaligned traces

The analysis honours the last alignment (including tiled alignment), exactly as
the t-test does. But alignment deserves more thought here than elsewhere,
because the failure is silent: a diluted `|t|` reads exactly like "this device
does not leak".

STFT magnitude discards phase, and a time shift is a phase ramp, so a shift
**smaller than one window** is largely absorbed. That is the whole reason
frequency-domain analysis is said to tolerate misalignment — and it holds only
while the leaking event stays inside the same window in every trace. Once the
residual jitter pushes it across a window boundary, its energy splits between
two windows and both statistics fall.

Measured on the test fixture, peak `|t|`:

| STFT window | Jitter | peak \|t\| |
|---|---|---|
| 512 | 0 (aligned) | 172 |
| 512 | ±16 | 171 — absorbed |
| 256 | ±8 | 162 |
| 256 | ±400 | 16 — diluted |
| 128 | ±192 | 29 |
| 1024 | ±192 | **82** — same jitter, longer window |

The last pair is the actionable one: at fixed jitter, an 8× longer window
recovers roughly 3× the statistic. So **size the window for the jitter, not for
the frequency resolution you would like**, and raise the overlap to 0.75 so an
event near a boundary is still well inside some window.

The dialog reports the residual shift spread it actually saw and warns when the
window is under 4 σ of it.

!!! danger "Aligning on a single pattern biases the time axis"
    If you aligned on one pattern you know leaks, the alignment holds near that
    anchor and decays away from it — jitter accumulates with distance. The
    result then shows leakage concentrated near the anchor **not because the
    device leaks there**, but because that is the only place you can still
    measure.

    Diagnose it with the **time-frequency map**: `|t|` falling off roughly
    symmetrically either side of the anchor is an alignment artefact. Real
    leakage does not know where you put your trigger. Confirm by re-running
    anchored on a different pattern — if the leaking *time* follows the anchor
    while the leaking *frequency* stays put, the time axis is telling you about
    your alignment, not your chip.

    The fix is [tiled alignment](alignment.md): align each tile independently so
    the alignment holds across the whole region instead of only near one point.

## Frequency units

Every dialog here accepts frequencies in **Hz / kHz / MHz / GHz** or in
**× Nyquist**, and shows both. The transforms themselves only ever store
fractions of Nyquist, because TRS files cannot be relied on to carry a sample
rate — which is also why each dialog has a sample-rate row. If the file's own
`XS` metadata is present it is used automatically; otherwise set one by hand.

## Worked example

A capture from a board whose DC-DC converter runs near 2 MHz, sampled at
125 MS/s:

1. **Analyze → FFT Spectrum…**, output *Magnitude (dB)*, run over the whole trace.
2. **Detect spectral lines…** — a comb at 2 MHz and its multiples is listed,
   strongest first.
3. **Analyze → Leakage vs Frequency…** — the comb bins show no t-statistic, so they
   are interference rather than signal.
4. Back in the spectrum window, **Excise checked…**, then **OK**.
5. Re-run the t-test. On a synthetic capture built this way, peak `|t|` in the
   leakage window goes from **2.0** (invisible, below the ±4.5 threshold) to
   **9.9**; the same three bands excised from an interference-free capture cost
   **0.04 %** of the leakage signal.

Steps 4–5 are a pipeline stage and an analysis, so both are saved in a
[chain](chain.md) and replay with one click.
