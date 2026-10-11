# Signal Processing Pipeline

The pipeline is an ordered list of stages applied to every trace, live. Each
stage takes the output of the one above it. The result is what you see in the
plot **and** what every analysis and export reads: t-test, SNR, CPA,
cross-correlation, FFT, alignment and exports. Nothing is written back to the
file, so you can add, reorder and remove stages freely.

## Using it

1. In the side panel, pick a stage from the **Processing Pipeline** drop-down.
   A one-line summary appears under the drop-down. Hover it, or an entry in
   the drop-down, to see when to use the stage and what to watch out for;
   **More…** jumps to its section below.
2. Click **Add**. Stages that need settings open a dialog; hover any field there
   for an explanation.
3. Reorder with **↑ / ↓**, remove with **Remove**. Hover a stage in the list to
   see its help again. **Ctrl+Z** undoes pipeline changes.

The status bar shows how many stages are active, and the **File** box shows the
trace length after the pipeline when a stage changes it.

!!! tip "Order matters"
    `Absolute Value → Moving Average` gives a smooth envelope.
    `Moving Average → Absolute Value` gives a smoothed signal folded upwards,
    which is a different thing. Similarly, apply a lowpass **Filter** *before*
    **Stride Resample**, never after.

## Which stage do I need?

| Goal | Stages |
|---|---|
| See operations more clearly / get something to align on | Absolute Value → Moving Average |
| Less high-frequency noise | Filter (lowpass), Moving Average |
| Remove slow drift or DC wander | Filter (highpass) |
| Smaller, faster traces from an oversampled capture | Window Resample |
| Analyse misaligned traces in the frequency domain | FFT Magnitude (no timing) or STFT Magnitude (coarse timing) |
| Remove a clock or DC-DC converter tone | Spectral Line Excision, Harmonic Notch, Periodic Subtract |
| Leakage rides on a carrier (EM probe, RF coupling) | Demodulate |
| Check how robust a leak is | Gaussian Noise (temporarily) |

The four interference stages have a dedicated guide with a worked example:
[Narrowband interference removal](interference.md).

## Overview

| Stage | Settings | Changes trace length? |
|---|---|---|
| [Absolute Value](#absolute-value) | none | no |
| [Negate](#negate) | none | no |
| [Moving Average](#moving-average) | window | no |
| [Window Resample](#window-resample) | window, overlap | shorter |
| [Stride Resample](#stride-resample) | stride | shorter |
| [Offset](#offset) | constant | no |
| [Scale](#scale) | factor | no |
| [FFT Magnitude](#fft-magnitude) | window function | `N/2 + 1` bins |
| [STFT Magnitude](#stft-magnitude) | window, overlap, window function | `windows × (window/2 + 1)` |
| [Gaussian Noise](#gaussian-noise) | relative σ | no |
| [Filter](#filter) | type, cutoff, Q, zero-phase | no |
| [Harmonic Notch](#harmonic-notch) | f0, harmonics, odd-only, Q, zero-phase | no |
| [Spectral Line Excision](#spectral-line-excision) | list of bands, phase mode | no |
| [Periodic Subtract](#periodic-subtract) | period (auto / samples / frequency) | no |
| [Demodulate](#demodulate) | centre, half-bandwidth, output | no |

---

### Absolute Value

`|x|` for every sample.

- **Use for:** rectifying before smoothing or resampling, so positive and
  negative swings don't cancel. `Absolute Value → Moving Average` produces an
  energy-like envelope where rounds and operations stand out, which also makes a
  good signal to [align](alignment.md) on.
- **Watch out:** the sign is lost. Remove it again (or use Ctrl+Z) before
  analyses where the sign of the leakage matters.

### Negate

`−x` for every sample.

- **Use for:** flipping probe polarity so the peaks you align on point upwards.
- **Note:** t-test, SNR and correlation magnitudes are unchanged; only signs flip.

### Moving Average

Causal sliding-window mean of the last `w` samples.

- **Use for:** suppressing high-frequency noise; building an envelope together
  with Absolute Value.
- **Watch out:** a causal average delays features by about `w/2` samples. The
  first `w − 1` output samples are a start-up transient and are skipped when the
  view is reset. A window wider than a leakage peak blurs it away.

### Window Resample

Sliding-window mean that advances by `hop = window × (1 − overlap)`.
`overlap = 0` is plain block averaging (decimation by `window`); `overlap → 1`
makes consecutive windows nearly identical (the hop is clamped to ≥ 1). The
dialog shows the resulting trace length.

- **Use for:** shrinking long, oversampled captures so analyses run faster and
  need less memory, without throwing energy away.
- **Watch out:** time resolution drops by the hop; the averaging also acts as a
  low-pass filter.

### Stride Resample

Keeps every N-th sample; output length is `ceil(N / stride)`.

- **Use for:** quick decimation of data that is already band-limited.
- **Watch out:** there is no averaging, so noise above the new Nyquist
  frequency **aliases** into the result. Prefer Window Resample, or put a lowpass
  Filter first.

### Offset

Adds a constant to every sample.

- **Use for:** removing a DC level or lining up baselines for display.
- **Note:** t-test, SNR and correlation are unaffected by a constant offset.

### Scale

Multiplies every sample by a constant.

- **Use for:** converting ADC codes to volts, or normalising for display.
- **Note:** t-test and correlation are unaffected (a negative factor only flips
  the sign).

### FFT Magnitude

Replaces each trace with its one-sided amplitude spectrum, `N/2 + 1` bins.
Normalised by `N` with non-DC/Nyquist bins doubled, so the output is in the same
amplitude units as the input. Window functions: Rectangular · Hann · Hamming ·
Blackman.

- **Use for:** frequency-domain analysis of misaligned traces. The magnitude
  spectrum ignores time shifts, so a t-test or CPA afterwards tells you *which
  frequencies* leak even when the traces don't line up.
- **Watch out:** all timing is gone. To see *when* and *at which frequency*, use
  STFT Magnitude or **Analyze › Leakage vs Frequency**. To just look at a
  spectrum without changing the traces, use **Analyze › FFT Spectrum** instead.

### STFT Magnitude

Short-time Fourier transform: the spectrum of each sliding window,
concatenated, giving `num_windows × (window/2 + 1)` samples per trace.

- **Use for:** frequency-domain t-test or CPA that keeps coarse timing and
  tolerates jitter smaller than one window.
- **Watch out:** size the window for the **jitter**, not for the frequency
  resolution you would like, and use overlap ≈ 0.75 so an event near a window
  edge is still inside some window. See
  [Misaligned traces](interference.md#misaligned-traces) for measurements.

### Gaussian Noise

Adds independent noise `N(0, (σ × trace_std)²)`. σ is *relative* to each
trace's own standard deviation, so the effect is the same across trace sets.

- **Use for:** robustness checks. How much extra noise before the leakage
  disappears? Roughly how would a noisier setup behave?
- **Watch out:** never leave it in for a real assessment. It is random on every
  run.

### Filter

2nd-order IIR filter (RBJ "Audio EQ Cookbook" formulas, Direct Form II
Transposed). Types: lowpass · highpass · bandpass · notch.

- **Use for:**
    - **lowpass**: remove high-frequency noise;
    - **highpass**: remove slow drift and DC wander;
    - **bandpass**: isolate a band such as the clock frequency;
    - **notch**: remove a single interfering tone.
- **Cutoff is normalised.** It is a fraction of Nyquist in `(0, 1)`, not Hz,
  because TRS files don't reliably carry a sample rate. `0.1` means 10 % of
  Nyquist; on a 1 GS/s capture that is 50 MHz.
- **Q:** for lowpass/highpass, `Q ≈ 0.707` is the maximally flat (Butterworth)
  response. For bandpass/notch, higher Q means a narrower band.
- **Zero-phase:** a causal filter delays different frequencies by different
  amounts, which smears leakage along the sample axis. Zero-phase filters
  forwards then backwards, cancelling the phase exactly; the stop band becomes
  twice as deep in dB and the whole trace is needed at once. See
  [Zero-phase filtering](interference.md#zero-phase-filtering).

### Harmonic Notch

A cascade of notches at `f0, 2·f0, 3·f0, …`, zero-phase by default, with an
odd-harmonics-only option. Multiples at or past Nyquist are dropped.

- **Use for:** a clock or DC-DC converter at a known frequency.
- **Watch out:** leakage can live at those harmonics too. Measure SNR or run a
  t-test before and after. Details:
  [Harmonic Notch](interference.md#harmonic-notch).

### Spectral Line Excision

Replaces the magnitude inside each listed band with the noise floor either side
of it, leaving phase untouched. No phase distortion, no time smearing, and a
band can be as narrow as one bin.

- **Use for:** the best general way to remove narrowband lines. Find them with
  **Analyze › FFT Spectrum › Detect spectral lines**, then **Excise checked…**
  adds this stage with the bands filled in. You can also add bands or a harmonic
  comb by hand.
- **Watch out:** needs the whole trace per pass. Widen the band for a carrier
  that falls between FFT bins. Details:
  [Spectral Line Excision](interference.md#spectral-line-excision).

### Periodic Subtract

Folds the trace at the interferer's period, averages to recover its waveform,
and subtracts it. The period can be detected automatically, entered in samples
(fractional values allowed), or derived from a frequency.

- **Use for:** periodic but non-sinusoidal interference, such as a switching
  converter. It removes the fundamental and every harmonic without punching
  notches into the spectrum.
- **Watch out:** if the interferer is locked to the device's clock or trigger,
  this removes signal. Details:
  [Periodic Subtract](interference.md#periodic-subtract).

### Demodulate

Selects the band around a centre frequency, shifts it to baseband and outputs
its envelope (or the baseband real part).

- **Use for:** when the leakage rides **on** a carrier instead of the carrier
  being noise. Examples are switching noise coupled into an RF front end, or
  clock sidebands at `f_clk ± n/T`.
- **Watch out:** notching that carrier would delete the signal. Make the
  half-bandwidth wide enough to contain the modulation. Details:
  [Demodulate](interference.md#demodulate).

---

## Frequencies and sample rate

The interference stages accept frequencies in Hz / kHz / MHz / GHz or in
× Nyquist, and show both. Stored values are always fractions of Nyquist. The
sample rate comes from the file's `XS` metadata when present; otherwise enter it
in the dialog's sample-rate row.

## Performance note

Moving Average, both resamplers, Filter, FFT and STFT carry state from sample to
sample, so the renderer reads their input in order instead of striding.
Spectral Line Excision, Periodic Subtract, Demodulate, and Filter / Harmonic
Notch with **Zero-phase** need the whole trace at once, so chunked exports
switch to whole-trace reads when one of them is in the pipeline.
