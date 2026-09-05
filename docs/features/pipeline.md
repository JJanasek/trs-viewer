# Signal Processing Pipeline

Transforms stack in order and are applied live on every render and on all SCA computations:

| Transform | Parameters | Description |
|---|---|---|
| Absolute Value | — | `\|x\|` point-wise |
| Negate | — | `-x` point-wise |
| Offset | constant | Add a constant to every sample |
| Scale | factor | Multiply every sample |
| Moving Average | window size | Causal sliding-window mean. The first `window − 1` output samples are a startup transient and are skipped when resetting the view |
| Window Resample | window size, overlap | Sliding-window mean advancing by `hop = window × (1 − overlap)`. `overlap = 0` is plain non-overlapping block decimation; `overlap → 1` makes consecutive windows nearly identical (hop is clamped to ≥ 1) |
| Stride Resample | stride | Keep every N-th sample — output is `ceil(N / stride)` samples |
| FFT Magnitude | window function | One-sided amplitude spectrum, `N/2 + 1` bins. Normalised by `N` with non-DC/Nyquist bins doubled, so the output is in the same amplitude units as the input |
| STFT Magnitude | window size, hop *or* overlap, window function | Short-Time Fourier Transform — each window's spectrum concatenated, giving `num_windows × (window/2 + 1)` samples. Intended for frequency-domain CPA |
| Gaussian Noise | relative σ | Adds i.i.d. `N(0, (σ × trace_std)²)`. σ is *relative* to each trace's own standard deviation, so the effect is scale-independent across trace sets |
| Biquad Filter | type, cutoff, Q | 2nd-order IIR filter (RBJ "Audio EQ Cookbook" formulas, Direct Form II Transposed). Types: lowpass · highpass · bandpass · notch |

**Window functions** (FFT and STFT): Rectangular · Hann · Hamming · Blackman.

**Biquad cutoff is normalised** — it is a fraction of Nyquist in `(0, 1)`, not Hz, because TRS files do not reliably carry a real sample rate. `0.1` means "10 % of Nyquist"; on a 1 GS/s capture that is 50 MHz, but the filter only ever sees the ratio. For lowpass/highpass, `Q ≈ 0.707` is the maximally-flat Butterworth response; for bandpass/notch, higher Q means a narrower band.

Transforms that carry state between samples (moving average, both resamplers, the filter, FFT/STFT) are marked *sequential* internally and force the renderer to read samples in order rather than striding.
