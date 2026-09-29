#pragma once

// ---------------------------------------------------------------------------
// Narrowband spectral line detection.
//
// Broadband denoising (filtering, SSA, wavelets) and narrowband interference
// removal are different problems. This header covers the second one: finding
// the *discrete spectral lines* in an averaged trace spectrum that come from
// something other than the chip's own switching activity — a radio carrier
// and its harmonics, a DC-DC converter's switching frequency, a reference
// clock leaking into the probe.
//
// Detection has to come before removal. Notching every multiple of an assumed
// f0 is actively dangerous: leakage is *not* confined to clock harmonics
// (Tian et al., "A model of the leakage in the frequency domain", JCEN 2015),
// and screaming-channel work shows exploitable leakage spread over hundreds
// of MHz with non-harmonic frequencies often beating the harmonics outright
// (Wouters et al., CARDIS 2023). So the job here is to report *what is
// actually there*, with a magnitude excess you can judge, and let the caller
// decide what to remove — not to assume a comb and start cutting.
//
// The discriminator is narrowness, not height. An interferer is a few bins
// wide standing on a smooth noise floor; real leakage is broadband and moves
// the floor itself. Both can be tall. So every candidate is measured against
// a *local* baseline (a running median, which a few-bin spike cannot drag
// with it) and rejected if its own width exceeds max_width_bins.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <vector>

// One detected narrowband line.
struct SpectralLine {
    int64_t bin       = 0;    // peak bin
    int64_t lo_bin    = 0;    // inclusive extent, down to half-excess
    int64_t hi_bin    = 0;    // inclusive
    float   mag       = 0.0f; // peak value, in the input spectrum's own units
    float   baseline  = 0.0f; // local median at the peak, same units
    float   excess_db = 0.0f; // how far the peak stands above that baseline

    int64_t widthBins() const { return hi_bin - lo_bin + 1; }
};

struct LineDetectParams {
    // Running-median window, in bins. Must be wider than any line you want
    // to detect (or the line pulls its own baseline up and vanishes) and
    // narrower than the real structure of the noise floor.
    int   baseline_window = 257;

    // Minimum peak-over-baseline excess for a bin run to count as a line.
    float min_excess_db   = 6.0f;

    // Runs wider than this are broadband, not interference — rejected. This
    // is the parameter that stops the detector eating your leakage.
    int   max_width_bins  = 24;

    // Bins below this are ignored. DC and the first few bins hold trace
    // offset and drift, which are not "lines" in any useful sense.
    int   min_bin         = 2;

    // True when spec[] is already in dB (the FFT dialog's "Magnitude (dB)"
    // output), in which case excess is a plain subtraction rather than a
    // log ratio.
    bool  input_is_db     = false;
};

// Runs the detection described above over a one-sided magnitude spectrum.
//
// baseline_out, when non-null, receives the full running-median baseline
// (same length as spec) so a caller can draw it under the spectrum — seeing
// where the baseline sits is how you tell a sane min_excess_db from a silly
// one.
//
// Lines come back sorted by descending excess_db: the worst offender first,
// which is the order you want to work down when deciding what to excise.
// Returns false only on malformed input (empty spectrum, window < 3).
bool detectSpectralLines(const std::vector<float>& spec,
                         const LineDetectParams& params,
                         std::vector<SpectralLine>& out,
                         std::vector<float>* baseline_out,
                         std::string& err);

// Running median of spec[] over a window of `window` bins, evaluated on a
// coarse grid and linearly interpolated between grid points. Exposed
// separately because it is a reasonable "flatten the noise floor" view in
// its own right, and because detectSpectralLines' behaviour is much easier
// to reason about when you can see the baseline it worked from.
//
// Sampling the median on a grid rather than at every bin is deliberate: the
// baseline is meant to be smooth, a stride of window/4 is far finer than any
// structure a median of that width can represent, and it turns an O(N log W)
// sliding-window problem into an O(N) one.
void runningMedianBaseline(const std::vector<float>& spec, int window,
                           std::vector<float>& out);

// Bin index -> Hz, for a one-sided spectrum of `fft_n` input samples taken at
// `sample_rate_hz`. Returns 0 when the sample rate is unknown (<= 0), which
// callers show as plain bin indices.
inline double binToHz(int64_t bin, int64_t fft_n, double sample_rate_hz) {
    if (sample_rate_hz <= 0.0 || fft_n <= 0) return 0.0;
    return static_cast<double>(bin) * sample_rate_hz / static_cast<double>(fft_n);
}

// Hz -> fraction of Nyquist, the unit every filter/excision transform takes
// (see BiquadFilterTransform's cutoff: TRS files can't be relied on to carry
// a sample rate, so the transforms themselves only ever see ratios).
inline double hzToNormalized(double hz, double sample_rate_hz) {
    if (sample_rate_hz <= 0.0) return 0.0;
    return 2.0 * hz / sample_rate_hz;
}

// ---------------------------------------------------------------------------
// Period estimation, for synchronous interference subtraction.
//
// A switching converter, a reference clock or an RFID reader field is
// *periodic* but generally not sinusoidal, so its energy sits in a fundamental
// plus a long tail of harmonics. Removing all of that with notches means a
// notch per harmonic, each costing passband. Estimating the period and
// subtracting the averaged waveform removes the whole shape at once — the
// fundamental, every harmonic, and the parts that were never sinusoidal — and
// punches no holes in the spectrum at all.
//
// The estimate maximises normalised autocorrelation over candidate lags. The
// signal of interest is (by assumption) triggered and so repeats at the trace
// rate, not at the interferer's rate, which is exactly what lets one be
// separated from the other.
// ---------------------------------------------------------------------------

// Best period in samples within [min_period, max_period], as a fractional
// value refined by parabolic interpolation around the winning integer lag —
// a switcher's period almost never lands on a whole number of samples, and
// rounding it visibly smears the folded template.
//
// Returns 0.0 when no candidate lag reaches `min_correlation` (default: the
// caller should treat this as "no periodic interferer found" rather than
// subtracting something meaningless). score_out, when non-null, receives the
// winning normalised autocorrelation so a UI can show how convincing it was.
double estimatePeriod(const float* x, int64_t n,
                      int64_t min_period, int64_t max_period,
                      double min_correlation = 0.3,
                      double* score_out = nullptr);
