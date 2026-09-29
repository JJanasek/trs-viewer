#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unsupported/Eigen/FFT>

// ---------------------------------------------------------------------------
// Base interface for all signal transforms.
// Transforms operate in-place on a float32 buffer.
// ---------------------------------------------------------------------------
class ITransform {
public:
    virtual ~ITransform() = default;

    virtual std::string name() const = 0;

    // Apply transform in-place.
    // Returns the number of valid output samples written to buf (may be less
    // than count if the transform decimates).
    // sample_offset: absolute sample index of buf[0] in the original trace.
    virtual int64_t apply(float* buf, int64_t count, int64_t sample_offset = 0) = 0;

    // Given an input sample count, return the number of output samples produced.
    // Most transforms return input_count unchanged; decimating transforms return less.
    virtual int64_t transformedCount(int64_t input_count) const { return input_count; }

    // Reset accumulated state — called before each fresh trace pass.
    virtual void reset() {}

    // Returns true if this transform has inter-chunk state (e.g. a causal filter).
    // Strided-sampling cannot be used with such transforms.
    virtual bool requiresSequential() const { return false; }

    // Returns true if apply() is only meaningful when handed a whole trace in
    // one call, because it makes two passes over the samples or transforms
    // them as a block: a reverse pass (zero-phase filtering), an FFT of the
    // entire trace (spectral line excision), a statistic estimated over the
    // full record (periodic-interference subtraction).
    //
    // requiresSequential() is not enough to express this. A causal IIR filter
    // is sequential but perfectly happy to be fed 256K samples at a time,
    // carrying its two delay registers across the boundary. These transforms
    // are not: split the trace and each piece gets its own reverse pass, its
    // own spectrum, its own interference estimate, and the seams show.
    //
    // Callers that chunk (the TRS export writer is the only one today — the
    // plot and every analysis already apply the pipeline to a whole window in
    // a single call) must feed the whole trace at once when any stage says
    // true. See pipelineRequiresWholeTrace() below.
    virtual bool requiresWholeTrace() const { return false; }

    // Number of leading output samples that are part of a startup transient
    // and should be skipped when resetting the view to "valid" data.
    virtual int64_t startupSamples() const { return 0; }

    virtual std::shared_ptr<ITransform> clone() const = 0;
};

// ---------------------------------------------------------------------------
// True when any stage must see a whole trace per apply() call — see
// ITransform::requiresWholeTrace(). Chunking callers check this once per run
// and size their chunk to the full trace when it holds.
// ---------------------------------------------------------------------------
inline bool pipelineRequiresWholeTrace(
    const std::vector<std::shared_ptr<ITransform>>& pipeline)
{
    for (const auto& t : pipeline)
        if (t && t->requiresWholeTrace()) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Pipeline-domain <-> raw-domain sample count conversion.
//
// transformedCount() only goes forward (raw count -> processed count); these
// let UI code that wants to think in the smaller, pipeline-processed domain
// (e.g. a decimating window-resample stage) work out the raw sample range
// that actually needs to be read, instead of the caller guessing a raw
// count and hoping it's enough (or reading far more than necessary).
// ---------------------------------------------------------------------------

// Forward: raw sample count -> processed sample count, chaining every stage.
inline int64_t processedCountForRaw(
    const std::vector<std::shared_ptr<ITransform>>& pipeline, int64_t raw_count)
{
    int64_t n = raw_count;
    for (const auto& t : pipeline) n = t->transformedCount(n);
    return n;
}

// Inverse: the smallest raw sample count (capped at max_raw_count) whose
// processed output is >= desired_processed_count. Exact for the common
// decimating/expanding stages (window/stride resample, FFT/STFT) since their
// transformedCount() is monotonically non-decreasing in its input; found by
// binary search rather than inverting each transform's formula individually,
// so it works for any pipeline without special-casing per transform type.
inline int64_t rawCountForProcessedCount(
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int64_t desired_processed_count, int64_t max_raw_count)
{
    if (pipeline.empty() || desired_processed_count <= 0)
        return std::clamp<int64_t>(desired_processed_count, 0, max_raw_count);
    int64_t lo = 0, hi = max_raw_count;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (processedCountForRaw(pipeline, mid) >= desired_processed_count) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// Point-wise transforms  (no inter-sample state, safe for strided sampling)
// ---------------------------------------------------------------------------

class AbsTransform : public ITransform {
public:
    std::string name() const override { return "Absolute Value"; }
    int64_t apply(float* buf, int64_t count, int64_t) override;
    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<AbsTransform>(*this);
    }
};

class NegateTransform : public ITransform {
public:
    std::string name() const override { return "Negate"; }
    int64_t apply(float* buf, int64_t count, int64_t) override;
    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<NegateTransform>(*this);
    }
};

class OffsetTransform : public ITransform {
public:
    explicit OffsetTransform(float offset = 0.0f) : offset_(offset) {}
    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t) override;
    void setOffset(float v) { offset_ = v; }
    float offset() const    { return offset_; }
    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<OffsetTransform>(*this);
    }
private:
    float offset_;
};

class ScaleTransform : public ITransform {
public:
    explicit ScaleTransform(float scale = 1.0f) : scale_(scale) {}
    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t) override;
    void setScale(float v) { scale_ = v; }
    float scale() const    { return scale_; }
    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<ScaleTransform>(*this);
    }
private:
    float scale_;
};

// ---------------------------------------------------------------------------
// Windowed / sequential transforms
// ---------------------------------------------------------------------------

// Causal moving average over a sliding window of `window_size` samples.
// Output has the same number of samples as input (smoothing filter).
// The first window_size-1 output samples are a startup transient.
class MovingAverageTransform : public ITransform {
public:
    explicit MovingAverageTransform(int window_size = 64);

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    void reset() override;
    bool requiresSequential() const override { return true; }
    int64_t startupSamples() const override { return window_size_ - 1; }

    void setWindowSize(int w);
    int  windowSize() const { return window_size_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<MovingAverageTransform>(*this);
    }

private:
    int   window_size_;
    std::vector<float> ring_;
    double ring_sum_;
    int    ring_pos_;
    int64_t ring_count_;
};

// ---------------------------------------------------------------------------
// Window Resample: slide a `window_size`-sample window across the trace,
// advancing by `hop_size = window_size * (1 - overlap)` samples each step,
// and replace each window with its mean. overlap = 0 (default) gives the
// original non-overlapping block-decimation behaviour; overlap → 1 makes
// consecutive windows nearly identical (hop clamped to >= 1 sample).
// Output has num_windows = (N - window_size) / hop_size + 1 samples.
// ---------------------------------------------------------------------------
class WindowResampleTransform : public ITransform {
public:
    explicit WindowResampleTransform(int window_size = 64, float overlap = 0.0f);

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    int64_t transformedCount(int64_t input_count) const override;
    void reset() override;
    bool requiresSequential() const override { return true; }
    // No startup transient — valid output from first complete window.
    int64_t startupSamples() const override { return 0; }

    void  setWindowSize(int w);
    int   windowSize() const { return window_size_; }
    void  setOverlap(float o);
    float overlap()    const { return overlap_; }
    // hop = window_size * (1 - overlap), rounded, clamped to >= 1.
    int   hopSize()     const;

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<WindowResampleTransform>(*this);
    }

private:
    int    window_size_;
    float  overlap_;   // 0 = no overlap; approaches 1 = near-total overlap

    // Streaming state — a ring buffer holding the current window's samples
    // (needed even for the non-overlapping case, since chunked callers, e.g.
    // the zoomed-out plot renderer, feed this transform one chunk at a time)
    std::vector<float> ring_;
    double  ring_sum_      = 0.0;
    int     ring_pos_      = 0;
    int64_t ring_count_    = 0;   // samples currently held (caps at window_size_)
    int64_t samples_seen_  = 0;   // total raw samples fed since last reset()
    int64_t next_emit_at_  = 0;   // absolute index of the next window's last sample
};

// ---------------------------------------------------------------------------
// Stride Resample: keep every stride-th sample.
// Output has ceil(N / stride) samples — identical to XCorr's stride method.
// ---------------------------------------------------------------------------
class StrideResampleTransform : public ITransform {
public:
    explicit StrideResampleTransform(int stride = 4);

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    int64_t transformedCount(int64_t input_count) const override;
    void reset() override;
    bool requiresSequential() const override { return true; }
    int64_t startupSamples() const override { return 0; }

    void setStride(int s);
    int  stride() const { return stride_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<StrideResampleTransform>(*this);
    }

private:
    int stride_;
    int pos_ = 0;  // position mod stride within the trace; 0 = emit sample
};

// ---------------------------------------------------------------------------
// FFT Magnitude: compute the one-sided amplitude spectrum.
// Input:  N time-domain samples
// Output: N/2+1 magnitude values  (DC at index 0, Nyquist at index N/2)
// Magnitudes are normalised by N and one-sided bins are doubled so that
// the result is in the same amplitude units as the input waveform.
// ---------------------------------------------------------------------------
class FFTMagnitudeTransform : public ITransform {
public:
    enum class Window { Rectangular, Hann, Hamming, Blackman };

    explicit FFTMagnitudeTransform(Window win = Window::Hann) : window_(win) {}

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    int64_t transformedCount(int64_t input_count) const override {
        return input_count / 2 + 1;
    }
    void reset() override {}
    bool requiresSequential() const override { return true; }

    void   setWindow(Window w) { window_ = w; }
    Window window()      const { return window_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<FFTMagnitudeTransform>(*this);
    }

private:
    Window window_;

    // Cached across calls: rebuilding the window envelope and FFT plan is
    // the dominant cost when the same instance is applied to many traces in
    // a row (e.g. t-test/CPA accumulation), so only rebuild when the input
    // size or window type actually changes.
    std::vector<float> window_cache_;
    int64_t            window_cache_size_ = -1;
    Window             window_cache_type_ = Window::Rectangular;
    Eigen::FFT<float>  fft_;
};

// ---------------------------------------------------------------------------
// STFT Magnitude: Short-Time Fourier Transform for frequency-domain CPA.
//
// The trace is divided into overlapping windows of length `window_size`.
// Each window is FFT'd independently and the magnitude at every frequency
// bin is written out.  The result is a new trace whose X-axis is
// (window_index * bins + bin_index), where bins = window_size/2+1.
//
// This preserves *both* time (which window) and frequency (which bin), so
// CPA can pinpoint the exact (time-window, frequency-bin) pair that leaks.
// It naturally defeats clock-jitter because per-window FFT magnitude is
// shift-invariant within that window.
//
// Output length = num_windows * (window_size/2 + 1)
//   where  num_windows = max(0, (N - window_size) / hop_size + 1)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Gaussian Noise: adds i.i.d. Gaussian noise N(0, σ²) to every sample.
// Stateless between traces; σ is in the same units as the trace amplitude.
// ---------------------------------------------------------------------------
// noise_std is a relative factor: actual noise = N(0, (noise_std × trace_std)²)
// so the effect is scale-independent across different trace amplitudes.
class GaussianNoiseTransform : public ITransform {
public:
    explicit GaussianNoiseTransform(float noise_std = 0.1f)
        : noise_std_(noise_std), rng_(std::random_device{}()), dist_(0.f, 1.f) {}

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t) override;

    void  setNoiseStd(float s) { noise_std_ = s; }
    float noiseStd()     const { return noise_std_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<GaussianNoiseTransform>(*this);
    }

private:
    float noise_std_;
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<float> dist_;
};

// ---------------------------------------------------------------------------
// BiquadFilterTransform — standard 2nd-order IIR filter (RBJ "Audio EQ
// Cookbook" formulas), Direct Form II Transposed for numerical stability.
// Causal and stateful (two delay registers), reset between traces like any
// other sequential transform.
//
// cutoff is a *normalised* frequency: fraction of Nyquist (0, 1), not Hz —
// TRS files don't reliably carry a real sample rate, so this keeps the
// filter portable across any trace set. 0.1 means "10% of Nyquist"; for a
// trace sampled at 1 GS/s that's 50 MHz, but the filter itself only ever
// sees the ratio.
//
// q shapes the resonance/rolloff:
//   Lowpass/Highpass: Q ~ 0.707 (1/sqrt(2)) is the maximally-flat Butterworth
//     response; higher Q peaks near the cutoff before rolling off.
//   Bandpass/Notch: Q sets how narrow the passband/notch is — bandwidth
//     (in octaves) ~ cutoff/Q, so higher Q = narrower.
// ---------------------------------------------------------------------------
class BiquadFilterTransform : public ITransform {
public:
    enum class FilterType { Lowpass, Highpass, Bandpass, Notch };

    explicit BiquadFilterTransform(FilterType type = FilterType::Lowpass,
                                    float cutoff = 0.1f, float q = 0.707f)
        : type_(type)
        , cutoff_(std::clamp(cutoff, 1e-4f, 0.999f))
        , q_(std::max(0.05f, q))
    {
        computeCoeffs();
    }

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t) override;
    void reset() override { z1_ = z2_ = 0.0; }
    bool requiresSequential() const override { return true; }

    void setType(FilterType t)   { type_ = t; computeCoeffs(); }
    void setCutoff(float c)      { cutoff_ = std::clamp(c, 1e-4f, 0.999f); computeCoeffs(); }
    void setQ(float q)           { q_ = std::max(0.05f, q); computeCoeffs(); }
    FilterType type() const { return type_; }
    float      cutoff() const { return cutoff_; }
    float      q()      const { return q_; }

    // Zero-phase (forward-backward, "filtfilt") mode.
    //
    // The causal default has a frequency-dependent group delay: it does not
    // just attenuate the stop band, it *delays different frequencies by
    // different amounts*, and a high-Q notch's delay spikes sharply right at
    // the notch. Every analysis in this app is indexed by sample — t-test,
    // CPA, SNR, alignment — so smearing leakage along the time axis is the
    // one thing preprocessing must not do, and it is easy to mistake the
    // result for a genuine change in where a device leaks.
    //
    // Running the same filter forwards and then backwards makes the combined
    // response the squared magnitude of the one-way response with *exactly
    // zero* phase: no delay, no smearing. The costs are that the stop band is
    // twice as deep in dB (a 20 dB notch becomes 40 dB, so design for half),
    // the filter is no longer causal, and the whole trace must be available
    // in one call — see requiresWholeTrace().
    void setZeroPhase(bool on) { zero_phase_ = on; }
    bool zeroPhase() const     { return zero_phase_; }
    bool requiresWholeTrace() const override { return zero_phase_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<BiquadFilterTransform>(*this);
    }

private:
    void computeCoeffs();

    FilterType type_;
    float  cutoff_;   // fraction of Nyquist, (0, 1)
    float  q_;

    bool   zero_phase_ = false;

    // Normalised coefficients (b0,b1,b2,a1,a2; a0 folded to 1) and the two
    // Direct-Form-II-Transposed delay registers. The delay registers are only
    // used by the causal path — the zero-phase path is stateless by
    // construction, since it always sees a whole trace.
    double b0_ = 1.0, b1_ = 0.0, b2_ = 0.0, a1_ = 0.0, a2_ = 0.0;
    double z1_ = 0.0, z2_ = 0.0;
};

// ---------------------------------------------------------------------------
class STFTMagnitudeTransform : public ITransform {
public:
    enum class Window { Rectangular, Hann, Hamming, Blackman };

    STFTMagnitudeTransform(int window_size = 256, int hop_size = 128,
                           Window win = Window::Hann)
        : window_size_(std::max(2, window_size))
        , hop_size_(std::max(1, hop_size))
        , window_(win)
    {}

    std::string name() const override;

    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;

    int64_t transformedCount(int64_t input_count) const override {
        if (input_count < window_size_) return 0;
        int64_t num_windows = (input_count - window_size_) / hop_size_ + 1;
        return num_windows * (window_size_ / 2 + 1);
    }

    void reset() override {}
    bool requiresSequential() const override { return true; }

    void   setWindowSize(int w) { window_size_ = std::max(2, w); }
    void   setHopSize(int h)    { hop_size_    = std::max(1, h); }
    void   setWindow(Window w)  { window_      = w; }
    int    windowSize()   const { return window_size_; }
    int    hopSize()      const { return hop_size_;    }
    Window window()       const { return window_;      }

    // Convenience view of hop_size_ as a fraction of window_size_: overlap =
    // 1 - hop/window. setOverlap() just derives and stores hop_size_ (hop
    // stays the single source of truth, so setHopSize()/hopSize() and
    // setOverlap()/overlap() can be mixed without going out of sync).
    void  setOverlap(float o) {
        float clamped = std::clamp(o, 0.0f, 1.0f);
        setHopSize(std::max(1, static_cast<int>(std::llround(window_size_ * (1.0 - clamped)))));
    }
    float overlap() const {
        return 1.0f - static_cast<float>(hop_size_) / static_cast<float>(window_size_);
    }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<STFTMagnitudeTransform>(*this);
    }

private:
    int    window_size_;
    int    hop_size_;
    Window window_;

    // Cached across calls — see FFTMagnitudeTransform for rationale.
    std::vector<float> window_cache_;
    int                window_cache_size_ = -1;
    Window             window_cache_type_ = Window::Rectangular;
    Eigen::FFT<float>  fft_;
};

// ---------------------------------------------------------------------------
// Narrowband interference removal
// ---------------------------------------------------------------------------
//
// One band to take out of the spectrum, in *normalised* frequency: a fraction
// of Nyquist, the same unit BiquadFilterTransform's cutoff uses and for the
// same reason — TRS files can't be relied on to carry a sample rate, so the
// transforms themselves only ever see ratios and stay portable across trace
// sets. UI code converts Hz to this unit with hzToNormalized() once it knows
// a rate; see spectral.h.
// ---------------------------------------------------------------------------
struct SpectralBand {
    double center    = 0.0;   // (0, 1], fraction of Nyquist
    double halfwidth = 0.0;   // fraction of Nyquist; the band is center ± this

    double lo() const { return center - halfwidth; }
    double hi() const { return center + halfwidth; }
};

// Compact round-trip encoding for chain.json, which stores a step's fields as
// flat scalars and has no list type: "center:halfwidth,center:halfwidth,…".
std::string encodeSpectralBands(const std::vector<SpectralBand>& bands);
bool        decodeSpectralBands(const std::string& s, std::vector<SpectralBand>& out);

// Expands a fundamental into the comb f0, 2·f0, … up to Nyquist. `odd_only`
// keeps just the odd multiples, which is what a square-wave-ish clock or
// switching converter actually puts most of its energy into — notching the
// even ones then costs passband for nothing. Bands at or past Nyquist are
// dropped rather than aliased.
std::vector<SpectralBand> harmonicComb(double f0_norm, double halfwidth_norm,
                                        int n_harmonics, bool odd_only);

// ---------------------------------------------------------------------------
// SpectralExciseTransform — removes narrowband lines by editing the spectrum
// directly, rather than by filtering.
//
// FFT the whole trace, replace the magnitude inside each flagged band with a
// linear interpolation between the floor either side of it, inverse FFT. This
// is the standard "flag and interpolate" excision used on radio interference,
// and for offline trace analysis it beats a notch filter on every axis that
// matters here:
//
//   * Exactly zero phase distortion. A notch has violently steep phase right
//     where you are operating, and a *causal* notch (which is what cascading
//     BiquadFilterTransform would give you) also delays the trace by a
//     frequency-dependent amount. Since every downstream analysis in this app
//     is per-sample — t-test, CPA, SNR, alignment — smearing leakage along
//     the time axis is the one thing preprocessing must not do.
//   * Arbitrarily narrow. A single-bin notch costs one bin of bandwidth; an
//     IIR notch that narrow rings for thousands of samples.
//   * Any number of lines in one pass, at no extra cost.
//
// The price is that it needs the whole trace in one call (requiresWholeTrace)
// and O(N log N) work per trace.
//
// Interpolating the magnitude — rather than zeroing the bins — is deliberate:
// zeroing punches a hole in the noise floor, which is itself a spectral
// feature and rings in the time domain. Filling the band with the surrounding
// floor level leaves the spectrum looking like the line was never there.
// ---------------------------------------------------------------------------
class SpectralExciseTransform : public ITransform {
public:
    // What to do with the phase of the excised bins. Preserve keeps each
    // bin's original phase and only rewrites its magnitude — the least
    // surprising choice, and the one that leaves the rest of the trace
    // bit-identical in the absence of lines. Randomize decorrelates the
    // replacement from the interferer, which matters when the line is strong
    // enough that even a floor-level residue at the original phase would
    // still correlate across traces.
    enum class PhaseMode { Preserve, Randomize };

    explicit SpectralExciseTransform(std::vector<SpectralBand> bands = {},
                                      PhaseMode phase = PhaseMode::Preserve)
        : bands_(std::move(bands))
        , phase_(phase)
        , rng_(12345u)   // fixed seed: excision must be reproducible run to run
    {}

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    void    reset() override {}
    bool    requiresSequential()  const override { return true; }
    bool    requiresWholeTrace() const override { return true; }

    void setBands(std::vector<SpectralBand> b) { bands_ = std::move(b); }
    const std::vector<SpectralBand>& bands() const { return bands_; }
    void      setPhaseMode(PhaseMode m) { phase_ = m; }
    PhaseMode phaseMode() const         { return phase_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<SpectralExciseTransform>(*this);
    }

private:
    std::vector<SpectralBand> bands_;
    PhaseMode                 phase_;

    mutable std::mt19937 rng_;
    // Scratch reused across traces — apply() runs once per trace over a whole
    // trace set, so reallocating a trace-sized complex spectrum every call is
    // the dominant cost if you let it be.
    std::vector<float>               in_scratch_;
    std::vector<std::complex<float>> freq_scratch_;
    Eigen::FFT<float>                fft_;
    int64_t                          fft_size_ = -1;
};

// ---------------------------------------------------------------------------
// Biquad cascade machinery, shared by BiquadFilterTransform's zero-phase path
// and HarmonicNotchTransform.
// ---------------------------------------------------------------------------

// One second-order section, normalised so a0 == 1.
struct BiquadCoeffs {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
};

// RBJ "Audio EQ Cookbook" design; `cutoff` is a fraction of Nyquist, matching
// BiquadFilterTransform::cutoff().
BiquadCoeffs designBiquad(BiquadFilterTransform::FilterType type,
                           double cutoff, double q);

// Runs `sections` in cascade over buf[0..count), forwards then backwards, so
// the net phase response is exactly zero. Stateless: every call sees a whole
// trace, which is what makes the reverse pass meaningful at all.
//
// Edges are handled by odd reflection about each endpoint — the extension used
// by scipy's filtfilt(padtype='odd') — and each pass starts from the steady
// state its first sample implies rather than from zero. Without both, a trace
// whose value at the edges is far from zero rings for hundreds of samples
// inward, which on a power trace is exactly where the interesting part of the
// window often sits.
void applyBiquadCascadeZeroPhase(const std::vector<BiquadCoeffs>& sections,
                                  float* buf, int64_t count);

// ---------------------------------------------------------------------------
// HarmonicNotchTransform — a comb of notches at f0, 2·f0, 3·f0, …
//
// The streaming-friendly counterpart to SpectralExciseTransform: no
// trace-sized FFT, so it copes with traces too long to transform whole (in
// causal mode), and it gives a smooth analogue-style rolloff rather than a
// hard edit of individual bins. Prefer excision when you have the whole trace
// and know where the lines are; prefer this when you know f0 and want a comb.
//
// Two deliberate design points:
//
//   * Zero-phase by default. A notch has the steepest phase slope of any
//     second-order section, right where you are operating. See
//     BiquadFilterTransform::setZeroPhase for why that matters here.
//
//   * An odd-harmonics-only mode. A square-ish clock or a switching converter
//     puts most of its energy into odd multiples of f0, so notching the even
//     ones spends passband — and passband is where the leakage lives — for
//     very little return.
//
// And one warning that belongs in the code as much as the UI: leakage is *not*
// confined to clock harmonics (Tian et al., JCEN 2015), and screaming-channel
// work finds exploitable leakage at non-harmonic frequencies routinely beating
// the harmonics (Wouters et al., CARDIS 2023). A comb is a blunt instrument;
// measure SNR or NICV before and after rather than assuming it helped.
// ---------------------------------------------------------------------------
class HarmonicNotchTransform : public ITransform {
public:
    HarmonicNotchTransform(float f0 = 0.1f, int n_harmonics = 5, float q = 20.0f,
                            bool odd_only = false, bool zero_phase = true)
        : f0_(std::clamp(f0, 1e-4f, 0.999f))
        , n_harmonics_(std::max(1, n_harmonics))
        , q_(std::max(0.5f, q))
        , odd_only_(odd_only)
        , zero_phase_(zero_phase)
    {
        rebuild();
    }

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    void    reset() override;
    bool    requiresSequential()  const override { return true; }
    bool    requiresWholeTrace() const override { return zero_phase_; }

    void  setF0(float f)           { f0_ = std::clamp(f, 1e-4f, 0.999f); rebuild(); }
    void  setHarmonics(int n)      { n_harmonics_ = std::max(1, n);      rebuild(); }
    void  setQ(float q)            { q_ = std::max(0.5f, q);             rebuild(); }
    void  setOddOnly(bool on)      { odd_only_ = on;                     rebuild(); }
    void  setZeroPhase(bool on)    { zero_phase_ = on;                   rebuild(); }
    float f0()         const { return f0_; }
    int   harmonics()  const { return n_harmonics_; }
    float q()          const { return q_; }
    bool  oddOnly()    const { return odd_only_; }
    bool  zeroPhase()  const { return zero_phase_; }

    // Which harmonics actually got a notch: multiples at or past Nyquist are
    // dropped, so this can be shorter than harmonics().
    int   activeNotches() const { return static_cast<int>(sections_.size()); }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<HarmonicNotchTransform>(*this);
    }

private:
    void rebuild();

    float f0_;
    int   n_harmonics_;
    float q_;
    bool  odd_only_;
    bool  zero_phase_;

    std::vector<BiquadCoeffs> sections_;
    // Causal-mode delay registers, two per section, carried across chunks.
    std::vector<double>       z1_, z2_;
};

// ---------------------------------------------------------------------------
// PeriodicSubtractTransform — synchronous averaging of a periodic interferer,
// then subtraction.
//
// The right tool when the interference is periodic and steady but *not*
// sinusoidal: a DC-DC converter's switching waveform, a reference clock, an
// RFID reader's field. Such a source puts energy into a fundamental and a long
// tail of harmonics, so removing it with notches costs one notch — and one
// hole in the passband — per harmonic.
//
// Instead: fold the trace at the interferer's period, average to get its
// waveform, subtract that. The fundamental, every harmonic, and whatever was
// never sinusoidal all go at once, and the spectrum keeps every bin.
//
// Why this does not also remove the signal: the cryptographic operation is
// triggered, so it sits at the same place in every trace and repeats at the
// *trace* rate. The interferer is free-running and repeats at its own rate.
// Folding at the interferer's period therefore averages the signal down toward
// its mean while the interferer adds coherently. The assumption fails if the
// two rates are commensurate — if the device clocks the converter, say — and
// then this will eat signal, which is why the period is reported back and the
// result should be checked with SNR or a t-test either way.
//
// The period is deliberately a double. A switcher's period almost never lands
// on a whole number of samples, and rounding it smears the folded template
// into uselessness within a few dozen periods; the template is accumulated and
// subtracted with linear interpolation in phase instead.
// ---------------------------------------------------------------------------
class PeriodicSubtractTransform : public ITransform {
public:
    // period <= 0 means "estimate from each trace" over
    // [auto_min_period, auto_max_period]; see estimatePeriod().
    explicit PeriodicSubtractTransform(double period = 0.0,
                                        int64_t auto_min_period = 8,
                                        int64_t auto_max_period = 100000,
                                        double auto_min_correlation = 0.3)
        : period_(period)
        , auto_min_(std::max<int64_t>(2, auto_min_period))
        , auto_max_(std::max<int64_t>(3, auto_max_period))
        , auto_min_corr_(auto_min_correlation)
    {}

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    void    reset() override {}
    bool    requiresSequential()  const override { return true; }
    bool    requiresWholeTrace() const override { return true; }

    void   setPeriod(double p)          { period_ = p; }
    double period() const               { return period_; }
    void   setAutoRange(int64_t lo, int64_t hi) {
        auto_min_ = std::max<int64_t>(2, lo);
        auto_max_ = std::max<int64_t>(auto_min_ + 1, hi);
    }
    int64_t autoMin() const { return auto_min_; }
    int64_t autoMax() const { return auto_max_; }
    void    setAutoMinCorrelation(double c) { auto_min_corr_ = c; }
    double  autoMinCorrelation() const      { return auto_min_corr_; }

    // The period actually used on the last apply() — equal to period() when
    // that was set, or whatever the estimator found. 0 means the last trace
    // was left untouched because no convincing period was found.
    double lastPeriodUsed() const { return last_period_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<PeriodicSubtractTransform>(*this);
    }

private:
    double  period_;
    int64_t auto_min_;
    int64_t auto_max_;
    double  auto_min_corr_;
    double  last_period_ = 0.0;

    std::vector<double> accum_;   // folded sum per template bin
    std::vector<double> weight_;  // interpolation weight per template bin
};

// ---------------------------------------------------------------------------
// ComplexDemodulateTransform — shift a band down to baseband and take its
// envelope.
//
// This is the transform for the case where the carrier is not interference but
// the *transport*. On a mixed-signal chip, digital switching noise couples into
// the RF front end and is amplified, modulated onto the carrier and broadcast —
// the "screaming channel" effect, exploitable at metres rather than
// centimetres. Notching the carrier there would delete the signal; what you
// want is to demodulate it. The same machinery reads clock-harmonic sidebands,
// where activity appears at f_clk ± n/T rather than at f_clk itself.
//
// Implementation is a band-select and frequency shift done directly on the
// spectrum: take the bins within `bandwidth` of `center`, move them to DC,
// inverse-transform, and report the magnitude of the resulting complex
// baseband signal. That is exact — no filter transition band, no phase
// distortion — at the cost of needing the whole trace.
//
// Setting `center` to a small value with a wide `bandwidth` degenerates to the
// Hilbert envelope of the low-frequency content, which is the usual way to
// demodulate an amplitude-modulated EM capture.
// ---------------------------------------------------------------------------
class ComplexDemodulateTransform : public ITransform {
public:
    // Envelope is the magnitude of the complex baseband signal — the usual
    // choice, and what an AM demodulator produces. Baseband is its real part,
    // i.e. the selected band shifted down but still oscillating, which is
    // occasionally what you want to feed a correlation.
    enum class Output { Envelope, Baseband };

    explicit ComplexDemodulateTransform(float center = 0.5f, float bandwidth = 0.05f,
                                         Output out = Output::Envelope)
        : center_(std::clamp(center, 0.0f, 1.0f))
        , bandwidth_(std::clamp(bandwidth, 1e-5f, 1.0f))
        , output_(out)
    {}

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    void    reset() override {}
    bool    requiresSequential()  const override { return true; }
    bool    requiresWholeTrace() const override { return true; }

    void   setCenter(float c)    { center_ = std::clamp(c, 0.0f, 1.0f); }
    void   setBandwidth(float b) { bandwidth_ = std::clamp(b, 1e-5f, 1.0f); }
    void   setOutput(Output o)   { output_ = o; }
    float  center()    const { return center_; }
    float  bandwidth() const { return bandwidth_; }
    Output output()    const { return output_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<ComplexDemodulateTransform>(*this);
    }

private:
    float  center_;      // fraction of Nyquist
    float  bandwidth_;   // half-width, fraction of Nyquist
    Output output_;

    std::vector<std::complex<float>> in_scratch_, freq_scratch_, band_scratch_;
    Eigen::FFT<float>                fft_;
    int64_t                          fft_size_ = -1;
};
