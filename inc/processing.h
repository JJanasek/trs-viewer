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

    // Number of leading output samples that are part of a startup transient
    // and should be skipped when resetting the view to "valid" data.
    virtual int64_t startupSamples() const { return 0; }

    virtual std::shared_ptr<ITransform> clone() const = 0;
};

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

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<BiquadFilterTransform>(*this);
    }

private:
    void computeCoeffs();

    FilterType type_;
    float  cutoff_;   // fraction of Nyquist, (0, 1)
    float  q_;

    // Normalised coefficients (b0,b1,b2,a1,a2; a0 folded to 1) and the two
    // Direct-Form-II-Transposed delay registers.
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
