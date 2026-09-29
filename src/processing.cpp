#include "processing.h"
#include "spectral.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <stdexcept>

// ---------------------------------------------------------------------------
// AbsTransform
// ---------------------------------------------------------------------------
int64_t AbsTransform::apply(float* buf, int64_t count, int64_t) {
    for (int64_t i = 0; i < count; i++)
        buf[i] = std::abs(buf[i]);
    return count;
}

// ---------------------------------------------------------------------------
// NegateTransform
// ---------------------------------------------------------------------------
int64_t NegateTransform::apply(float* buf, int64_t count, int64_t) {
    for (int64_t i = 0; i < count; i++)
        buf[i] = -buf[i];
    return count;
}

// ---------------------------------------------------------------------------
// OffsetTransform
// ---------------------------------------------------------------------------
std::string OffsetTransform::name() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Offset (%+.4g)", offset_);
    return buf;
}

int64_t OffsetTransform::apply(float* buf, int64_t count, int64_t) {
    for (int64_t i = 0; i < count; i++)
        buf[i] += offset_;
    return count;
}

// ---------------------------------------------------------------------------
// ScaleTransform
// ---------------------------------------------------------------------------
std::string ScaleTransform::name() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Scale (×%.4g)", scale_);
    return buf;
}

int64_t ScaleTransform::apply(float* buf, int64_t count, int64_t) {
    for (int64_t i = 0; i < count; i++)
        buf[i] *= scale_;
    return count;
}

// ---------------------------------------------------------------------------
// MovingAverageTransform  (causal, same output length as input)
// ---------------------------------------------------------------------------
MovingAverageTransform::MovingAverageTransform(int window_size)
    : window_size_(std::max(1, window_size))
    , ring_sum_(0.0)
    , ring_pos_(0)
    , ring_count_(0)
{
    ring_.resize(window_size_, 0.0f);
}

std::string MovingAverageTransform::name() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Moving Average (w=%d)", window_size_);
    return buf;
}

void MovingAverageTransform::reset() {
    std::fill(ring_.begin(), ring_.end(), 0.0f);
    ring_sum_   = 0.0;
    ring_pos_   = 0;
    ring_count_ = 0;
}

void MovingAverageTransform::setWindowSize(int w) {
    window_size_ = std::max(1, w);
    ring_.assign(window_size_, 0.0f);
    reset();
}

int64_t MovingAverageTransform::apply(float* buf, int64_t count, int64_t) {
    for (int64_t i = 0; i < count; i++) {
        ring_sum_ -= static_cast<double>(ring_[ring_pos_]);
        ring_[ring_pos_] = buf[i];
        ring_sum_ += static_cast<double>(buf[i]);
        if (++ring_pos_ >= window_size_) ring_pos_ = 0;
        ring_count_++;

        int64_t n = std::min(ring_count_, static_cast<int64_t>(window_size_));
        buf[i] = static_cast<float>(ring_sum_ / static_cast<double>(n));
    }
    return count;
}

// ---------------------------------------------------------------------------
// GaussianNoiseTransform
// ---------------------------------------------------------------------------
std::string GaussianNoiseTransform::name() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Gaussian Noise (%.4g × trace std)", noise_std_);
    return buf;
}

int64_t GaussianNoiseTransform::apply(float* buf, int64_t count, int64_t) {
    if (noise_std_ <= 0.f || count == 0) return count;

    // Compute per-trace mean and std so noise is scale-independent.
    double mean = 0.0;
    for (int64_t i = 0; i < count; i++) mean += buf[i];
    mean /= static_cast<double>(count);

    double var = 0.0;
    for (int64_t i = 0; i < count; i++) {
        double d = static_cast<double>(buf[i]) - mean;
        var += d * d;
    }
    float trace_std = static_cast<float>(std::sqrt(var / static_cast<double>(count)));
    trace_std = std::max(trace_std, 1e-8f);

    float effective_sigma = noise_std_ * trace_std;
    for (int64_t i = 0; i < count; i++)
        buf[i] += dist_(rng_) * effective_sigma;
    return count;
}

// ---------------------------------------------------------------------------
// BiquadFilterTransform
// ---------------------------------------------------------------------------
std::string BiquadFilterTransform::name() const {
    const char* t = type_ == FilterType::Lowpass  ? "Lowpass"  :
                    type_ == FilterType::Highpass ? "Highpass" :
                    type_ == FilterType::Bandpass ? "Bandpass" : "Notch";
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s Filter (fc=%.3f×Nyq, Q=%.2f)",
                  t, static_cast<double>(cutoff_), static_cast<double>(q_));
    return buf;
}

// RBJ "Audio EQ Cookbook" biquad design. w0 = pi * cutoff since cutoff is
// already a fraction of Nyquist (i.e. w0 = 2*pi*(f0/fs) with f0/fs =
// cutoff/2). Coefficients normalised by a0 up front so apply() never divides.
void BiquadFilterTransform::computeCoeffs() {
    const double w0    = M_PI * static_cast<double>(cutoff_);
    const double cw     = std::cos(w0);
    const double sw     = std::sin(w0);
    const double alpha = sw / (2.0 * static_cast<double>(q_));

    double b0, b1, b2, a0, a1, a2;
    switch (type_) {
    case FilterType::Lowpass:
        b0 = (1.0 - cw) / 2.0; b1 = 1.0 - cw; b2 = (1.0 - cw) / 2.0;
        a0 = 1.0 + alpha;      a1 = -2.0 * cw; a2 = 1.0 - alpha;
        break;
    case FilterType::Highpass:
        b0 = (1.0 + cw) / 2.0; b1 = -(1.0 + cw); b2 = (1.0 + cw) / 2.0;
        a0 = 1.0 + alpha;      a1 = -2.0 * cw;   a2 = 1.0 - alpha;
        break;
    case FilterType::Bandpass:
        b0 = sw / 2.0; b1 = 0.0; b2 = -sw / 2.0;
        a0 = 1.0 + alpha; a1 = -2.0 * cw; a2 = 1.0 - alpha;
        break;
    case FilterType::Notch:
    default:
        b0 = 1.0; b1 = -2.0 * cw; b2 = 1.0;
        a0 = 1.0 + alpha; a1 = -2.0 * cw; a2 = 1.0 - alpha;
        break;
    }
    b0_ = b0 / a0; b1_ = b1 / a0; b2_ = b2 / a0;
    a1_ = a1 / a0; a2_ = a2 / a0;
}

int64_t BiquadFilterTransform::apply(float* buf, int64_t count, int64_t) {
    if (zero_phase_) {
        // Stateless by construction — a zero-phase pass always sees the whole
        // trace (requiresWholeTrace()), so there is nothing to carry across
        // calls and z1_/z2_ stay untouched.
        const std::vector<BiquadCoeffs> one{ BiquadCoeffs{b0_, b1_, b2_, a1_, a2_} };
        applyBiquadCascadeZeroPhase(one, buf, count);
        return count;
    }
    for (int64_t i = 0; i < count; i++) {
        double x = static_cast<double>(buf[i]);
        double y = b0_ * x + z1_;
        z1_ = b1_ * x + z2_ - a1_ * y;
        z2_ = b2_ * x - a2_ * y;
        buf[i] = static_cast<float>(y);
    }
    return count;
}

// ---------------------------------------------------------------------------
// WindowResampleTransform  (sliding-window mean, hop = window*(1-overlap))
// ---------------------------------------------------------------------------
WindowResampleTransform::WindowResampleTransform(int window_size, float overlap)
    : window_size_(std::max(1, window_size))
    , overlap_(std::clamp(overlap, 0.0f, 1.0f))
{
    reset();
}

std::string WindowResampleTransform::name() const {
    char buf[80];
    std::snprintf(buf, sizeof(buf), "Window Resample (w=%d, overlap=%.2f)",
                 window_size_, static_cast<double>(overlap_));
    return buf;
}

int WindowResampleTransform::hopSize() const {
    int hop = static_cast<int>(std::llround(window_size_ * (1.0 - overlap_)));
    return std::max(1, hop);
}

int64_t WindowResampleTransform::transformedCount(int64_t input_count) const {
    if (input_count < window_size_) return 0;
    return (input_count - window_size_) / hopSize() + 1;
}

void WindowResampleTransform::reset() {
    ring_.assign(static_cast<size_t>(window_size_), 0.0f);
    ring_sum_     = 0.0;
    ring_pos_     = 0;
    ring_count_   = 0;
    samples_seen_ = 0;
    next_emit_at_ = window_size_ - 1;
}

void WindowResampleTransform::setWindowSize(int w) {
    window_size_ = std::max(1, w);
    reset();
}

void WindowResampleTransform::setOverlap(float o) {
    overlap_ = std::clamp(o, 0.0f, 1.0f);
    reset();
}

int64_t WindowResampleTransform::apply(float* buf, int64_t count, int64_t) {
    const int64_t hop = hopSize();
    int64_t out = 0;

    for (int64_t i = 0; i < count; i++) {
        float v = buf[i];
        if (ring_count_ < window_size_) {
            ring_sum_ += v;
            ring_[static_cast<size_t>(ring_pos_)] = v;
            ring_count_++;
        } else {
            ring_sum_ += static_cast<double>(v) - ring_[static_cast<size_t>(ring_pos_)];
            ring_[static_cast<size_t>(ring_pos_)] = v;
        }
        ring_pos_ = (ring_pos_ + 1) % window_size_;

        if (samples_seen_ == next_emit_at_) {
            buf[out++] = static_cast<float>(ring_sum_ / window_size_);
            next_emit_at_ += hop;
        }
        samples_seen_++;
    }

    return out;
}

// ---------------------------------------------------------------------------
// StrideResampleTransform  (keep every stride-th sample)
// ---------------------------------------------------------------------------
StrideResampleTransform::StrideResampleTransform(int stride)
    : stride_(std::max(1, stride))
{}

std::string StrideResampleTransform::name() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Stride Resample (s=%d)", stride_);
    return buf;
}

int64_t StrideResampleTransform::transformedCount(int64_t input_count) const {
    return (input_count + stride_ - 1) / stride_;
}

void StrideResampleTransform::reset() {
    pos_ = 0;
}

void StrideResampleTransform::setStride(int s) {
    stride_ = std::max(1, s);
    reset();
}

int64_t StrideResampleTransform::apply(float* buf, int64_t count, int64_t) {
    int64_t out = 0;
    for (int64_t i = 0; i < count; i++) {
        if (pos_ == 0) buf[out++] = buf[i];
        if (++pos_ >= stride_) pos_ = 0;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Shared helper: fill a vector with window coefficients.
// ---------------------------------------------------------------------------
namespace {
enum class WinType { Rectangular, Hann, Hamming, Blackman };

static void buildWindowCoeffs(std::vector<float>& w, int N, WinType type) {
    w.resize(static_cast<size_t>(N));
    if (type == WinType::Rectangular) {
        std::fill(w.begin(), w.end(), 1.0f);
        return;
    }
    const double N1 = static_cast<double>(N - 1);
    for (int i = 0; i < N; ++i) {
        double phi = 2.0 * M_PI * i / N1;
        double v   = 1.0;
        switch (type) {
            case WinType::Hann:     v = 0.5  * (1.0 - std::cos(phi)); break;
            case WinType::Hamming:  v = 0.54 - 0.46 * std::cos(phi);  break;
            case WinType::Blackman: v = 0.42 - 0.5  * std::cos(phi)
                                          + 0.08 * std::cos(2.0 * phi); break;
            default: break;
        }
        w[i] = static_cast<float>(v);
    }
}
} // namespace

// ---------------------------------------------------------------------------
// FFTMagnitudeTransform
// ---------------------------------------------------------------------------

std::string FFTMagnitudeTransform::name() const {
    const char* wnames[] = { "Rectangular", "Hann", "Hamming", "Blackman" };
    return std::string("FFT Magnitude (") + wnames[static_cast<int>(window_)] + ")";
}

int64_t FFTMagnitudeTransform::apply(float* buf, int64_t count, int64_t) {
    if (count < 2) return count;

    const int64_t N     = count;
    const int64_t out_n = N / 2 + 1;

    if (window_cache_size_ != N || window_cache_type_ != window_) {
        buildWindowCoeffs(window_cache_, static_cast<int>(N),
                          static_cast<WinType>(static_cast<int>(window_)));
        window_cache_size_ = N;
        window_cache_type_ = window_;
        // Eigen's (unsupported) FFT module corrupts its internal kissfft
        // plan cache when the same instance is reused across a size change
        // — rebuild it fresh whenever the size changes, matching the plan
        // it would have built anyway had a new instance been used per call.
        fft_ = Eigen::FFT<float>();
    }
    const std::vector<float>& win = window_cache_;

    std::vector<float> in_vec(N);
    for (int64_t i = 0; i < N; ++i) in_vec[i] = buf[i] * win[i];

    std::vector<std::complex<float>> freq_vec;
    fft_.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    fft_.fwd(freq_vec, in_vec);

    const float norm = 1.0f / static_cast<float>(N);
    for (int64_t k = 0; k < out_n; ++k) {
        float mag = std::abs(freq_vec[k]) * norm;
        if (k > 0 && k < out_n - 1) mag *= 2.0f;
        buf[k] = mag;
    }
    return out_n;
}

// ---------------------------------------------------------------------------
// STFTMagnitudeTransform
// ---------------------------------------------------------------------------

std::string STFTMagnitudeTransform::name() const {
    const char* wnames[] = { "Rectangular", "Hann", "Hamming", "Blackman" };
    return std::string("STFT Magnitude (W=") + std::to_string(window_size_)
         + ", H=" + std::to_string(hop_size_)
         + ", " + wnames[static_cast<int>(window_)] + ")";
}

int64_t STFTMagnitudeTransform::apply(float* buf, int64_t count, int64_t) {
    if (count < window_size_) return 0;

    const int64_t W           = window_size_;
    const int64_t bins        = W / 2 + 1;
    const int64_t num_windows = (count - W) / hop_size_ + 1;
    const int64_t out_n       = num_windows * bins;

    if (window_cache_size_ != W || window_cache_type_ != window_) {
        buildWindowCoeffs(window_cache_, static_cast<int>(W),
                          static_cast<WinType>(static_cast<int>(window_)));
        window_cache_size_ = W;
        window_cache_type_ = window_;
        // Eigen's (unsupported) FFT module corrupts its internal kissfft
        // plan cache when the same instance is reused across a size change
        // — rebuild it fresh whenever the size changes, matching the plan
        // it would have built anyway had a new instance been used per call.
        fft_ = Eigen::FFT<float>();
    }
    const std::vector<float>& win = window_cache_;

    // Write results to a separate buffer to avoid aliasing the input.
    std::vector<float> out(static_cast<size_t>(out_n));

    fft_.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    std::vector<float>               in_vec(static_cast<size_t>(W));
    std::vector<std::complex<float>> freq_vec;

    const float norm = 1.0f / static_cast<float>(W);

    for (int64_t wi = 0; wi < num_windows; ++wi) {
        const int64_t pos = wi * hop_size_;
        for (int64_t i = 0; i < W; ++i)
            in_vec[i] = buf[pos + i] * win[i];

        fft_.fwd(freq_vec, in_vec);

        float* dst = out.data() + wi * bins;
        for (int64_t k = 0; k < bins; ++k) {
            float mag = std::abs(freq_vec[k]) * norm;
            if (k > 0 && k < bins - 1) mag *= 2.0f;
            dst[k] = mag;
        }
    }

    std::copy(out.begin(), out.end(), buf);
    return out_n;
}

// ---------------------------------------------------------------------------
// SpectralBand helpers
// ---------------------------------------------------------------------------
std::string encodeSpectralBands(const std::vector<SpectralBand>& bands) {
    std::string s;
    char tmp[80];
    for (size_t i = 0; i < bands.size(); ++i) {
        std::snprintf(tmp, sizeof(tmp), "%.9g:%.9g", bands[i].center, bands[i].halfwidth);
        if (i) s += ',';
        s += tmp;
    }
    return s;
}

bool decodeSpectralBands(const std::string& s, std::vector<SpectralBand>& out) {
    out.clear();
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        const std::string tok = s.substr(pos, comma - pos);
        pos = comma + 1;
        if (tok.empty()) continue;

        const size_t colon = tok.find(':');
        if (colon == std::string::npos) { out.clear(); return false; }
        try {
            SpectralBand b;
            b.center    = std::stod(tok.substr(0, colon));
            b.halfwidth = std::stod(tok.substr(colon + 1));
            out.push_back(b);
        } catch (...) {
            out.clear();
            return false;
        }
    }
    return true;
}

std::vector<SpectralBand> harmonicComb(double f0_norm, double halfwidth_norm,
                                        int n_harmonics, bool odd_only)
{
    std::vector<SpectralBand> out;
    if (f0_norm <= 0.0 || n_harmonics <= 0) return out;
    for (int k = 1, emitted = 0; emitted < n_harmonics; ++k) {
        if (odd_only && (k % 2) == 0) continue;
        const double c = f0_norm * static_cast<double>(k);
        // Past Nyquist there is nothing to notch — a real comb keeps going,
        // but those harmonics have folded back somewhere the acquisition
        // can no longer tell us about, so inventing bands for them would be
        // guessing.
        if (c >= 1.0) break;
        out.push_back(SpectralBand{c, halfwidth_norm});
        ++emitted;
    }
    return out;
}

// ---------------------------------------------------------------------------
// SpectralExciseTransform
// ---------------------------------------------------------------------------
std::string SpectralExciseTransform::name() const {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "Spectral Excise (%zu band%s, %s phase)",
                  bands_.size(), bands_.size() == 1 ? "" : "s",
                  phase_ == PhaseMode::Preserve ? "keep" : "random");
    return buf;
}

int64_t SpectralExciseTransform::apply(float* buf, int64_t count, int64_t) {
    if (count < 8 || bands_.empty()) return count;

    const int64_t N    = count;
    const int64_t bins = N / 2 + 1;     // one-sided; bin N/2 is Nyquist

    // Eigen's FFT corrupts its kissfft plan cache when one instance is reused
    // across a size change — same trap FFTMagnitudeTransform documents.
    if (fft_size_ != N) {
        fft_       = Eigen::FFT<float>();
        fft_size_  = N;
    }
    fft_.SetFlag(Eigen::FFT<float>::HalfSpectrum);

    in_scratch_.assign(buf, buf + N);
    fft_.fwd(freq_scratch_, in_scratch_);
    if (static_cast<int64_t>(freq_scratch_.size()) < bins) return count;

    // Translate each normalised band into a bin range. Normalised frequency is
    // a fraction of Nyquist, and bin `bins-1` *is* Nyquist, so the mapping is
    // simply f * (bins - 1).
    const double bin_scale = static_cast<double>(bins - 1);
    std::uniform_real_distribution<float> phase_dist(-static_cast<float>(M_PI),
                                                      static_cast<float>(M_PI));

    for (const auto& band : bands_) {
        if (band.halfwidth < 0.0) continue;

        int64_t lo = static_cast<int64_t>(std::floor(band.lo() * bin_scale));
        int64_t hi = static_cast<int64_t>(std::ceil (band.hi() * bin_scale));
        // Never touch DC: bin 0 carries the trace's offset, not interference,
        // and rewriting it shifts the whole trace vertically.
        lo = std::max<int64_t>(1, lo);
        hi = std::min<int64_t>(bins - 1, hi);
        if (lo > hi) continue;

        // Floor level either side of the band, averaged over a guard as wide
        // as the band itself (at least 2 bins). Averaging rather than taking
        // the single adjacent bin keeps one noisy neighbour from setting the
        // whole replacement level.
        const int64_t width = hi - lo + 1;
        const int64_t guard = std::max<int64_t>(2, width);

        auto meanMag = [&](int64_t from, int64_t to) -> double {
            from = std::max<int64_t>(1, from);
            to   = std::min<int64_t>(bins - 1, to);
            if (from > to) return -1.0;             // nothing available
            double acc = 0.0;
            for (int64_t k = from; k <= to; ++k)
                acc += static_cast<double>(std::abs(freq_scratch_[static_cast<size_t>(k)]));
            return acc / static_cast<double>(to - from + 1);
        };

        double left  = meanMag(lo - guard, lo - 1);
        double right = meanMag(hi + 1, hi + guard);
        // A band hard against DC or Nyquist has floor on one side only; fall
        // back to that side rather than refusing to excise.
        if (left  < 0.0) left  = right;
        if (right < 0.0) right = left;
        if (left  < 0.0) continue;                  // no floor anywhere: skip

        for (int64_t k = lo; k <= hi; ++k) {
            const double t = (width > 1)
                ? (static_cast<double>(k - lo) / static_cast<double>(width - 1))
                : 0.5;
            const float target = static_cast<float>(left + (right - left) * t);

            std::complex<float>& c = freq_scratch_[static_cast<size_t>(k)];
            if (phase_ == PhaseMode::Preserve) {
                const float mag = std::abs(c);
                c = (mag > 1e-20f) ? c * (target / mag)
                                   : std::complex<float>(target, 0.0f);
            } else {
                const float th = phase_dist(rng_);
                c = std::polar(target, th);
            }
        }
    }

    // The Nyquist bin of an even-length real signal has no conjugate partner,
    // so it must stay real or the inverse transform is not a real signal.
    if ((N % 2) == 0)
        freq_scratch_[static_cast<size_t>(bins - 1)] =
            std::complex<float>(std::abs(freq_scratch_[static_cast<size_t>(bins - 1)]), 0.0f);

    // N must be passed explicitly. Left to itself, Eigen's half-spectrum
    // inverse infers the output length as 2*(bins-1), which is N only when N
    // is even — an odd-length trace would come back one sample short, and
    // silently losing the last sample shifts everything a later analysis
    // indexes by sample number.
    in_scratch_.assign(static_cast<size_t>(N), 0.0f);
    fft_.inv(in_scratch_, freq_scratch_, static_cast<Eigen::Index>(N));
    const int64_t n_out = std::min<int64_t>(N, static_cast<int64_t>(in_scratch_.size()));
    std::copy(in_scratch_.begin(), in_scratch_.begin() + static_cast<ptrdiff_t>(n_out), buf);
    return n_out;
}

// ---------------------------------------------------------------------------
// Biquad cascade machinery
// ---------------------------------------------------------------------------
BiquadCoeffs designBiquad(BiquadFilterTransform::FilterType type,
                           double cutoff, double q)
{
    using FT = BiquadFilterTransform::FilterType;
    cutoff = std::clamp(cutoff, 1e-4, 0.999);
    q      = std::max(0.05, q);

    const double w0    = M_PI * cutoff;
    const double cw    = std::cos(w0);
    const double sw    = std::sin(w0);
    const double alpha = sw / (2.0 * q);

    double b0, b1, b2, a0, a1, a2;
    switch (type) {
    case FT::Lowpass:
        b0 = (1.0 - cw) / 2.0; b1 = 1.0 - cw;     b2 = (1.0 - cw) / 2.0;
        a0 = 1.0 + alpha;      a1 = -2.0 * cw;    a2 = 1.0 - alpha;
        break;
    case FT::Highpass:
        b0 = (1.0 + cw) / 2.0; b1 = -(1.0 + cw);  b2 = (1.0 + cw) / 2.0;
        a0 = 1.0 + alpha;      a1 = -2.0 * cw;    a2 = 1.0 - alpha;
        break;
    case FT::Bandpass:
        b0 = sw / 2.0;         b1 = 0.0;          b2 = -sw / 2.0;
        a0 = 1.0 + alpha;      a1 = -2.0 * cw;    a2 = 1.0 - alpha;
        break;
    case FT::Notch:
    default:
        b0 = 1.0;              b1 = -2.0 * cw;    b2 = 1.0;
        a0 = 1.0 + alpha;      a1 = -2.0 * cw;    a2 = 1.0 - alpha;
        break;
    }
    BiquadCoeffs c;
    c.b0 = b0 / a0; c.b1 = b1 / a0; c.b2 = b2 / a0;
    c.a1 = a1 / a0; c.a2 = a2 / a0;
    return c;
}

namespace {

// Direct-Form-II-Transposed delay registers that leave the section in the
// steady state a constant input of `x` would produce. Starting a pass from
// zero state instead makes the filter treat a trace whose first sample is far
// from zero as a step input, and a high-Q section rings on that step for
// hundreds of samples — right across the start of the window.
inline void biquadSteadyState(const BiquadCoeffs& c, double x, double& z1, double& z2) {
    const double denom = 1.0 + c.a1 + c.a2;
    const double y     = (std::abs(denom) > 1e-12) ? x * (c.b0 + c.b1 + c.b2) / denom : 0.0;
    z2 = c.b2 * x - c.a2 * y;
    z1 = c.b1 * x + z2 - c.a1 * y;
}

// One causal pass of a cascade over data[0..n), each section initialised to its
// own steady state from the first sample.
void cascadePassSteadyStart(const std::vector<BiquadCoeffs>& sections,
                            double* data, int64_t n)
{
    if (n <= 0) return;
    for (const auto& c : sections) {
        double z1, z2;
        biquadSteadyState(c, data[0], z1, z2);
        for (int64_t i = 0; i < n; ++i) {
            const double x = data[i];
            const double y = c.b0 * x + z1;
            z1 = c.b1 * x + z2 - c.a1 * y;
            z2 = c.b2 * x - c.a2 * y;
            data[i] = y;
        }
    }
}

} // namespace

void applyBiquadCascadeZeroPhase(const std::vector<BiquadCoeffs>& sections,
                                  float* buf, int64_t count)
{
    if (sections.empty() || count <= 0) return;

    // Odd reflection about each endpoint: ext[-k] = 2*x[0] - x[k]. This keeps
    // the signal's value *and* slope continuous across the boundary, which a
    // plain mirror (2*x[0] - x[k] vs x[k]) does not, so the filter never sees
    // an artificial kink at the edge. Padding proportional to the cascade
    // length is what scipy's filtfilt does; the floor of 32 gives high-Q
    // sections room to settle.
    const int64_t pad = std::min<int64_t>(count - 1,
                            std::max<int64_t>(32, 6 * static_cast<int64_t>(sections.size())));
    const int64_t n   = count + 2 * pad;

    std::vector<double> work(static_cast<size_t>(n));
    const double first = static_cast<double>(buf[0]);
    const double last  = static_cast<double>(buf[count - 1]);
    for (int64_t k = 0; k < pad; ++k)
        work[static_cast<size_t>(k)] = 2.0 * first - static_cast<double>(buf[pad - k]);
    for (int64_t i = 0; i < count; ++i)
        work[static_cast<size_t>(pad + i)] = static_cast<double>(buf[i]);
    for (int64_t k = 0; k < pad; ++k)
        work[static_cast<size_t>(pad + count + k)] =
            2.0 * last - static_cast<double>(buf[count - 2 - k]);

    // Forward pass, then reverse the buffer, filter again, and reverse back.
    // The second pass undoes the first pass's phase exactly, leaving a purely
    // real (zero-phase) net response whose magnitude is the one-way magnitude
    // squared.
    cascadePassSteadyStart(sections, work.data(), n);
    std::reverse(work.begin(), work.end());
    cascadePassSteadyStart(sections, work.data(), n);
    std::reverse(work.begin(), work.end());

    for (int64_t i = 0; i < count; ++i)
        buf[i] = static_cast<float>(work[static_cast<size_t>(pad + i)]);
}

// ---------------------------------------------------------------------------
// HarmonicNotchTransform
// ---------------------------------------------------------------------------
void HarmonicNotchTransform::rebuild() {
    sections_.clear();
    for (int k = 1, emitted = 0; emitted < n_harmonics_; ++k) {
        if (odd_only_ && (k % 2) == 0) continue;
        const double c = static_cast<double>(f0_) * static_cast<double>(k);
        if (c >= 1.0) break;              // past Nyquist: nothing to notch
        sections_.push_back(designBiquad(BiquadFilterTransform::FilterType::Notch, c,
                                          static_cast<double>(q_)));
        ++emitted;
    }
    z1_.assign(sections_.size(), 0.0);
    z2_.assign(sections_.size(), 0.0);
}

void HarmonicNotchTransform::reset() {
    std::fill(z1_.begin(), z1_.end(), 0.0);
    std::fill(z2_.begin(), z2_.end(), 0.0);
}

std::string HarmonicNotchTransform::name() const {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "Harmonic Notch (f0=%.4f×Nyq, %d%s, Q=%.1f, %s)",
                  static_cast<double>(f0_), activeNotches(),
                  odd_only_ ? " odd" : "", static_cast<double>(q_),
                  zero_phase_ ? "zero-phase" : "causal");
    return buf;
}

int64_t HarmonicNotchTransform::apply(float* buf, int64_t count, int64_t) {
    if (count <= 0 || sections_.empty()) return count;

    if (zero_phase_) {
        applyBiquadCascadeZeroPhase(sections_, buf, count);
        return count;
    }

    // Causal path: carry each section's delay registers across calls so a
    // chunked caller gets the same result as a single-call one.
    for (size_t si = 0; si < sections_.size(); ++si) {
        const BiquadCoeffs& c = sections_[si];
        double z1 = z1_[si], z2 = z2_[si];
        for (int64_t i = 0; i < count; ++i) {
            const double x = static_cast<double>(buf[i]);
            const double y = c.b0 * x + z1;
            z1 = c.b1 * x + z2 - c.a1 * y;
            z2 = c.b2 * x - c.a2 * y;
            buf[i] = static_cast<float>(y);
        }
        z1_[si] = z1;
        z2_[si] = z2;
    }
    return count;
}

// ---------------------------------------------------------------------------
// PeriodicSubtractTransform
// ---------------------------------------------------------------------------
std::string PeriodicSubtractTransform::name() const {
    char buf[96];
    if (period_ > 0.0)
        std::snprintf(buf, sizeof(buf), "Periodic Subtract (T=%.3f samples)", period_);
    else
        std::snprintf(buf, sizeof(buf), "Periodic Subtract (auto T in [%lld, %lld])",
                      static_cast<long long>(auto_min_), static_cast<long long>(auto_max_));
    return buf;
}

int64_t PeriodicSubtractTransform::apply(float* buf, int64_t count, int64_t) {
    last_period_ = 0.0;
    if (count < 16) return count;

    double T = period_;
    if (T <= 0.0)
        T = estimatePeriod(buf, count, auto_min_, auto_max_, auto_min_corr_, nullptr);
    // No convincing period: leave the trace alone rather than subtract noise
    // shaped like whatever the autocorrelation happened to favour.
    if (T < 2.0 || T > static_cast<double>(count) / 2.0) return count;
    last_period_ = T;

    // One template bin per sample of the period. Folding into fewer would
    // throw away resolution the trace actually has.
    const int64_t L = std::max<int64_t>(2, static_cast<int64_t>(std::llround(T)));
    accum_.assign(static_cast<size_t>(L), 0.0);
    weight_.assign(static_cast<size_t>(L), 0.0);

    // Accumulate with linear interpolation in phase: sample i lands at
    // fractional bin (i mod T)/T * L, and its value is split between the two
    // neighbouring bins. Rounding to the nearest bin instead would let the
    // template drift by a full sample every 1/frac(T) periods and smear the
    // very waveform we are trying to measure.
    for (int64_t i = 0; i < count; ++i) {
        const double ph  = std::fmod(static_cast<double>(i), T) / T * static_cast<double>(L);
        const int64_t b0 = static_cast<int64_t>(ph);
        const double  fr = ph - static_cast<double>(b0);
        const int64_t b1 = (b0 + 1) % L;
        const int64_t bb = b0 % L;
        const double  v  = static_cast<double>(buf[i]);
        accum_ [static_cast<size_t>(bb)] += v * (1.0 - fr);
        weight_[static_cast<size_t>(bb)] +=     (1.0 - fr);
        accum_ [static_cast<size_t>(b1)] += v * fr;
        weight_[static_cast<size_t>(b1)] +=     fr;
    }

    for (int64_t b = 0; b < L; ++b) {
        const double w = weight_[static_cast<size_t>(b)];
        accum_[static_cast<size_t>(b)] = (w > 1e-12)
            ? accum_[static_cast<size_t>(b)] / w : 0.0;
    }

    // Centre the template before subtracting. Its mean is the trace's own DC
    // level, which is not part of the interference — subtracting it would
    // shift the whole trace vertically and desynchronise it from every trace
    // this stage did not touch.
    double tmean = 0.0;
    for (int64_t b = 0; b < L; ++b) tmean += accum_[static_cast<size_t>(b)];
    tmean /= static_cast<double>(L);
    for (int64_t b = 0; b < L; ++b) accum_[static_cast<size_t>(b)] -= tmean;

    for (int64_t i = 0; i < count; ++i) {
        const double ph  = std::fmod(static_cast<double>(i), T) / T * static_cast<double>(L);
        const int64_t b0 = static_cast<int64_t>(ph);
        const double  fr = ph - static_cast<double>(b0);
        const double  e  = accum_[static_cast<size_t>(b0 % L)] * (1.0 - fr)
                         + accum_[static_cast<size_t>((b0 + 1) % L)] * fr;
        buf[i] = static_cast<float>(static_cast<double>(buf[i]) - e);
    }
    return count;
}

// ---------------------------------------------------------------------------
// ComplexDemodulateTransform
// ---------------------------------------------------------------------------
std::string ComplexDemodulateTransform::name() const {
    char buf[112];
    std::snprintf(buf, sizeof(buf), "Demodulate (f=%.4f×Nyq, BW=±%.4f, %s)",
                  static_cast<double>(center_), static_cast<double>(bandwidth_),
                  output_ == Output::Envelope ? "envelope" : "baseband");
    return buf;
}

int64_t ComplexDemodulateTransform::apply(float* buf, int64_t count, int64_t) {
    if (count < 8) return count;
    const int64_t N = count;

    if (fft_size_ != N) {            // Eigen's plan cache; see FFTMagnitudeTransform
        fft_      = Eigen::FFT<float>();
        fft_size_ = N;
    }
    // Full complex spectrum, not half: the band is shifted to DC by an index
    // rotation, which needs the negative frequencies to be addressable.
    fft_.ClearFlag(Eigen::FFT<float>::HalfSpectrum);

    in_scratch_.assign(static_cast<size_t>(N), std::complex<float>(0.0f, 0.0f));
    for (int64_t i = 0; i < N; ++i)
        in_scratch_[static_cast<size_t>(i)] = std::complex<float>(buf[i], 0.0f);
    fft_.fwd(freq_scratch_, in_scratch_);
    if (static_cast<int64_t>(freq_scratch_.size()) < N) return count;

    // Normalised frequency is a fraction of Nyquist, and bin N/2 is Nyquist,
    // so a fraction f maps to bin f * N/2.
    const double half_n = static_cast<double>(N) / 2.0;
    const int64_t k0  = static_cast<int64_t>(std::llround(static_cast<double>(center_) * half_n));
    const int64_t kbw = std::max<int64_t>(1,
        static_cast<int64_t>(std::llround(static_cast<double>(bandwidth_) * half_n)));

    // Y[m] = X[k0 + m] for m in [-kbw, kbw], zero elsewhere — the selected
    // band, with k0 moved to DC. Only the positive-frequency half is taken, so
    // the result is an analytic (one-sided) signal and its magnitude is the
    // envelope rather than a rectified carrier.
    band_scratch_.assign(static_cast<size_t>(N), std::complex<float>(0.0f, 0.0f));
    for (int64_t d = -kbw; d <= kbw; ++d) {
        const int64_t src = k0 + d;
        if (src < 0 || src >= N) continue;
        band_scratch_[static_cast<size_t>((d + N) % N)] = freq_scratch_[static_cast<size_t>(src)];
    }

    std::vector<std::complex<float>> base;
    fft_.inv(base, band_scratch_, static_cast<Eigen::Index>(N));
    if (static_cast<int64_t>(base.size()) < N) return count;

    // The factor of 2 restores the amplitude discarded with the negative
    // frequencies, so an input tone of amplitude A demodulates to an envelope
    // of A rather than A/2.
    for (int64_t i = 0; i < N; ++i) {
        const std::complex<float> v = base[static_cast<size_t>(i)];
        buf[i] = (output_ == Output::Envelope) ? 2.0f * std::abs(v)
                                                : 2.0f * v.real();
    }
    return N;
}
