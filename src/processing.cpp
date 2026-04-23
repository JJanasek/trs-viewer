#include "processing.h"

#include <cmath>
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
// WindowResampleTransform  (block decimation: floor(N/W) output samples)
// ---------------------------------------------------------------------------
WindowResampleTransform::WindowResampleTransform(int window_size)
    : window_size_(std::max(1, window_size))
{}

std::string WindowResampleTransform::name() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Window Resample (w=%d)", window_size_);
    return buf;
}

int64_t WindowResampleTransform::transformedCount(int64_t input_count) const {
    // Number of complete blocks (partial last block is discarded)
    return input_count / window_size_;
}

void WindowResampleTransform::reset() {
    partial_sum_   = 0.0;
    partial_count_ = 0;
}

void WindowResampleTransform::setWindowSize(int w) {
    window_size_ = std::max(1, w);
    reset();
}

int64_t WindowResampleTransform::apply(float* buf, int64_t count, int64_t) {
    int64_t out = 0;
    int64_t i   = 0;

    // First: try to complete any partial block carried over from the previous chunk
    if (partial_count_ > 0) {
        while (i < count && partial_count_ < window_size_) {
            partial_sum_ += buf[i++];
            partial_count_++;
        }
        if (partial_count_ == window_size_) {
            buf[out++] = static_cast<float>(partial_sum_ / window_size_);
            partial_sum_   = 0.0;
            partial_count_ = 0;
        }
        // else: still not enough samples to complete a block — keep accumulating
    }

    // Full blocks from the remaining input
    while (i + window_size_ <= count) {
        double sum = 0.0;
        for (int j = 0; j < window_size_; j++)
            sum += buf[i + j];
        buf[out++] = static_cast<float>(sum / window_size_);
        i += window_size_;
    }

    // Save any trailing partial block for the next chunk
    while (i < count) {
        partial_sum_ += buf[i++];
        partial_count_++;
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

    std::vector<float> win;
    buildWindowCoeffs(win, static_cast<int>(N), static_cast<WinType>(static_cast<int>(window_)));

    std::vector<float> in_vec(N);
    for (int64_t i = 0; i < N; ++i) in_vec[i] = buf[i] * win[i];

    std::vector<std::complex<float>> freq_vec;
    Eigen::FFT<float> fft;
    fft.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    fft.fwd(freq_vec, in_vec);

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

    std::vector<float> win;
    buildWindowCoeffs(win, static_cast<int>(W), static_cast<WinType>(static_cast<int>(window_)));

    // Write results to a separate buffer to avoid aliasing the input.
    std::vector<float> out(static_cast<size_t>(out_n));

    Eigen::FFT<float> fft;
    fft.SetFlag(Eigen::FFT<float>::HalfSpectrum);
    std::vector<float>               in_vec(static_cast<size_t>(W));
    std::vector<std::complex<float>> freq_vec;

    const float norm = 1.0f / static_cast<float>(W);

    for (int64_t wi = 0; wi < num_windows; ++wi) {
        const int64_t pos = wi * hop_size_;
        for (int64_t i = 0; i < W; ++i)
            in_vec[i] = buf[pos + i] * win[i];

        fft.fwd(freq_vec, in_vec);

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
