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
