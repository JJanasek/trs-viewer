#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
// Window Resample: divide trace into non-overlapping blocks of `window_size`
// and replace each block with its mean.  Output has floor(N / window_size)
// samples — the trace is shortened (decimated).
// ---------------------------------------------------------------------------
class WindowResampleTransform : public ITransform {
public:
    explicit WindowResampleTransform(int window_size = 64);

    std::string name() const override;
    int64_t apply(float* buf, int64_t count, int64_t sample_offset) override;
    int64_t transformedCount(int64_t input_count) const override;
    void reset() override;
    bool requiresSequential() const override { return true; }
    // No startup transient — valid output from first complete block.
    int64_t startupSamples() const override { return 0; }

    void setWindowSize(int w);
    int  windowSize() const { return window_size_; }

    std::shared_ptr<ITransform> clone() const override {
        return std::make_shared<WindowResampleTransform>(*this);
    }

private:
    int     window_size_;
    // Partial-block state (carries over between chunks)
    double  partial_sum_   = 0.0;
    int     partial_count_ = 0;
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
