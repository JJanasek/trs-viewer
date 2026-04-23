#include <gtest/gtest.h>
#include "processing.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> applyTransform(ITransform& tx, std::vector<float> data) {
    tx.reset();
    int64_t n = tx.apply(data.data(), static_cast<int64_t>(data.size()), 0);
    data.resize(static_cast<size_t>(std::max(INT64_C(0), n)));
    return data;
}

// ---------------------------------------------------------------------------
// AbsTransform
// ---------------------------------------------------------------------------

TEST(AbsTransform, NegativesBecomPositive) {
    AbsTransform tx;
    auto out = applyTransform(tx, {-3.f, 1.f, -0.5f, 0.f, 2.f});
    EXPECT_FLOAT_EQ(out[0], 3.f);
    EXPECT_FLOAT_EQ(out[1], 1.f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 0.f);
    EXPECT_FLOAT_EQ(out[4], 2.f);
}

TEST(AbsTransform, PositivesUnchanged) {
    AbsTransform tx;
    auto out = applyTransform(tx, {1.f, 2.f, 3.f});
    EXPECT_FLOAT_EQ(out[0], 1.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
    EXPECT_FLOAT_EQ(out[2], 3.f);
}

TEST(AbsTransform, EmptyBuffer) {
    AbsTransform tx;
    auto out = applyTransform(tx, {});
    EXPECT_TRUE(out.empty());
}

TEST(AbsTransform, TransformedCountUnchanged) {
    AbsTransform tx;
    EXPECT_EQ(tx.transformedCount(0),   0);
    EXPECT_EQ(tx.transformedCount(100), 100);
}

// ---------------------------------------------------------------------------
// NegateTransform
// ---------------------------------------------------------------------------

TEST(NegateTransform, FlipsSign) {
    NegateTransform tx;
    auto out = applyTransform(tx, {1.f, -2.f, 0.f});
    EXPECT_FLOAT_EQ(out[0], -1.f);
    EXPECT_FLOAT_EQ(out[1],  2.f);
    EXPECT_FLOAT_EQ(out[2],  0.f);
}

TEST(NegateTransform, DoubleNegateIsIdentity) {
    NegateTransform tx;
    std::vector<float> orig = {1.f, -2.f, 3.5f, 0.f};
    auto once  = applyTransform(tx, orig);
    auto twice = applyTransform(tx, once);
    for (size_t i = 0; i < orig.size(); ++i)
        EXPECT_FLOAT_EQ(twice[i], orig[i]);
}

// ---------------------------------------------------------------------------
// OffsetTransform
// ---------------------------------------------------------------------------

TEST(OffsetTransform, AddsPositiveOffset) {
    OffsetTransform tx(10.f);
    auto out = applyTransform(tx, {1.f, 2.f, 3.f});
    EXPECT_FLOAT_EQ(out[0], 11.f);
    EXPECT_FLOAT_EQ(out[1], 12.f);
    EXPECT_FLOAT_EQ(out[2], 13.f);
}

TEST(OffsetTransform, AddsNegativeOffset) {
    OffsetTransform tx(-5.f);
    auto out = applyTransform(tx, {3.f, 5.f, 10.f});
    EXPECT_FLOAT_EQ(out[0], -2.f);
    EXPECT_FLOAT_EQ(out[1],  0.f);
    EXPECT_FLOAT_EQ(out[2],  5.f);
}

TEST(OffsetTransform, ZeroOffsetIsIdentity) {
    OffsetTransform tx(0.f);
    auto out = applyTransform(tx, {1.f, -2.f, 3.f});
    EXPECT_FLOAT_EQ(out[0],  1.f);
    EXPECT_FLOAT_EQ(out[1], -2.f);
    EXPECT_FLOAT_EQ(out[2],  3.f);
}

// ---------------------------------------------------------------------------
// ScaleTransform
// ---------------------------------------------------------------------------

TEST(ScaleTransform, ScaleByTwo) {
    ScaleTransform tx(2.f);
    auto out = applyTransform(tx, {1.f, -3.f, 0.5f});
    EXPECT_FLOAT_EQ(out[0],  2.f);
    EXPECT_FLOAT_EQ(out[1], -6.f);
    EXPECT_FLOAT_EQ(out[2],  1.f);
}

TEST(ScaleTransform, ScaleByZeroGivesZeros) {
    ScaleTransform tx(0.f);
    auto out = applyTransform(tx, {1.f, 2.f, 3.f});
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.f);
}

TEST(ScaleTransform, ScaleByOneIsIdentity) {
    ScaleTransform tx(1.f);
    auto out = applyTransform(tx, {1.f, -2.f, 3.f});
    EXPECT_FLOAT_EQ(out[0],  1.f);
    EXPECT_FLOAT_EQ(out[1], -2.f);
    EXPECT_FLOAT_EQ(out[2],  3.f);
}

// ---------------------------------------------------------------------------
// MovingAverageTransform
// ---------------------------------------------------------------------------

TEST(MovingAverageTransform, Window1IsIdentity) {
    MovingAverageTransform tx(1);
    auto out = applyTransform(tx, {1.f, 2.f, 3.f, 4.f});
    EXPECT_FLOAT_EQ(out[0], 1.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
    EXPECT_FLOAT_EQ(out[2], 3.f);
    EXPECT_FLOAT_EQ(out[3], 4.f);
}

TEST(MovingAverageTransform, Window2AveragesPairs) {
    MovingAverageTransform tx(2);
    // Causal: out[i] = mean(in[i-W+1..i])
    // out[0] = in[0]              (only 1 element in ring)
    // out[1] = (in[0]+in[1])/2
    // out[2] = (in[1]+in[2])/2
    auto out = applyTransform(tx, {2.f, 4.f, 6.f, 8.f});
    EXPECT_FLOAT_EQ(out[0], 2.f);
    EXPECT_FLOAT_EQ(out[1], 3.f);
    EXPECT_FLOAT_EQ(out[2], 5.f);
    EXPECT_FLOAT_EQ(out[3], 7.f);
}

TEST(MovingAverageTransform, OutputSameLength) {
    MovingAverageTransform tx(4);
    std::vector<float> in(10, 1.f);
    auto out = applyTransform(tx, in);
    EXPECT_EQ(out.size(), 10u);
}

TEST(MovingAverageTransform, RequiresSequential) {
    MovingAverageTransform tx(3);
    EXPECT_TRUE(tx.requiresSequential());
}

TEST(MovingAverageTransform, StartupSamples) {
    MovingAverageTransform tx(5);
    EXPECT_EQ(tx.startupSamples(), 4);
}

TEST(MovingAverageTransform, ResetClearsState) {
    MovingAverageTransform tx(3);
    std::vector<float> in = {1.f, 2.f, 3.f, 4.f};

    // First pass
    std::vector<float> buf1 = in;
    tx.reset();
    tx.apply(buf1.data(), 4, 0);

    // After reset, second pass should give identical results
    std::vector<float> buf2 = in;
    tx.reset();
    tx.apply(buf2.data(), 4, 0);

    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(buf1[i], buf2[i]);
}

// ---------------------------------------------------------------------------
// WindowResampleTransform
// ---------------------------------------------------------------------------

TEST(WindowResampleTransform, DecimateByTwo) {
    WindowResampleTransform tx(2);
    // [1,3, 2,4, 5,7] → [2, 3, 6]
    auto out = applyTransform(tx, {1.f, 3.f, 2.f, 4.f, 5.f, 7.f});
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], 2.f);
    EXPECT_FLOAT_EQ(out[1], 3.f);
    EXPECT_FLOAT_EQ(out[2], 6.f);
}

TEST(WindowResampleTransform, TransformedCountCorrect) {
    WindowResampleTransform tx(4);
    EXPECT_EQ(tx.transformedCount(8),  2);
    EXPECT_EQ(tx.transformedCount(9),  2);
    EXPECT_EQ(tx.transformedCount(10), 2);
    EXPECT_EQ(tx.transformedCount(12), 3);
    EXPECT_EQ(tx.transformedCount(0),  0);
    EXPECT_EQ(tx.transformedCount(3),  0);
}

TEST(WindowResampleTransform, InputSmallerThanWindow) {
    WindowResampleTransform tx(10);
    auto out = applyTransform(tx, {1.f, 2.f, 3.f});
    EXPECT_TRUE(out.empty());
}

TEST(WindowResampleTransform, RequiresSequential) {
    WindowResampleTransform tx(4);
    EXPECT_TRUE(tx.requiresSequential());
}

// ---------------------------------------------------------------------------
// StrideResampleTransform
// ---------------------------------------------------------------------------

TEST(StrideResampleTransform, StrideTwo) {
    StrideResampleTransform tx(2);
    auto out = applyTransform(tx, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], 0.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
    EXPECT_FLOAT_EQ(out[2], 4.f);
}

TEST(StrideResampleTransform, StrideOne) {
    StrideResampleTransform tx(1);
    std::vector<float> in = {3.f, 1.f, 4.f, 1.f, 5.f};
    auto out = applyTransform(tx, in);
    EXPECT_EQ(out, in);
}

TEST(StrideResampleTransform, TransformedCountCorrect) {
    StrideResampleTransform tx(3);
    EXPECT_EQ(tx.transformedCount(9),  3);
    EXPECT_EQ(tx.transformedCount(10), 4);
    EXPECT_EQ(tx.transformedCount(0),  0);
    EXPECT_EQ(tx.transformedCount(1),  1);
}

TEST(StrideResampleTransform, LargerThanInputGivesOneSample) {
    StrideResampleTransform tx(100);
    auto out = applyTransform(tx, {7.f, 8.f, 9.f});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0], 7.f);
}

// ---------------------------------------------------------------------------
// FFTMagnitudeTransform
// ---------------------------------------------------------------------------

TEST(FFTMagnitudeTransform, OutputSizeIsHalfPlusOne) {
    FFTMagnitudeTransform tx(FFTMagnitudeTransform::Window::Rectangular);
    const int N = 256;
    std::vector<float> buf(N, 0.f);
    int64_t n_out = tx.apply(buf.data(), N, 0);
    EXPECT_EQ(n_out, N / 2 + 1);
    EXPECT_EQ(tx.transformedCount(N), N / 2 + 1);
}

TEST(FFTMagnitudeTransform, TransformedCountVariousSizes) {
    FFTMagnitudeTransform tx;
    EXPECT_EQ(tx.transformedCount(2),   2);
    EXPECT_EQ(tx.transformedCount(4),   3);
    EXPECT_EQ(tx.transformedCount(256), 129);
    EXPECT_EQ(tx.transformedCount(512), 257);
}

TEST(FFTMagnitudeTransform, AllZeroInputGivesAllZeroOutput) {
    FFTMagnitudeTransform tx(FFTMagnitudeTransform::Window::Rectangular);
    const int N = 128;
    std::vector<float> buf(N, 0.f);
    int64_t n_out = tx.apply(buf.data(), N, 0);
    for (int64_t k = 0; k < n_out; ++k)
        EXPECT_FLOAT_EQ(buf[k], 0.f) << "bin " << k;
}

TEST(FFTMagnitudeTransform, MagnitudesNonNegative) {
    FFTMagnitudeTransform tx(FFTMagnitudeTransform::Window::Hann);
    const int N = 128;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i) buf[i] = std::sin(2.f * M_PI * 8.f * i / N);
    int64_t n_out = tx.apply(buf.data(), N, 0);
    for (int64_t k = 0; k < n_out; ++k)
        EXPECT_GE(buf[k], 0.f) << "bin " << k;
}

TEST(FFTMagnitudeTransform, SinePeakAtExpectedBin) {
    // Pure sine at bin K0 → peak at bin K0 with rectangular window
    FFTMagnitudeTransform tx(FFTMagnitudeTransform::Window::Rectangular);
    const int N = 256;
    const int K0 = 16;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = std::sin(2.f * M_PI * K0 * i / N);

    int64_t n_out = tx.apply(buf.data(), N, 0);
    int peak_bin = static_cast<int>(
        std::max_element(buf.data(), buf.data() + n_out) - buf.data());
    EXPECT_EQ(peak_bin, K0);

    // Amplitude of a unit sine after one-sided normalization should be ≈ 1.0
    EXPECT_NEAR(buf[K0], 1.0f, 0.05f);
}

TEST(FFTMagnitudeTransform, RequiresSequential) {
    FFTMagnitudeTransform tx;
    EXPECT_TRUE(tx.requiresSequential());
}

// ---------------------------------------------------------------------------
// STFTMagnitudeTransform
// ---------------------------------------------------------------------------

TEST(STFTMagnitudeTransform, TransformedCountCorrect) {
    // num_windows = (N - W) / H + 1,  bins = W/2 + 1
    STFTMagnitudeTransform tx(256, 128);
    const int N = 1024;
    int64_t num_windows = (N - 256) / 128 + 1;  // 7
    int64_t bins        = 256 / 2 + 1;           // 129
    EXPECT_EQ(tx.transformedCount(N), num_windows * bins);
}

TEST(STFTMagnitudeTransform, InputSmallerThanWindowGivesZero) {
    STFTMagnitudeTransform tx(256, 128);
    EXPECT_EQ(tx.transformedCount(128), 0);
    EXPECT_EQ(tx.transformedCount(255), 0);
}

TEST(STFTMagnitudeTransform, OutputSizeMatchesTransformedCount) {
    STFTMagnitudeTransform tx(64, 32);
    const int N = 512;
    std::vector<float> buf(N, 0.f);
    int64_t n_out = tx.apply(buf.data(), N, 0);
    EXPECT_EQ(n_out, tx.transformedCount(N));
}

TEST(STFTMagnitudeTransform, MagnitudesNonNegative) {
    STFTMagnitudeTransform tx(64, 32, STFTMagnitudeTransform::Window::Hann);
    const int N = 512;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i) buf[i] = std::sin(2.f * M_PI * 4.f * i / 64);

    int64_t n_out = tx.apply(buf.data(), N, 0);
    for (int64_t k = 0; k < n_out; ++k)
        EXPECT_GE(buf[k], 0.f) << "index " << k;
}

TEST(STFTMagnitudeTransform, SinePeakPerWindowAtExpectedBin) {
    // Rectangular window: pure sine at K0 cycles/window → peak at bin K0
    const int W = 256, H = 128, K0 = 8, N = 1024;
    STFTMagnitudeTransform tx(W, H, STFTMagnitudeTransform::Window::Rectangular);

    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = std::sin(2.f * M_PI * K0 * i / W);

    int64_t n_out = tx.apply(buf.data(), N, 0);
    ASSERT_GT(n_out, 0);

    const int64_t bins        = W / 2 + 1;
    const int64_t num_windows = tx.transformedCount(N) / bins;

    for (int64_t wi = 0; wi < num_windows; ++wi) {
        float* spec  = buf.data() + wi * bins;
        int    pk    = static_cast<int>(std::max_element(spec, spec + bins) - spec);
        EXPECT_EQ(pk, K0) << "window " << wi;
    }
}

TEST(STFTMagnitudeTransform, RequiresSequential) {
    STFTMagnitudeTransform tx(64, 32);
    EXPECT_TRUE(tx.requiresSequential());
}

// ---------------------------------------------------------------------------
// Clone independence
// ---------------------------------------------------------------------------

TEST(Clone, MovingAverageIndependentState) {
    auto original = std::make_shared<MovingAverageTransform>(4);
    auto clone    = original->clone();

    // Advance state of original
    std::vector<float> buf(8, 1.f);
    original->apply(buf.data(), 8, 0);

    // Clone reset independently — its output should equal a fresh transform
    std::vector<float> buf_clone(4, 1.f);
    std::vector<float> buf_fresh(4, 1.f);
    clone->reset();
    clone->apply(buf_clone.data(), 4, 0);

    auto fresh = std::make_shared<MovingAverageTransform>(4);
    fresh->apply(buf_fresh.data(), 4, 0);

    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(buf_clone[i], buf_fresh[i]);
}

TEST(Clone, StrideResampleIndependentState) {
    auto tx    = std::make_shared<StrideResampleTransform>(3);
    auto clone = tx->clone();

    // Advance tx's position counter by processing 2 samples
    float tmp[2] = {1.f, 2.f};
    tx->apply(tmp, 2, 0);

    // Clone should start from position 0 (reset)
    clone->reset();
    float buf_clone[6] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    float buf_fresh[6] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    int64_t n_clone = clone->apply(buf_clone, 6, 0);

    auto fresh = std::make_shared<StrideResampleTransform>(3);
    int64_t n_fresh = fresh->apply(buf_fresh, 6, 0);

    ASSERT_EQ(n_clone, n_fresh);
    for (int64_t i = 0; i < n_clone; ++i)
        EXPECT_FLOAT_EQ(buf_clone[i], buf_fresh[i]);
}

// ---------------------------------------------------------------------------
// transformedCount matches actual apply output for all decimating transforms
// ---------------------------------------------------------------------------

TEST(TransformedCountConsistency, AllTransforms) {
    const std::vector<int64_t> input_sizes = {1, 2, 7, 16, 100, 257, 1000};

    for (int64_t N : input_sizes) {
        // WindowResample (window=4)
        {
            WindowResampleTransform tx(4);
            std::vector<float> buf(static_cast<size_t>(N), 1.f);
            int64_t predicted = tx.transformedCount(N);
            int64_t actual    = tx.apply(buf.data(), N, 0);
            EXPECT_EQ(predicted, actual) << "WindowResample N=" << N;
        }
        // StrideResample (stride=3)
        {
            StrideResampleTransform tx(3);
            std::vector<float> buf(static_cast<size_t>(N), 1.f);
            int64_t predicted = tx.transformedCount(N);
            int64_t actual    = tx.apply(buf.data(), N, 0);
            EXPECT_EQ(predicted, actual) << "StrideResample N=" << N;
        }
        // FFTMagnitude (needs N >= 2)
        if (N >= 2) {
            FFTMagnitudeTransform tx(FFTMagnitudeTransform::Window::Rectangular);
            std::vector<float> buf(static_cast<size_t>(N), 0.f);
            int64_t predicted = tx.transformedCount(N);
            int64_t actual    = tx.apply(buf.data(), N, 0);
            EXPECT_EQ(predicted, actual) << "FFTMagnitude N=" << N;
        }
        // STFT (W=16, H=8)
        if (N >= 16) {
            STFTMagnitudeTransform tx(16, 8, STFTMagnitudeTransform::Window::Rectangular);
            // Buffer must be large enough for expanded output
            int64_t max_out = std::max(N, tx.transformedCount(N));
            std::vector<float> buf(static_cast<size_t>(max_out), 0.f);
            int64_t predicted = tx.transformedCount(N);
            int64_t actual    = tx.apply(buf.data(), N, 0);
            EXPECT_EQ(predicted, actual) << "STFT N=" << N;
        }
    }
}
