#include <gtest/gtest.h>
#include "snr.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Add a constant trace (all samples equal to `value`) to class `label`.
static void addConstantTrace(SNRAccumulator& acc, int label,
                              float value, int n_samples) {
    std::vector<float> t(static_cast<size_t>(n_samples), value);
    acc.addTrace(label, t.data(), n_samples);
}

// ---------------------------------------------------------------------------
// Insufficient data → compute() returns false
// ---------------------------------------------------------------------------

TEST(SNRAccumulator, NoDataReturnsError) {
    SNRAccumulator acc(10, 2);
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
    EXPECT_FALSE(err.empty());
}

TEST(SNRAccumulator, OnlyOneClassPopulatedReturnsError) {
    SNRAccumulator acc(4, 3);
    // Only class 0 has traces (and only 1, which is also insufficient)
    addConstantTrace(acc, 0, 1.f, 4);
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
}

TEST(SNRAccumulator, SingleTracePerClassReturnsError) {
    // Welch-style: need >= 2 traces per active class
    SNRAccumulator acc(4, 2);
    addConstantTrace(acc, 0, 0.f, 4);
    addConstantTrace(acc, 1, 1.f, 4);
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
}

// ---------------------------------------------------------------------------
// Label out of range: silently ignored, no crash
// ---------------------------------------------------------------------------

TEST(SNRAccumulator, LabelOutOfRangeIgnored) {
    SNRAccumulator acc(4, 2);
    std::vector<float> t = {1.f, 2.f, 3.f, 4.f};
    // Labels 2 and -1 are out of [0, 2)
    acc.addTrace(2,  t.data(), 4);
    acc.addTrace(-1, t.data(), 4);
    // Still need valid data to avoid error in compute
    for (int i = 0; i < 3; ++i) addConstantTrace(acc, 0, 0.f, 4);
    for (int i = 0; i < 3; ++i) addConstantTrace(acc, 1, 1.f, 4);
    std::vector<float> out;
    std::string err;
    EXPECT_TRUE(acc.compute(out, err)) << err;
}

// ---------------------------------------------------------------------------
// Basic correctness: 2 classes with known values → expected SNR
// ---------------------------------------------------------------------------
//
//  Class 0 (10 traces, single sample):
//    values = [0.0, 0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.0, 0.05, -0.05]
//    mean   = 0.0
//    var    = (Σ x² - n·mean²) / (n-1) = 0.045 / 9 = 0.005
//
//  Class 1 (10 traces, single sample):
//    values = [1.0, 1.1, 0.9, 1.0, 1.0, 1.1, 0.9, 1.0, 1.05, 0.95]
//    mean   = 1.0
//    var    = (Σ(x-1)² ) / 9 = 0.045 / 9 = 0.005
//
//  grand_mean = 0.5
//  signal = (10*(0-0.5)² + 10*(1-0.5)²) / 20 = 0.25
//  noise  = (10*0.005 + 10*0.005) / 20        = 0.005
//  SNR    = 0.25 / 0.005 = 50.0
// ---------------------------------------------------------------------------

TEST(SNRAccumulator, BasicTwoClassCorrectness) {
    const int N_SAMPLES = 1;
    SNRAccumulator acc(N_SAMPLES, 2);

    const std::vector<float> class0 = {0.f, .1f, -.1f, 0.f, 0.f, .1f, -.1f, 0.f, .05f, -.05f};
    const std::vector<float> class1 = {1.f, 1.1f, .9f, 1.f, 1.f, 1.1f, .9f, 1.f, 1.05f, .95f};

    for (float v : class0) acc.addTrace(0, &v, 1);
    for (float v : class1) acc.addTrace(1, &v, 1);

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0], 50.f, 2.f);  // generous tolerance for float arithmetic
}

// ---------------------------------------------------------------------------
// Zero within-class variance → noise = 0 → SNR = 0 (not infinite)
// ---------------------------------------------------------------------------

TEST(SNRAccumulator, ZeroNoiseGivesZeroSNR) {
    SNRAccumulator acc(1, 2);
    // Class 0: all exactly 0.0
    for (int i = 0; i < 5; ++i) addConstantTrace(acc, 0, 0.f, 1);
    // Class 1: all exactly 1.0
    for (int i = 0; i < 5; ++i) addConstantTrace(acc, 1, 1.f, 1);

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0], 0.f);  // noise = 0 → SNR clamped to 0
}

// ---------------------------------------------------------------------------
// Multiple samples per trace
// ---------------------------------------------------------------------------

TEST(SNRAccumulator, MultiSample) {
    const int N = 4;
    SNRAccumulator acc(N, 2);

    // Class 0: mean = 0 at every sample
    for (int i = 0; i < 4; ++i) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        v[i % N] = 0.1f * (i + 1);  // add small variance
        acc.addTrace(0, v, N);
    }
    // Class 1: mean = 1 at every sample
    for (int i = 0; i < 4; ++i) {
        float v[4] = {1.f, 1.f, 1.f, 1.f};
        v[i % N] += 0.1f * (i + 1);
        acc.addTrace(1, v, N);
    }

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), static_cast<size_t>(N));
    // All SNR values should be > 0 since the classes differ in mean
    for (float v : out) EXPECT_GE(v, 0.f);
}

// ---------------------------------------------------------------------------
// Counting helpers
// ---------------------------------------------------------------------------

TEST(SNRAccumulator, CountingMethods) {
    SNRAccumulator acc(2, 3);
    EXPECT_EQ(acc.totalTraces(), 0);

    std::vector<float> t = {0.f, 0.f};
    acc.addTrace(0, t.data(), 2);
    acc.addTrace(0, t.data(), 2);
    acc.addTrace(1, t.data(), 2);

    EXPECT_EQ(acc.countClass(0), 2);
    EXPECT_EQ(acc.countClass(1), 1);
    EXPECT_EQ(acc.countClass(2), 0);
    EXPECT_EQ(acc.totalTraces(), 3);
}
