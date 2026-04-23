#include <gtest/gtest.h>
#include "ttest.h"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void addTrace(TTestAccumulator& acc, int group,
                     std::initializer_list<float> vals) {
    std::vector<float> v = vals;
    acc.addTrace(group, v.data(), static_cast<int32_t>(v.size()));
}

// ---------------------------------------------------------------------------
// Insufficient data → compute() returns false
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, NoDataReturnsError) {
    TTestAccumulator acc(8);
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
    EXPECT_FALSE(err.empty());
}

TEST(TTestAccumulator, OnlyGroup0ReturnsError) {
    TTestAccumulator acc(4);
    for (int i = 0; i < 5; ++i) addTrace(acc, 0, {0.f, 1.f, 2.f, 3.f});
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
}

TEST(TTestAccumulator, OnlyGroup1ReturnsError) {
    TTestAccumulator acc(4);
    for (int i = 0; i < 5; ++i) addTrace(acc, 1, {0.f, 1.f, 2.f, 3.f});
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
}

TEST(TTestAccumulator, SingleTracePerGroupReturnsError) {
    TTestAccumulator acc(2);
    addTrace(acc, 0, {0.f, 0.f});
    addTrace(acc, 1, {1.f, 1.f});
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));
}

// ---------------------------------------------------------------------------
// Group label outside {0,1}: silently ignored
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, InvalidGroupLabelIgnored) {
    TTestAccumulator acc(2);
    for (int i = 0; i < 4; ++i) addTrace(acc, 0, {0.f, 0.f});
    for (int i = 0; i < 4; ++i) addTrace(acc, 1, {1.f, 1.f});
    // These should be silently dropped, not crash
    addTrace(acc, 2,  {99.f, 99.f});
    addTrace(acc, -1, {99.f, 99.f});

    std::vector<float> out;
    std::string err;
    EXPECT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 2u);
    // With zero within-group variance and means 0 vs 1, t would be very large
    // or 0 (if denominator clips to 0). Either way, no crash.
}

// ---------------------------------------------------------------------------
// Identical groups → t ≈ 0
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, IdenticalGroupsGiveZeroT) {
    TTestAccumulator acc(1);
    for (int i = 0; i < 10; ++i) {
        float v = static_cast<float>(i);
        acc.addTrace(0, &v, 1);
    }
    for (int i = 0; i < 10; ++i) {
        float v = static_cast<float>(i);
        acc.addTrace(1, &v, 1);
    }

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0], 0.f, 1e-4f);
}

// ---------------------------------------------------------------------------
// Known t-statistic
//
//  Group 0: [0.0,  0.2, -0.1,  0.1]  → mean0 = 0.05,  var0 ≈ 0.016667
//  Group 1: [1.0,  1.2,  0.8,  1.1]  → mean1 = 1.025,  var1 ≈ 0.029167
//
//  Welch t = (mean0 - mean1) / sqrt(var0/n0 + var1/n1)
//          = (0.05 - 1.025) / sqrt(0.016667/4 + 0.029167/4)
//          = -0.975 / sqrt(0.004167 + 0.007292)
//          = -0.975 / sqrt(0.011458)
//          ≈ -0.975 / 0.10705
//          ≈ -9.11
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, KnownTStatistic) {
    TTestAccumulator acc(1);

    const std::vector<float> g0 = {0.f, .2f, -.1f, .1f};
    const std::vector<float> g1 = {1.f, 1.2f, .8f, 1.1f};
    for (float v : g0) acc.addTrace(0, &v, 1);
    for (float v : g1) acc.addTrace(1, &v, 1);

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0], -9.11f, 0.1f);
}

// ---------------------------------------------------------------------------
// Sign convention: group0 > group1 → t > 0
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, SignConvention) {
    TTestAccumulator acc(1);
    // Group 0 has higher mean
    const std::vector<float> g0 = {2.f, 2.1f, 1.9f, 2.2f};
    const std::vector<float> g1 = {0.f, 0.1f, -0.1f, 0.05f};
    for (float v : g0) acc.addTrace(0, &v, 1);
    for (float v : g1) acc.addTrace(1, &v, 1);

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    EXPECT_GT(out[0], 0.f);
}

// ---------------------------------------------------------------------------
// Zero variance in both groups → denominator = 0 → t = 0
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, ZeroVarianceBothGroupsGivesZero) {
    TTestAccumulator acc(1);
    for (int i = 0; i < 5; ++i) { float v = 0.f; acc.addTrace(0, &v, 1); }
    for (int i = 0; i < 5; ++i) { float v = 1.f; acc.addTrace(1, &v, 1); }

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    ASSERT_EQ(out.size(), 1u);
    // denominator = 0 → implementation returns 0
    EXPECT_FLOAT_EQ(out[0], 0.f);
}

// ---------------------------------------------------------------------------
// reset() clears state
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, ResetClearsState) {
    TTestAccumulator acc(1);
    for (int i = 0; i < 5; ++i) { float v = 10.f; acc.addTrace(0, &v, 1); }
    for (int i = 0; i < 5; ++i) { float v = -10.f; acc.addTrace(1, &v, 1); }

    acc.reset();
    EXPECT_EQ(acc.countGroup(0), 0);
    EXPECT_EQ(acc.countGroup(1), 0);

    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(acc.compute(out, err));  // no data after reset
}

// ---------------------------------------------------------------------------
// Multi-sample output size
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, OutputSizeMatchesNumSamples) {
    const int N = 16;
    TTestAccumulator acc(N);
    std::vector<float> t(N, 0.f);
    for (int i = 0; i < 5; ++i) acc.addTrace(0, t.data(), N);
    for (int i = 0; i < 5; ++i) { t[0] = 1.f; acc.addTrace(1, t.data(), N); }

    std::vector<float> out;
    std::string err;
    ASSERT_TRUE(acc.compute(out, err)) << err;
    EXPECT_EQ(out.size(), static_cast<size_t>(N));
}

// ---------------------------------------------------------------------------
// estimatedBytes
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, EstimatedBytes) {
    TTestAccumulator acc(100);
    // 4 * 100 * sizeof(double) = 3200 bytes
    EXPECT_EQ(acc.estimatedBytes(), 4LL * 100 * static_cast<int64_t>(sizeof(double)));
}
