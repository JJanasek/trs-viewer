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

// ---------------------------------------------------------------------------
// mergeFrom: splitting traces across several accumulators and merging them
// must give the exact same result as accumulating everything into one —
// this is the correctness guarantee the parallel t-test loop relies on.
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, MergeFromMatchesDirectAccumulation) {
    const std::vector<float> g0 = {0.f, .2f, -.1f, .1f, .05f, -.05f};
    const std::vector<float> g1 = {1.f, 1.2f, .8f, 1.1f, 0.9f, 1.05f};

    TTestAccumulator direct(1);
    for (float v : g0) direct.addTrace(0, &v, 1);
    for (float v : g1) direct.addTrace(1, &v, 1);

    // Split the same traces across three "worker" accumulators, interleaved,
    // then merge them into a fresh accumulator — mirroring how the parallel
    // t-test loop distributes traces across threads.
    TTestAccumulator worker0(1), worker1(1), worker2(1);
    TTestAccumulator* workers[3] = {&worker0, &worker1, &worker2};
    for (size_t i = 0; i < g0.size(); i++) workers[i % 3]->addTrace(0, &g0[i], 1);
    for (size_t i = 0; i < g1.size(); i++) workers[i % 3]->addTrace(1, &g1[i], 1);

    TTestAccumulator merged(1);
    merged.mergeFrom(worker0);
    merged.mergeFrom(worker1);
    merged.mergeFrom(worker2);

    EXPECT_EQ(merged.countGroup(0), direct.countGroup(0));
    EXPECT_EQ(merged.countGroup(1), direct.countGroup(1));

    std::vector<float> direct_out, merged_out;
    std::string err;
    ASSERT_TRUE(direct.compute(direct_out, err)) << err;
    ASSERT_TRUE(merged.compute(merged_out, err)) << err;
    ASSERT_EQ(direct_out.size(), merged_out.size());
    for (size_t i = 0; i < direct_out.size(); i++)
        EXPECT_FLOAT_EQ(direct_out[i], merged_out[i]);
}

// ---------------------------------------------------------------------------
// computeMerged: the live-snapshot path computes t straight from the
// per-thread parts. It must equal reset() + mergeFrom() each part + compute()
// exactly, since it adds the parts in the same order per sample.
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, ComputeMergedEqualsMergeThenCompute) {
    constexpr int32_t kNs = 1000;
    std::vector<TTestAccumulator> parts(5, TTestAccumulator(kNs));
    std::vector<float> tr(kNs);
    for (int t = 0; t < 47; t++) {
        for (int32_t i = 0; i < kNs; i++)
            tr[static_cast<size_t>(i)] = std::sin(0.37f * t + 0.11f * i) * (1.0f + (t & 1) * 0.1f);
        parts[static_cast<size_t>(t % 5)].addTrace(t & 1, tr.data(), kNs);
    }
    TTestAccumulator merged(kNs);
    for (const auto& p : parts) merged.mergeFrom(p);
    std::vector<float> want, got1, got4;
    std::string err;
    ASSERT_TRUE(merged.compute(want, err)) << err;
    ASSERT_TRUE(TTestAccumulator::computeMerged(parts, got1, err, 1)) << err;
    ASSERT_TRUE(TTestAccumulator::computeMerged(parts, got4, err, 4)) << err;
    EXPECT_EQ(got1, want);
    EXPECT_EQ(got4, want);
}

TEST(TTestAccumulator, ComputeMergedNeedsTwoTracesPerGroup) {
    std::vector<TTestAccumulator> parts(3, TTestAccumulator(4));
    const float v[4] = {1, 2, 3, 4};
    parts[0].addTrace(0, v, 4);
    parts[1].addTrace(0, v, 4);
    parts[2].addTrace(1, v, 4);   // group 1 has only one trace in total
    std::vector<float> out;
    std::string err;
    EXPECT_FALSE(TTestAccumulator::computeMerged(parts, out, err, 2));
    EXPECT_NE(err.find("Group 1: 1"), std::string::npos) << err;
    std::vector<TTestAccumulator> none;
    EXPECT_FALSE(TTestAccumulator::computeMerged(none, out, err, 2));
}

// ---------------------------------------------------------------------------
// addTraces: parallel runs share one accumulator and add each group of
// traces split by sample range. Must equal addTrace() on each, in order.
// ---------------------------------------------------------------------------

TEST(TTestAccumulator, AddTracesEqualsAddTraceInOrder) {
    constexpr int32_t kNs = 100000;   // long enough to be split over threads
    TTestAccumulator one(kNs), batched(kNs);
    std::vector<std::vector<float>> trs(6, std::vector<float>(kNs));
    for (int t = 0; t < 6; t++)
        for (int32_t i = 0; i < kNs; i++)
            trs[static_cast<size_t>(t)][static_cast<size_t>(i)] = std::sin(0.37f * t + 0.011f * i) + 0.1f * (t & 1);
    const int     groups[6] = {0, 1, -1, 1, 0, 0};        // trace 2 is left out
    const int32_t counts[6] = {kNs, kNs, kNs, kNs - 777, kNs, kNs};
    std::vector<const float*> bufs;
    for (auto& tr : trs) bufs.push_back(tr.data());
    for (int t = 0; t < 6; t++)
        if (groups[t] >= 0) one.addTrace(groups[t], bufs[static_cast<size_t>(t)], counts[t]);
    batched.addTraces(groups, bufs.data(), counts, 3, 4);
    batched.addTraces(groups + 3, bufs.data() + 3, counts + 3, 3, 4);
    EXPECT_EQ(batched.countGroup(0), 3);
    EXPECT_EQ(batched.countGroup(1), 2);
    std::vector<float> want, got;
    std::string err;
    ASSERT_TRUE(one.compute(want, err)) << err;
    ASSERT_TRUE(batched.compute(got, err, 4)) << err;
    EXPECT_EQ(got, want);
}
