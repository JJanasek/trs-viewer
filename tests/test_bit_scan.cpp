#include <gtest/gtest.h>
#include <utility>
#include "bit_scan.h"
#include "cpa.h"
#include "ttest_run.h"

#include <cmath>
#include <vector>

namespace {

constexpr int kN = 500, kNS = 64, kBytes = 3;

float noise(int t, int s) {
    const double x = std::sin((t * 131 + s * 17) * 12.9898) * 43758.5453;
    return static_cast<float>(x - std::floor(x)) - 0.5f;
}
uint8_t randomByte(int t, int b) {
    const double x = std::sin((t * 71 + b * 311 + 5) * 78.233) * 12543.123;
    return static_cast<uint8_t>(static_cast<int>((x - std::floor(x)) * 256.0));
}

// Data byte 0 is something else; bytes 1..2 are the label. Bit 3 of byte 1
// leaks at sample 20, bit 6 of byte 2 at sample 40. Bit 0 of byte 2 is never set.
struct Fixture {
    std::vector<float>   x;
    std::vector<uint8_t> d;
    TrsFile file;
    Fixture() {
        x.resize(static_cast<size_t>(kN) * kNS);
        d.resize(static_cast<size_t>(kN) * kBytes);
        for (int t = 0; t < kN; t++) {
            uint8_t* row = d.data() + static_cast<size_t>(t) * kBytes;
            for (int b = 0; b < kBytes; b++) row[b] = randomByte(t, b);
            row[2] &= 0xFE;
            for (int s = 0; s < kNS; s++) {
                float v = 100.0f + noise(t, s);
                if (s == 20 && ((row[1] >> 3) & 1)) v += 0.4f;
                if (s == 40 && ((row[2] >> 6) & 1)) v -= 0.4f;
                x[static_cast<size_t>(t) * kNS + s] = v;
            }
        }
        file.openFromArray(x.data(), kN, kNS, "scan", d.data(), kBytes);
    }
};

std::vector<int32_t> allTraces() { return cpaTraces(kN, {}); }

bool scan(Fixture& f, BitScanResult& r, bool curves = true, int64_t block_bytes = kBitScanBlockBytes) {
    std::string err;
    const bool ok = computeBitScan(&f.file, 0, allTraces(), 0, 0, {}, {}, 1, 2, curves, r, {}, err, block_bytes);
    EXPECT_TRUE(ok) << err;
    return ok;
}

}  // namespace

TEST(BitScan, EveryBitMatchesItsOwnTTest) {
    Fixture f;
    BitScanResult r;
    ASSERT_TRUE(scan(f, r));
    ASSERT_EQ(r.n_bits, 16);
    ASSERT_EQ(r.n_samples, kNS);
    EXPECT_EQ(r.n_traces, kN);
    for (int k = 0; k < 16; k++) {
        if (k == 8) continue;   // the bit that never changes
        std::vector<float> t;
        int64_t n0 = 0, n1 = 0;
        std::shared_ptr<TTestAccumulator> acc;
        QString warn, err;
        ASSERT_TRUE(computeTTestCore(&f.file, {}, 0, kN, 0, 0, GroupBy(1 + k / 8, static_cast<uint8_t>(1u << (k % 8))),
                                     {}, 0, 0, kNS, false, {}, acc, t, n0, n1, warn, err)) << err.toStdString();
        EXPECT_EQ(r.n0[static_cast<size_t>(k)], n0);
        EXPECT_EQ(r.n1[static_cast<size_t>(k)], n1);
        for (int s = 0; s < kNS; s++)
            EXPECT_NEAR(r.t[static_cast<size_t>(k) * kNS + s], t[static_cast<size_t>(s)], 1e-3f) << k << " " << s;
    }
}

TEST(BitScan, FindsTheBitsThatLeak) {
    Fixture f;
    BitScanResult r;
    ASSERT_TRUE(scan(f, r));
    int best = 0, second = -1;
    for (int k = 1; k < 16; k++) {
        if (std::fabs(r.peak_t[k]) > std::fabs(r.peak_t[best])) { second = best; best = k; }
        else if (second < 0 || std::fabs(r.peak_t[k]) > std::fabs(r.peak_t[second])) second = k;
    }
    const std::pair<int, int> top = std::minmax(best, second);
    EXPECT_EQ(top.first, 3);          // byte 1 bit 3
    EXPECT_EQ(top.second, 8 + 6);     // byte 2 bit 6
    EXPECT_EQ(r.peak_sample[3], 20);
    EXPECT_EQ(r.peak_sample[14], 40);
    EXPECT_GE(r.n_over[3], 1);
}

TEST(BitScan, ABitThatNeverChangesIsFlat) {
    Fixture f;
    BitScanResult r;
    ASSERT_TRUE(scan(f, r));
    EXPECT_EQ(r.n1[8], 0);
    EXPECT_EQ(r.n0[8], kN);
    EXPECT_EQ(r.peak_t[8], 0.0f);
    EXPECT_EQ(r.n_over[8], 0);
    for (int s = 0; s < kNS; s++) EXPECT_EQ(r.t[8u * kNS + static_cast<size_t>(s)], 0.0f);
}

TEST(BitScan, SmallBlocksGiveTheSameResult) {
    Fixture f;
    BitScanResult one, many;
    ASSERT_TRUE(scan(f, one));
    ASSERT_TRUE(scan(f, many, true, /*block_bytes=*/1));   // one sample at a time
    EXPECT_EQ(one.t, many.t);
    EXPECT_EQ(one.peak_sample, many.peak_sample);
    EXPECT_EQ(one.n_over, many.n_over);
}

TEST(BitScan, WithoutCurvesTheStatisticsAreTheSame) {
    Fixture f;
    BitScanResult with, without;
    ASSERT_TRUE(scan(f, with));
    ASSERT_TRUE(scan(f, without, false));
    EXPECT_FALSE(without.hasCurves());
    EXPECT_EQ(with.peak_t, without.peak_t);
    EXPECT_EQ(with.peak_sample, without.peak_sample);
    EXPECT_EQ(with.n_over, without.n_over);
}

TEST(BitScan, DiscardedTracesAndARangeAreRespected) {
    Fixture f;
    std::vector<int32_t> shifts(kN, 0);
    for (int t = 0; t < 100; t++) shifts[static_cast<size_t>(t)] = kAlignDiscardShift;
    BitScanResult r;
    std::string err;
    ASSERT_TRUE(computeBitScan(&f.file, 0, cpaTraces(kN, shifts), 10, 20, shifts, {}, 1, 1, true, r, {}, err)) << err;
    EXPECT_EQ(r.n_traces, 400);
    EXPECT_EQ(r.n_samples, 20);
    EXPECT_EQ(r.n_bits, 8);
    EXPECT_EQ(r.n0[3] + r.n1[3], 400);
    EXPECT_EQ(r.peak_sample[3], 10);   // sample 20, from the range's start
}

TEST(BitScan, RefusesBytesOutsideTheData) {
    Fixture f;
    BitScanResult r;
    std::string err;
    EXPECT_FALSE(computeBitScan(&f.file, 0, allTraces(), 0, 0, {}, {}, 2, 2, true, r, {}, err));
    EXPECT_FALSE(err.empty());
}

TEST(BitScan, CancelStops) {
    Fixture f;
    BitScanResult r;
    std::string err;
    EXPECT_FALSE(computeBitScan(&f.file, 0, allTraces(), 0, 0, {}, {}, 1, 2, true, r,
                                [](int64_t, int64_t) { return false; }, err));
    EXPECT_EQ(err, "Cancelled.");
}
