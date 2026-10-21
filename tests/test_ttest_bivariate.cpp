#include <gtest/gtest.h>
#include "ttest_bivariate.h"
#include "ttest_run.h"
#include "align.h"

#include <cmath>
#include <vector>

// Software masking: the share m is handled at sample kS1 and m XOR v at
// sample kS2 (v = the group). Each sample alone is a uniform random bit in
// both groups, so neither the 1st-order nor the univariate 2nd-order test
// sees anything; only the pair (kS1, kS2) does: their centred product is
// positive for v = 0 and negative for v = 1.

namespace {

constexpr int kNS = 64, kS1 = 10, kS2 = 40;

double u01(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0;
}
double gauss(uint64_t seed) {
    const double a = std::max(1e-12, u01(seed * 2 + 1)), b = u01(seed * 2 + 2);
    return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b);
}

struct Shares {
    std::vector<float>   samples;
    std::vector<uint8_t> data;
    std::vector<int32_t> delay;   // per-trace delay of the second share
    TrsFile              file;
    Shares(int nt, int jitter = 0) {
        samples.resize(static_cast<size_t>(nt) * kNS);
        data.resize(static_cast<size_t>(nt));
        delay.resize(static_cast<size_t>(nt));
        for (int t = 0; t < nt; t++) {
            const int v = static_cast<int>(u01(t * 7 + 1) * 2);
            const int m = static_cast<int>(u01(t * 7 + 2) * 2);
            const int d = jitter ? static_cast<int>(u01(t * 7 + 3) * (jitter + 1)) : 0;
            data[static_cast<size_t>(t)]  = static_cast<uint8_t>(v);
            delay[static_cast<size_t>(t)] = d;
            for (int s = 0; s < kNS; s++) {
                double x = 0.3 * gauss(static_cast<uint64_t>(t) * 1000 + s);
                if (s == kS1)     x += m;
                if (s == kS2 + d) x += m ^ v;
                samples[static_cast<size_t>(t) * kNS + s] = static_cast<float>(x);
            }
        }
        file.openFromArray(samples.data(), nt, kNS, "shares", data.data(), 1);
    }
};

BivariateWindow whole(const char* name) {
    BivariateWindow w;
    w.name = name; w.raw_start = 0; w.raw_len = kNS;
    return w;
}

bool run(TrsFile& f, int nt, const BivariateWindow& a, const BivariateWindow& b,
         BivariateResult& r, bool sanity = false, bool abs_value = false) {
    QString warn, err;
    const bool ok = computeBivariateTTest(&f, {}, 0, nt, 0, a, b, 0, kNS, sanity, abs_value,
                                          {}, r, warn, err);
    EXPECT_TRUE(ok) << err.toStdString();
    return ok;
}

float at(const BivariateResult& r, int i, int j) { return r.t[static_cast<size_t>(i) * r.cols + j]; }

}  // namespace

TEST(TTestBivariate, UnivariateTestsAreBlind) {
    Shares d(4000);
    TTestOptions o;
    for (int order : {1, 2}) {
        o.order = order;
        std::shared_ptr<TTestAccumulator> acc;
        std::vector<float> t;
        int64_t n0, n1;
        QString w, e;
        ASSERT_TRUE(computeTTest(&d.file, {}, 0, 4000, 0, 0, 0, {}, 0, 0, kNS, false, o, {},
                                 acc, t, n0, n1, w, e));
        EXPECT_LT(std::fabs(t[kS1]), 4.5f) << "order " << order;
        EXPECT_LT(std::fabs(t[kS2]), 4.5f) << "order " << order;
    }
}

TEST(TTestBivariate, FindsThePairOfShares) {
    Shares d(4000);
    BivariateResult r;
    ASSERT_TRUE(run(d.file, 4000, whole("A"), whole("A"), r));
    EXPECT_TRUE(r.symmetric);
    ASSERT_EQ(r.rows, kNS);
    ASSERT_EQ(r.cols, kNS);
    EXPECT_GT(std::fabs(at(r, kS1, kS2)), 20.0f);
    EXPECT_EQ(std::min(r.max_row, r.max_col), kS1);
    EXPECT_EQ(std::max(r.max_row, r.max_col), kS2);
    // Symmetric, and every other pair stays quiet.
    EXPECT_FLOAT_EQ(at(r, kS2, kS1), at(r, kS1, kS2));
    float other = 0;
    for (int i = 0; i < kNS; i++)
        for (int j = i; j < kNS; j++)
            if (!(i == kS1 && j == kS2)) other = std::max(other, std::fabs(at(r, i, j)));
    EXPECT_LT(other, 5.5f);
    EXPECT_EQ(r.n0 + r.n1, 4000);
}

TEST(TTestBivariate, TwoWindowsGiveTheSameValueAsOne) {
    Shares d(1000);
    BivariateResult full, part;
    ASSERT_TRUE(run(d.file, 1000, whole("A"), whole("A"), full));
    BivariateWindow a = whole("A"), b = whole("B");
    a.raw_start = 5;  a.raw_len = 10;   // samples 5..14
    b.raw_start = 35; b.raw_len = 12;   // samples 35..46
    ASSERT_TRUE(run(d.file, 1000, a, b, part));
    EXPECT_FALSE(part.symmetric);
    ASSERT_EQ(part.rows, 10);
    ASSERT_EQ(part.cols, 12);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 12; j++)
            EXPECT_NEAR(at(part, i, j), at(full, 5 + i, 35 + j), 1e-3f) << i << "," << j;
}

TEST(TTestBivariate, MatchesDirectComputation) {
    Shares d(300);
    BivariateWindow a = whole("A"), b = whole("B");
    a.raw_start = 8;  a.raw_len = 4;
    b.raw_start = 38; b.raw_len = 5;
    BivariateResult r;
    ASSERT_TRUE(run(d.file, 300, a, b, r));
    std::vector<double> mu[2][2];   // [window][group]
    int cnt[2] = {0, 0};
    for (auto& w : mu) for (auto& g : w) g.assign(kNS, 0.0);
    for (int t = 0; t < 300; t++) {
        const int g = d.data[static_cast<size_t>(t)];
        cnt[g]++;
        for (int s = 0; s < kNS; s++) mu[0][g][static_cast<size_t>(s)] += d.samples[static_cast<size_t>(t) * kNS + s];
    }
    for (int g = 0; g < 2; g++) for (double& m : mu[0][g]) m /= cnt[g];
    TTestAccumulator ref(4 * 5);
    std::vector<float> y(20);
    for (int t = 0; t < 300; t++) {
        const int g = d.data[static_cast<size_t>(t)];
        const float* x = d.samples.data() + static_cast<size_t>(t) * kNS;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 5; j++)
                y[static_cast<size_t>(i * 5 + j)] = static_cast<float>(
                    (x[8 + i] - mu[0][g][static_cast<size_t>(8 + i)]) * (x[38 + j] - mu[0][g][static_cast<size_t>(38 + j)]));
        ref.addTrace(g, y.data(), 20);
    }
    std::vector<float> want;
    std::string err;
    ASSERT_TRUE(ref.compute(want, err));
    for (size_t k = 0; k < want.size(); k++) EXPECT_NEAR(r.t[k], want[k], 1e-3f) << k;
}

TEST(TTestBivariate, WindowShiftsLineUpAJitteredShare) {
    Shares d(4000, 3);   // second share delayed by 0..3 samples
    BivariateWindow a = whole("A"), b = whole("B");
    a.raw_start = kS1 - 2; a.raw_len = 5;
    b.raw_start = kS2 - 2; b.raw_len = 5;
    BivariateResult plain, shifted;
    ASSERT_TRUE(run(d.file, 4000, a, b, plain));
    b.shifts = d.delay;   // B's own shifts, as from pattern alignment
    ASSERT_TRUE(run(d.file, 4000, a, b, shifted));
    EXPECT_GT(shifted.max_abs, 1.5f * plain.max_abs);
    EXPECT_EQ(shifted.max_row, 2);
    EXPECT_EQ(shifted.max_col, 2);
}

TEST(TTestBivariate, DiscardedTracesAreLeftOutOfBothWindows) {
    Shares d(200);
    BivariateWindow a = whole("A"), b = whole("B");
    a.raw_len = 16; b.raw_start = 32; b.raw_len = 16;
    b.shifts.assign(200, 0);
    for (int t = 0; t < 200; t += 4) b.shifts[static_cast<size_t>(t)] = kAlignDiscardShift;
    BivariateResult r;
    QString warn, err;
    ASSERT_TRUE(computeBivariateTTest(&d.file, {}, 0, 200, 0, a, b, 0, kNS, false, false, {},
                                      r, warn, err)) << err.toStdString();
    EXPECT_EQ(r.n0 + r.n1, 150);
    EXPECT_TRUE(warn.contains("50 traces excluded")) << warn.toStdString();
}

TEST(TTestBivariate, SanitySplitIsClean) {
    Shares d(4000);
    BivariateResult r;
    ASSERT_TRUE(run(d.file, 4000, whole("A"), whole("A"), r, /*sanity=*/true));
    EXPECT_LT(r.max_abs, 5.5f);
    EXPECT_GT(r.n0, 800);
    EXPECT_GT(r.n1, 800);
}

TEST(TTestBivariate, TooManyPairsIsRefused) {
    Shares d(10);
    BivariateWindow a = whole("A");
    a.raw_len = 5000;
    BivariateResult r;
    QString warn, err;
    EXPECT_FALSE(computeBivariateTTest(&d.file, {}, 0, 10, 0, a, a, 0, kNS, false, false, {},
                                       r, warn, err));
    EXPECT_TRUE(err.contains("too many")) << err.toStdString();
    EXPECT_EQ(estimateBivariateBytes(1000, 1000, 8), 32000000 + 8 * 4000000);
}
