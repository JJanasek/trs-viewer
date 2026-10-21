#include <gtest/gtest.h>
#include "ttest_run.h"

#include <algorithm>
#include <cmath>
#include <vector>

// A first-order-masked "device": at sample kLeak the trace carries
// m + (m XOR v) for a random mask bit m and the secret-dependent bit v
// (the group). Both groups have the same mean (1) but different variance
// (v=0: values 0/2, v=1: always 1), so a first-order t-test sees nothing and
// a second-order one must.

namespace {

constexpr int kNS   = 64;
constexpr int kLeak = 30;

double u01(uint64_t x) {   // deterministic uniform in [0,1)
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0;
}
double gauss(uint64_t seed) {   // Box–Muller
    const double a = std::max(1e-12, u01(seed * 2 + 1)), b = u01(seed * 2 + 2);
    return std::sqrt(-2 * std::log(a)) * std::cos(6.283185307179586 * b);
}

struct Masked {
    std::vector<float>   samples;
    std::vector<uint8_t> data;
    TrsFile              file;
    Masked(int nt, int jitter) {
        samples.resize(static_cast<size_t>(nt) * kNS);
        data.resize(static_cast<size_t>(nt));
        for (int t = 0; t < nt; t++) {
            const int v = static_cast<int>(u01(t * 7 + 1) * 2);
            const int m = static_cast<int>(u01(t * 7 + 2) * 2);
            const int j = jitter ? static_cast<int>(u01(t * 7 + 3) * (2 * jitter + 1)) - jitter : 0;
            data[static_cast<size_t>(t)] = static_cast<uint8_t>(v);
            for (int s = 0; s < kNS; s++) {
                double x = 0.4 * gauss(static_cast<uint64_t>(t) * 1000 + s);
                if (s == kLeak + j) x += m + (m ^ v);
                samples[static_cast<size_t>(t) * kNS + s] = static_cast<float>(x);
            }
        }
        file.openFromArray(samples.data(), nt, kNS, "masked", data.data(), 1);
    }
};

struct RunOut { bool ok; std::vector<float> t, t1; QString err; };

RunOut run(TrsFile& f, int nt, const TTestOptions& o) {
    RunOut r;
    std::shared_ptr<TTestAccumulator> acc;
    int64_t n0, n1;
    QString warn;
    r.ok = computeTTest(&f, {}, 0, nt, 0, 0, 0, {}, 0, 0, kNS, false, o, {},
                        acc, r.t, n0, n1, warn, r.err, nullptr, &r.t1);
    return r;
}

float maxAbs(const std::vector<float>& v, int skip_from = -1, int skip_to = -1) {
    float m = 0;
    for (int i = 0; i < static_cast<int>(v.size()); i++)
        if (i < skip_from || i > skip_to) m = std::max(m, std::fabs(v[static_cast<size_t>(i)]));
    return m;
}

}  // namespace

TEST(TTestOrder2, FirstOrderIsBlindToTheMaskedLeak) {
    Masked d(4000, 0);
    TTestOptions o;   // order 1
    RunOut r = run(d.file, 4000, o);
    ASSERT_TRUE(r.ok) << r.err.toStdString();
    EXPECT_LT(std::fabs(r.t[kLeak]), 4.5f);
}

TEST(TTestOrder2, SecondOrderFindsTheMaskedLeak) {
    Masked d(4000, 0);
    TTestOptions o; o.order = 2;
    RunOut r = run(d.file, 4000, o);
    ASSERT_TRUE(r.ok) << r.err.toStdString();
    ASSERT_EQ(r.t.size(), static_cast<size_t>(kNS));
    EXPECT_GT(std::fabs(r.t[kLeak]), 15.0f);
    EXPECT_LT(maxAbs(r.t, kLeak, kLeak), 5.5f);          // nowhere else
    ASSERT_EQ(r.t1.size(), static_cast<size_t>(kNS));    // pass-1 first order comes along
    EXPECT_LT(std::fabs(r.t1[kLeak]), 4.5f);
}

TEST(TTestOrder2, IntegrationWindowRecoversJitteredLeak) {
    Masked d(4000, 3);   // leak position jitters by ±3 samples
    TTestOptions o; o.order = 2;
    const float plain = maxAbs(run(d.file, 4000, o).t);
    o.integrate = 7;
    const float windowed = maxAbs(run(d.file, 4000, o).t);
    EXPECT_GT(windowed, 1.5f * plain);
    EXPECT_GT(windowed, 10.0f);
}

TEST(TTestOrder2, RandomSplitSanityCheckIsClean) {
    Masked d(4000, 0);
    TTestOptions o; o.order = 2; o.sanity_split = true;
    RunOut r = run(d.file, 4000, o);
    ASSERT_TRUE(r.ok) << r.err.toStdString();
    EXPECT_LT(maxAbs(r.t), 5.0f);
    o.order = 1;
    RunOut r1 = run(d.file, 4000, o);
    ASSERT_TRUE(r1.ok);
    EXPECT_LT(maxAbs(r1.t), 5.0f);
}

TEST(TTestOrder2, OrderOneMatchesTheCore) {
    Masked d(500, 0);
    TTestOptions o;
    RunOut a = run(d.file, 500, o);
    std::shared_ptr<TTestAccumulator> acc;
    std::vector<float> t;
    int64_t n0, n1;
    QString w, e;
    ASSERT_TRUE(computeTTestCore(&d.file, {}, 0, 500, 0, 0, 0, {}, 0, 0, kNS, false, {},
                                 acc, t, n0, n1, w, e));
    ASSERT_EQ(a.t.size(), t.size());
    for (size_t i = 0; i < t.size(); i++) EXPECT_NEAR(a.t[i], t[i], 1e-5f);
}

TEST(TTestOrder2, WindowSumMatchesDirectComputation) {
    Masked d(300, 2);
    TTestOptions o; o.order = 2; o.integrate = 5;
    RunOut r = run(d.file, 300, o);
    ASSERT_TRUE(r.ok) << r.err.toStdString();

    // Reference: group means, then y = Σ over the centred window of (x − μ)².
    std::vector<double> mu[2] = {std::vector<double>(kNS, 0.0), std::vector<double>(kNS, 0.0)};
    int cnt[2] = {0, 0};
    for (int t = 0; t < 300; t++) {
        const int g = d.data[static_cast<size_t>(t)];
        cnt[g]++;
        for (int s = 0; s < kNS; s++) mu[g][static_cast<size_t>(s)] += d.samples[static_cast<size_t>(t) * kNS + s];
    }
    for (int g = 0; g < 2; g++) for (double& m : mu[g]) m /= cnt[g];
    TTestAccumulator ref(kNS);
    std::vector<float> y(kNS);
    for (int t = 0; t < 300; t++) {
        const int g = d.data[static_cast<size_t>(t)];
        for (int s = 0; s < kNS; s++) {
            double sum = 0;
            for (int k = std::max(0, s - 2); k <= std::min(kNS - 1, s + 2); k++) {
                const double dd = d.samples[static_cast<size_t>(t) * kNS + k] - mu[g][static_cast<size_t>(k)];
                sum += dd * dd;
            }
            y[static_cast<size_t>(s)] = static_cast<float>(sum);
        }
        ref.addTrace(g, y.data(), kNS);
    }
    std::vector<float> want;
    std::string err;
    ASSERT_TRUE(ref.compute(want, err)) << err;
    ASSERT_EQ(r.t.size(), want.size());
    for (size_t i = 0; i < want.size(); i++) EXPECT_NEAR(r.t[i], want[i], 1e-3f) << "sample " << i;
}

TEST(TTestOrder2, MemoryEstimateIsOneAccumulatorPlusThreadBuffers) {
    // 1M samples, 8 threads: 32 MB accumulator + 8 × 4 MB trace buffers.
    EXPECT_EQ(estimateTTestBytes(1000000, 1000000, 1, 8), 64000000);
    // Order 2 adds the group means (2 doubles) and the 1st-order curve (a float).
    EXPECT_EQ(estimateTTestBytes(1000000, 1000000, 2, 8), 84000000);
    // A resampled pipeline: small accumulator, but raw-length buffers.
    EXPECT_EQ(estimateTTestBytes(1000000, 10000, 1, 8), 320000 + 32000000);
    // More cores than the run uses don't count.
    EXPECT_EQ(estimateTTestBytes(1000000, 1000000, 1, 24), 64000000);
}
