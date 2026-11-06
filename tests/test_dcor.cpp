#include <gtest/gtest.h>
#include "dcor.h"

#include <cmath>
#include <vector>

namespace {

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

// Pairwise-distance sums straight from the definition, O(n²).
DistanceSums naiveSums(const std::vector<float>& x, int64_t p, const std::vector<uint8_t>& g) {
    const int64_t n = static_cast<int64_t>(g.size());
    DistanceSums s;
    std::vector<double> row(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < n; i++) {
        (g[i] ? s.n1 : s.n0)++;
        for (int64_t j = 0; j < n; j++) {
            if (i == j) continue;
            double d2 = 0;
            for (int64_t c = 0; c < p; c++) {
                const double d = static_cast<double>(x[i * p + c]) - x[j * p + c];
                d2 += d * d;
            }
            const double d = std::sqrt(d2);
            row[i] += d;
            s.sq += d2;
            if (j > i) (g[i] != g[j] ? s.cross : (g[i] ? s.within1 : s.within0)) += d;
        }
    }
    for (double r : row) s.row_sq += r * r;
    return s;
}

// Eq. (4)/(5) of the paper written out with the class distances b_ij.
double naiveDcor(const std::vector<float>& x, int64_t p, const std::vector<uint8_t>& g) {
    const int64_t n = static_cast<int64_t>(g.size());
    auto A = [&](int64_t i, int64_t j) {
        double d2 = 0;
        for (int64_t c = 0; c < p; c++) {
            const double d = static_cast<double>(x[i * p + c]) - x[j * p + c];
            d2 += d * d;
        }
        return std::sqrt(d2);
    };
    auto B = [&](int64_t i, int64_t j) { return g[i] != g[j] ? 1.0 : 0.0; };
    auto omega = [&](auto&& a, auto&& b) {
        double sab = 0, ar_br = 0, aa = 0, bb = 0;
        std::vector<double> ar(static_cast<size_t>(n), 0.0), br(static_cast<size_t>(n), 0.0);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) {
                if (i == j) continue;
                sab += a(i, j) * b(i, j);
                ar[i] += a(i, j); br[i] += b(i, j);
                aa += a(i, j); bb += b(i, j);
            }
        for (int64_t i = 0; i < n; i++) ar_br += ar[i] * br[i];
        const double nn = static_cast<double>(n);
        return sab / (nn * (nn - 3)) - 2 * ar_br / (nn * (nn - 2) * (nn - 3))
             + aa * bb / (nn * (nn - 1) * (nn - 2) * (nn - 3));
    };
    const double xy = omega(A, B), xx = omega(A, A), yy = omega(B, B);
    return xx > 0 && yy > 0 ? xy / std::sqrt(xx * yy) : 0.0;
}

void expectSums(const DistanceSums& a, const DistanceSums& b, double tol) {
    EXPECT_EQ(a.n0, b.n0);
    EXPECT_EQ(a.n1, b.n1);
    EXPECT_NEAR(a.within0, b.within0, tol * std::max(1.0, std::fabs(b.within0)));
    EXPECT_NEAR(a.within1, b.within1, tol * std::max(1.0, std::fabs(b.within1)));
    EXPECT_NEAR(a.cross,   b.cross,   tol * std::max(1.0, std::fabs(b.cross)));
    EXPECT_NEAR(a.row_sq,  b.row_sq,  tol * std::max(1.0, std::fabs(b.row_sq)));
    EXPECT_NEAR(a.sq,      b.sq,      tol * std::max(1.0, std::fabs(b.sq)));
}

}  // namespace

TEST(Dcor, OneDimensionalSumsMatchTheDefinition) {
    std::vector<float> x;
    std::vector<uint8_t> g;
    for (int i = 0; i < 57; i++) {
        x.push_back(static_cast<float>(gauss(i) * 3.0 + (i % 7 == 0 ? 2.0 : 0.0)));
        g.push_back(static_cast<uint8_t>(u01(1000 + i) < 0.4));
    }
    x[5] = x[9];   // ties
    std::vector<std::pair<float, uint8_t>> scratch;
    expectSums(distanceSums1D(x.data(), g.data(), 57, scratch), naiveSums(x, 1, g), 1e-9);
}

TEST(Dcor, HistogramSumsMatchTheDefinition) {
    // 8-bit samples: integers −128…127, many ties.
    std::vector<float> x;
    std::vector<uint8_t> g;
    std::vector<uint32_t> c0(256, 0), c1(256, 0);
    for (int i = 0; i < 80; i++) {
        const int v = static_cast<int>(std::lround(gauss(i) * 20.0)) + (i & 1) * 3;
        const uint8_t c = static_cast<uint8_t>(i & 1);
        x.push_back(static_cast<float>(v));
        g.push_back(c);
        (c ? c1 : c0)[static_cast<size_t>(v + 128)]++;
    }
    expectSums(distanceSumsHistogram(c0.data(), c1.data(), 256, -128.0), naiveSums(x, 1, g), 1e-9);
}

TEST(Dcor, RowSumsMatchTheDefinition) {
    // 600 rows (two 512-row blocks, so off-diagonal blocks are covered) × 7.
    const int64_t n = 600, p = 7;
    std::vector<float> x;
    std::vector<uint8_t> g;
    for (int64_t i = 0; i < n; i++) {
        g.push_back(static_cast<uint8_t>(i % 3 == 0));
        for (int64_t c = 0; c < p; c++) x.push_back(static_cast<float>(100.0 + gauss(i * 31 + c) * 5.0));
    }
    const DistanceSums ref = naiveSums(x, p, g);
    std::vector<float> work = x;
    expectSums(distanceSumsRows(work.data(), n, p, g.data(), 4), ref, 1e-5);
}

TEST(Dcor, EstimatorMatchesEquationFour) {
    for (int64_t p : {1, 3}) {
        std::vector<float> x;
        std::vector<uint8_t> g;
        for (int i = 0; i < 40; i++) {
            g.push_back(static_cast<uint8_t>(i % 2));
            for (int64_t c = 0; c < p; c++)
                x.push_back(static_cast<float>(gauss(i * 5 + c) * (1.0 + 0.8 * (i % 2))));
        }
        EXPECT_NEAR(dcorSquared(naiveSums(x, p, g)), naiveDcor(x, p, g), 1e-10) << "p=" << p;
    }
}

TEST(Dcor, FalsePositiveRateUnderIndependence) {
    // z behaves like |N(0,1)| when the class says nothing: P(z > 1.96) ≈ 5 %
    // (at most, for vectors).
    std::vector<std::pair<float, uint8_t>> scratch;
    int hits1 = 0, hitsv = 0;
    const int reps = 1500, n = 200, p = 6;
    for (int r = 0; r < reps; r++) {
        std::vector<float> x1, xv;
        std::vector<uint8_t> g;
        for (int i = 0; i < n; i++) {
            g.push_back(static_cast<uint8_t>(u01(r * 7919ULL + i) < 0.5));
            x1.push_back(static_cast<float>(gauss(r * 100003ULL + i)));
            for (int c = 0; c < p; c++) xv.push_back(static_cast<float>(gauss(r * 777777ULL + i * 13 + c)));
        }
        if (dcorZ(distanceSums1D(x1.data(), g.data(), n, scratch)) > 1.96f) hits1++;
        if (dcorZ(distanceSumsRows(xv.data(), n, p, g.data(), 2)) > 1.96f) hitsv++;
    }
    const double r1 = static_cast<double>(hits1) / reps, rv = static_cast<double>(hitsv) / reps;
    EXPECT_GT(r1, 0.03) << "univariate rate " << r1;
    EXPECT_LT(r1, 0.075) << "univariate rate " << r1;
    EXPECT_LT(rv, 0.075) << "multivariate rate " << rv;
}

TEST(Dcor, FindsADifferenceInVarianceOnly) {
    // Same mean, twice the standard deviation in class 1: the 1st-order
    // t-test is blind to this, distance correlation is not.
    std::vector<float> x;
    std::vector<uint8_t> g;
    for (int i = 0; i < 2000; i++) {
        g.push_back(static_cast<uint8_t>(i & 1));
        x.push_back(static_cast<float>(gauss(i) * ((i & 1) ? 2.0 : 1.0)));
    }
    std::vector<std::pair<float, uint8_t>> scratch;
    EXPECT_GT(dcorZ(distanceSums1D(x.data(), g.data(), 2000, scratch)), 8.0f);
    EXPECT_NEAR(dcorPValue(4.5), 6.8e-6, 0.1e-6);
}

TEST(Dcor, WindowFindsMaskedSharesAtDifferentTimes) {
    // Point 0 carries a mask bit m, point 1 carries m XOR v (v = class): each
    // alone is independent of v, together they are not.
    const int n = 3000;
    std::vector<float> x, p0, p1;
    std::vector<uint8_t> g;
    for (int i = 0; i < n; i++) {
        const int v = static_cast<int>(u01(i * 3 + 1) * 2), m = static_cast<int>(u01(i * 3 + 2) * 2);
        g.push_back(static_cast<uint8_t>(v));
        p0.push_back(static_cast<float>(m + 0.3 * gauss(i * 2)));
        p1.push_back(static_cast<float>((m ^ v) + 0.3 * gauss(i * 2 + 1)));
        x.push_back(p0.back());
        x.push_back(p1.back());
    }
    std::vector<std::pair<float, uint8_t>> scratch;
    EXPECT_LT(dcorZ(distanceSums1D(p0.data(), g.data(), n, scratch)), 4.0f);
    EXPECT_LT(dcorZ(distanceSums1D(p1.data(), g.data(), n, scratch)), 4.0f);
    EXPECT_GT(dcorZ(distanceSumsRows(x.data(), n, 2, g.data(), 2)), 10.0f);
}

// ---------------------------------------------------------------------------
// Runners over trace files (ttest_dcor.h)
// ---------------------------------------------------------------------------

#include "processing.h"
#include "ttest_dcor.h"

#include <filesystem>
#include <fstream>

namespace {

// 64 samples; class v = data byte. Sample 20: class 1 has twice the spread
// (same mean). Samples 40 and 50: masked shares m and m XOR v.
struct Fixture {
    static constexpr int kNS = 64;
    std::vector<float>   x;
    std::vector<uint8_t> d;
    TrsFile file;
    explicit Fixture(int nt) {
        for (int t = 0; t < nt; t++) {
            const int v = static_cast<int>(u01(t * 5 + 1) * 2), m = static_cast<int>(u01(t * 5 + 2) * 2);
            d.push_back(static_cast<uint8_t>(v));
            for (int s = 0; s < kNS; s++) {
                double val = gauss(static_cast<uint64_t>(t) * 1000 + s) * (s == 20 && v ? 2.0 : 1.0);
                if (s == 40) val = m + 0.3 * gauss(t * 7 + 1);
                if (s == 50) val = (m ^ v) + 0.3 * gauss(t * 7 + 2);
                x.push_back(static_cast<float>(val));
            }
        }
        file.openFromArray(x.data(), nt, kNS, "dcor", d.data(), 1);
    }
};

BivariateWindow rangeOf(int64_t start, int64_t len, const QString& name = "w") {
    BivariateWindow w;
    w.name = name; w.raw_start = start; w.raw_len = len;
    return w;
}

}  // namespace

TEST(DcorRun, CurveFindsTheVarianceLeakOnly) {
    Fixture f(2000);
    DcorCurveResult r;
    QString warn, err;
    ASSERT_TRUE(computeDcorCurve(&f.file, {}, 0, 2000, rangeOf(0, Fixture::kNS), 0, 0, Fixture::kNS,
                                 false, {}, r, warn, err)) << err.toStdString();
    ASSERT_EQ(r.z.size(), static_cast<size_t>(Fixture::kNS));
    EXPECT_EQ(r.n0 + r.n1, 2000);
    EXPECT_GT(r.z[20], 8.0f);
    for (int s = 0; s < Fixture::kNS; s++)
        if (s != 20) EXPECT_LT(r.z[s], 4.5f) << s;
    // The same value as computing that column directly.
    std::vector<float> col;
    for (int t = 0; t < 2000; t++) col.push_back(f.x[static_cast<size_t>(t) * Fixture::kNS + 20]);
    std::vector<std::pair<float, uint8_t>> scratch;
    EXPECT_FLOAT_EQ(r.z[20], dcorZ(distanceSums1D(col.data(), f.d.data(), 2000, scratch)));
}

TEST(DcorRun, BlocksGiveTheSameCurve) {
    Fixture f(300);
    QString warn, err;
    for (bool resample : {false, true}) {
        std::vector<std::shared_ptr<ITransform>> pl;
        if (resample) pl.push_back(std::make_shared<WindowResampleTransform>(4));
        DcorCurveResult whole, blocks;
        ASSERT_TRUE(computeDcorCurve(&f.file, pl, 0, 300, rangeOf(0, Fixture::kNS), 0, 0, Fixture::kNS,
                                     false, {}, whole, warn, err)) << err.toStdString();
        // 300 traces × 5 samples per block.
        ASSERT_TRUE(computeDcorCurve(&f.file, pl, 0, 300, rangeOf(0, Fixture::kNS), 0, 0, Fixture::kNS,
                                     false, {}, blocks, warn, err, 300 * 5 * 4)) << err.toStdString();
        ASSERT_EQ(whole.z.size(), blocks.z.size());
        for (size_t i = 0; i < whole.z.size(); i++)
            EXPECT_FLOAT_EQ(whole.z[i], blocks.z[i]) << "resample " << resample << " sample " << i;
    }
}

TEST(DcorRun, EightBitSamplesUseTheHistogramPath) {
    // The same integer data as an int8 TRS file and as float32 in memory.
    const int nt = 400, ns = 16;
    std::vector<int8_t>  raw;
    std::vector<float>   asf;
    std::vector<uint8_t> d;
    for (int t = 0; t < nt; t++) {
        d.push_back(static_cast<uint8_t>(t & 1));
        for (int s = 0; s < ns; s++) {
            const int v = std::clamp(static_cast<int>(std::lround(gauss(t * 100 + s) * 20.0 * (s == 3 && (t & 1) ? 2.5 : 1.0))), -128, 127);
            raw.push_back(static_cast<int8_t>(v));
            asf.push_back(static_cast<float>(v));
        }
    }
    const std::string path = (std::filesystem::temp_directory_path() / "trs_viewer_test_dcor_int8.trs").string();
    {
        std::ofstream o(path, std::ios::binary);
        auto u8 = [&](int v) { const char c = static_cast<char>(v); o.write(&c, 1); };
        u8(0x41); u8(4); u8(nt & 0xFF); u8(nt >> 8); u8(0); u8(0);
        u8(0x42); u8(4); u8(ns); u8(0); u8(0); u8(0);
        u8(0x43); u8(1); u8(0x01);
        u8(0x44); u8(2); u8(1); u8(0);
        u8(0x5F); u8(0);
        for (int t = 0; t < nt; t++) {
            u8(d[static_cast<size_t>(t)]);
            o.write(reinterpret_cast<const char*>(raw.data() + static_cast<size_t>(t) * ns), ns);
        }
    }
    TrsFile i8, f32;
    std::string e;
    ASSERT_TRUE(i8.open(path, e)) << e;
    f32.openFromArray(asf.data(), nt, ns, "f32", d.data(), 1);
    DcorCurveResult a, b;
    QString warn, err;
    ASSERT_TRUE(computeDcorCurve(&i8, {}, 0, nt, rangeOf(0, ns), 0, 0, ns, false, {}, a, warn, err));
    ASSERT_TRUE(computeDcorCurve(&f32, {}, 0, nt, rangeOf(0, ns), 0, 0, ns, false, {}, b, warn, err));
    for (int s = 0; s < ns; s++) EXPECT_NEAR(a.z[s], b.z[s], 1e-4f) << s;
    EXPECT_GT(a.z[3], 4.5f);
    std::filesystem::remove(path);
}

TEST(DcorRun, WindowsFindTheMaskedPair) {
    // Second-order leakage fades as a window grows (z ~ 1/width), so the
    // window around the two shares is kept short: 11 samples, 9 of them noise.
    Fixture f(6000);
    const std::vector<BivariateWindow> w = {
        rangeOf(40, 11, "both shares"),   // 40..50
        rangeOf(35, 10, "one share"),     // 35..44: sample 40 only
        rangeOf(0, 15, "quiet"),
    };
    DcorWindowResult r;
    QString warn, err;
    ASSERT_TRUE(computeWindowDcor(&f.file, {}, 0, 6000, w, 0, 0, Fixture::kNS, false, true, {},
                                  r, warn, err)) << err.toStdString();
    ASSERT_EQ(r.z.size(), 3u);
    EXPECT_GT(r.z[0], 4.5f);
    EXPECT_LT(r.z[1], 4.5f);
    EXPECT_LT(r.z[2], 4.5f);
    EXPECT_EQ(r.n0[0] + r.n1[0], 6000);
    // Without unit variance the low-variance shares count for less.
    DcorWindowResult raw;
    ASSERT_TRUE(computeWindowDcor(&f.file, {}, 0, 6000, {w[0]}, 0, 0, Fixture::kNS, false, false, {},
                                  raw, warn, err)) << err.toStdString();
    EXPECT_LT(raw.z[0], r.z[0]);
}

TEST(DcorRun, SanitySplitFindsNothing) {
    Fixture f(2000);
    DcorCurveResult r;
    QString warn, err;
    ASSERT_TRUE(computeDcorCurve(&f.file, {}, 0, 2000, rangeOf(0, Fixture::kNS), 0, 0, Fixture::kNS,
                                 true, {}, r, warn, err)) << err.toStdString();
    EXPECT_LT(*std::max_element(r.z.begin(), r.z.end()), 4.5f);
}

TEST(DcorRun, SplitsOnOneBitLikeOnAFlag) {
    // The same traces with the class in bit 2 of a byte that is never 0.
    Fixture flag(600);
    std::vector<uint8_t> in_bit(flag.d.size());
    for (size_t t = 0; t < flag.d.size(); t++)
        in_bit[t] = static_cast<uint8_t>(0x91 | (flag.d[t] << 2));
    TrsFile bit_file;
    bit_file.openFromArray(flag.x.data(), 600, Fixture::kNS, "bit", in_bit.data(), 1);

    DcorCurveResult a, b;
    QString warn, err;
    ASSERT_TRUE(computeDcorCurve(&flag.file, {}, 0, 600, rangeOf(0, Fixture::kNS), 0, 0, Fixture::kNS,
                                 false, {}, a, warn, err)) << err.toStdString();
    ASSERT_TRUE(computeDcorCurve(&bit_file, {}, 0, 600, rangeOf(0, Fixture::kNS), GroupBy(0, 0x04), 0,
                                 Fixture::kNS, false, {}, b, warn, err)) << err.toStdString();
    EXPECT_EQ(a.n0, b.n0);
    EXPECT_EQ(a.n1, b.n1);
    EXPECT_EQ(a.z, b.z);
}
