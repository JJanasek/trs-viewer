#include <gtest/gtest.h>
#include "pattern_align.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kNT = 60;
constexpr int kNS = 4000;

double feature(double u, int seed) {
    // A few smooth bumps (visible after resampling) plus fine texture
    // (needed to pin the shift down to one raw sample).
    double v = 0;
    for (int k = 0; k < 6; k++) {
        const double c = -150 + 60 * k + 7 * std::sin(seed + k);
        const double a = 0.6 + 0.4 * std::sin(3.1 * k + seed);
        v += a * std::exp(-(u - c) * (u - c) / (2 * 12.0 * 12.0));
    }
    if (std::fabs(u) < 190) v += 0.25 * std::sin(0.9 * u + seed) * std::cos(0.13 * u);
    return v;
}

double noise(int t, int s) {
    const double x = std::sin((t * 131 + s * 17) * 12.9898) * 43758.5453;
    return (x - std::floor(x) - 0.5) * 0.02;
}

int shiftA(int t) { return ((t * 37) % 81) - 40; }
int shiftB(int t) { return ((t * 53) % 61) - 30; }

struct TwoPatternSet {
    std::vector<float> m;
    TrsFile f;
    TwoPatternSet() : m(static_cast<size_t>(kNT) * kNS) {
        for (int t = 0; t < kNT; t++)
            for (int s = 0; s < kNS; s++)
                m[static_cast<size_t>(t) * kNS + s] = static_cast<float>(
                    feature(s - (1700 + shiftA(t)), 1) + feature(s - (3100 + shiftB(t)), 2) + noise(t, s));
        f.openFromArray(m.data(), kNT, kNS, "patterns");
    }
};

std::vector<std::shared_ptr<ITransform>> resample16() {
    return {std::make_shared<WindowResampleTransform>(16, 0.0f)};
}

PatternDef defAround(int64_t centre, const char* name) {
    PatternDef d;
    d.name = name;
    d.raw_start = centre - 260;
    d.raw_end   = centre + 260;
    d.search_half = 64;
    return d;
}

}  // namespace

TEST(PatternAlign, RawStartForProcessedIndex) {
    EXPECT_EQ(rawStartForProcessedIndex({}, 37, 10000), 37);
    EXPECT_EQ(rawStartForProcessedIndex(resample16(), 37, 10000), 37 * 16);
    std::vector<std::shared_ptr<ITransform>> half = {std::make_shared<WindowResampleTransform>(16, 0.5f)};
    EXPECT_EQ(rawStartForProcessedIndex(half, 37, 10000), 37 * 8);
    std::vector<std::shared_ptr<ITransform>> stride = {std::make_shared<StrideResampleTransform>(4)};
    EXPECT_EQ(rawStartForProcessedIndex(stride, 37, 10000), 37 * 4);
    std::vector<std::shared_ptr<ITransform>> ma = {std::make_shared<MovingAverageTransform>(8)};
    EXPECT_EQ(rawStartForProcessedIndex(ma, 37, 10000), 37);
}

TEST(PatternAlign, RefinementRecoversExactShiftsThroughResampling) {
    TwoPatternSet ts;
    PatternResult r;
    std::string err;
    ASSERT_TRUE(alignPattern(&ts.f, resample16(), 0, kNT, defAround(1700, "A"), r, {}, err)) << err;
    ASSERT_EQ(r.shifts.size(), static_cast<size_t>(kNT));
    for (int t = 0; t < kNT; t++)
        EXPECT_EQ(r.shifts[t], shiftA(t) - shiftA(0)) << "trace " << t;
    EXPECT_GT(r.median_score, 0.95f);
    EXPECT_EQ(r.discarded, 0);
}

TEST(PatternAlign, WithoutRefinementShiftsAreOnlyStepAccurate) {
    TwoPatternSet ts;
    PatternDef d = defAround(1700, "A");
    d.refine_raw = false;
    PatternResult r;
    std::string err;
    ASSERT_TRUE(alignPattern(&ts.f, resample16(), 0, kNT, d, r, {}, err)) << err;
    int off = 0;
    for (int t = 0; t < kNT; t++) {
        const int e = std::abs(r.shifts[t] - (shiftA(t) - shiftA(0)));
        EXPECT_LE(e, 16) << "trace " << t;
        if (e > 0) off++;
    }
    EXPECT_GT(off, 0);   // proves the refinement test above is meaningful
}

TEST(PatternAlign, PatternsAlignIndependently) {
    TwoPatternSet ts;
    PatternResult ra, rb;
    std::string err;
    ASSERT_TRUE(alignPattern(&ts.f, resample16(), 0, kNT, defAround(1700, "A"), ra, {}, err)) << err;
    ASSERT_TRUE(alignPattern(&ts.f, resample16(), 0, kNT, defAround(3100, "B"), rb, {}, err)) << err;
    for (int t = 0; t < kNT; t++) {
        EXPECT_EQ(ra.shifts[t], shiftA(t) - shiftA(0));
        EXPECT_EQ(rb.shifts[t], shiftB(t) - shiftB(0));
    }
}

TEST(PatternAlign, NoPipelineIsExactWithoutRefinement) {
    TwoPatternSet ts;
    PatternResult r;
    std::string err;
    ASSERT_TRUE(alignPattern(&ts.f, {}, 0, kNT, defAround(3100, "B"), r, {}, err)) << err;
    for (int t = 0; t < kNT; t++) EXPECT_EQ(r.shifts[t], shiftB(t) - shiftB(0));
}

TEST(PatternAlign, MinCorrelationDiscards) {
    TwoPatternSet ts;
    // Trace 5 gets garbage in pattern A's region.
    for (int s = 1300; s < 2100; s++)
        ts.m[static_cast<size_t>(5) * kNS + s] = static_cast<float>(std::sin(s * 2.7) * 3.0);
    ts.f.openFromArray(ts.m.data(), kNT, kNS, "patterns");   // the file keeps its own copy
    PatternDef d = defAround(1700, "A");
    d.min_corr = 0.8f;
    PatternResult r;
    std::string err;
    ASSERT_TRUE(alignPattern(&ts.f, resample16(), 0, kNT, d, r, {}, err)) << err;
    EXPECT_EQ(r.shifts[5], kAlignDiscardShift);
    EXPECT_EQ(r.discarded, 1);
}

TEST(PatternAlign, ShiftFileRoundTrip) {
    std::vector<Pattern> ps(2);
    ps[0].def.name = "key load";  ps[0].def.raw_start = 100; ps[0].def.raw_end = 900;
    ps[0].res.first_trace = 10;   ps[0].res.shifts = {3, kAlignDiscardShift, -7};
    ps[1].def.name = "round 1";   ps[1].def.raw_start = 2000; ps[1].def.raw_end = 2600;
    ps[1].res.first_trace = 10;   ps[1].res.shifts = {0, 12, -1};
    const std::string path = ::testing::TempDir() + "pattern_shifts.csv";
    std::string err;
    ASSERT_TRUE(savePatternShifts(path, ps, err)) << err;
    std::vector<Pattern> back;
    ASSERT_TRUE(loadPatternShifts(path, 5000, back, err)) << err;
    ASSERT_EQ(back.size(), 2u);
    EXPECT_EQ(back[0].def.name, "key load");
    EXPECT_EQ(back[0].def.raw_start, 100);
    EXPECT_EQ(back[1].def.raw_end, 2600);
    EXPECT_EQ(back[0].res.first_trace, 10);
    EXPECT_EQ(back[0].res.shifts, (std::vector<int32_t>{3, kAlignDiscardShift, -7}));
    EXPECT_EQ(back[1].res.shifts, (std::vector<int32_t>{0, 12, -1}));
    std::remove(path.c_str());
}

TEST(PatternAlign, LegacyShiftFileLoadsAsOnePattern) {
    const std::string path = ::testing::TempDir() + "legacy_shifts.csv";
    {
        FILE* fp = std::fopen(path.c_str(), "w");
        std::fputs("# trs-viewer alignment shifts (trace,offset)\n0,4\n1,-2\n3,9\n", fp);
        std::fclose(fp);
    }
    std::vector<Pattern> back;
    std::string err;
    ASSERT_TRUE(loadPatternShifts(path, 5000, back, err)) << err;
    ASSERT_EQ(back.size(), 1u);
    EXPECT_EQ(back[0].def.raw_end, 5000);
    EXPECT_EQ(back[0].res.shifts, (std::vector<int32_t>{4, -2, kAlignDiscardShift, 9}));
    std::remove(path.c_str());
}
