#include <gtest/gtest.h>
#include "spectral.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// A flat noise floor with a deterministic ripple, so the baseline has
// something to track but no bin is ever a "line" on its own.
static std::vector<float> makeFloor(int64_t n, float level = 1.0f) {
    std::vector<float> s(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        s[static_cast<size_t>(i)] =
            level * (1.0f + 0.05f * std::sin(static_cast<float>(i) * 0.037f));
    return s;
}

// Plants a narrow line of `half_width` bins either side of `center`, scaling
// the floor there by `gain` with a triangular profile so it has skirts.
static void addLine(std::vector<float>& s, int64_t center, int64_t half_width, float gain) {
    for (int64_t d = -half_width; d <= half_width; ++d) {
        int64_t i = center + d;
        if (i < 0 || i >= static_cast<int64_t>(s.size())) continue;
        const float taper = 1.0f - static_cast<float>(std::abs(d)) /
                                   static_cast<float>(half_width + 1);
        s[static_cast<size_t>(i)] *= 1.0f + (gain - 1.0f) * taper;
    }
}

static LineDetectParams defaultParams() {
    LineDetectParams p;
    p.baseline_window = 257;
    p.min_excess_db   = 6.0f;
    p.max_width_bins  = 24;
    p.min_bin         = 2;
    p.input_is_db     = false;
    return p;
}

// ---------------------------------------------------------------------------
// runningMedianBaseline
// ---------------------------------------------------------------------------

TEST(RunningMedianBaseline, FlatSpectrumGivesFlatBaseline) {
    std::vector<float> s(2000, 3.5f);
    std::vector<float> base;
    runningMedianBaseline(s, 129, base);
    ASSERT_EQ(base.size(), s.size());
    for (float v : base) EXPECT_NEAR(v, 3.5f, 1e-5f);
}

// The property the whole detector rests on: a few-bin spike must not pull its
// own baseline up with it, or it would hide itself.
TEST(RunningMedianBaseline, NarrowSpikeDoesNotMoveBaseline) {
    std::vector<float> s(4000, 1.0f);
    addLine(s, 2000, 3, 100.0f);
    std::vector<float> base;
    runningMedianBaseline(s, 257, base);
    EXPECT_NEAR(base[2000], 1.0f, 1e-3f);
}

// ...whereas a feature wide enough to be real signal *should* move it, which
// is exactly why such features never show up as lines.
TEST(RunningMedianBaseline, BroadBumpDoesMoveBaseline) {
    std::vector<float> s(4000, 1.0f);
    for (int64_t i = 1500; i < 2500; ++i) s[static_cast<size_t>(i)] = 8.0f;
    std::vector<float> base;
    runningMedianBaseline(s, 257, base);
    EXPECT_NEAR(base[2000], 8.0f, 1e-3f);
}

TEST(RunningMedianBaseline, HandlesWindowLargerThanSpectrum) {
    std::vector<float> s(11, 2.0f);
    std::vector<float> base;
    runningMedianBaseline(s, 999, base);   // must not read out of bounds
    ASSERT_EQ(base.size(), s.size());
    for (float v : base) EXPECT_NEAR(v, 2.0f, 1e-5f);
}

TEST(RunningMedianBaseline, EmptyInput) {
    std::vector<float> s;
    std::vector<float> base;
    runningMedianBaseline(s, 129, base);
    EXPECT_TRUE(base.empty());
}

// ---------------------------------------------------------------------------
// detectSpectralLines — basic detection
// ---------------------------------------------------------------------------

TEST(DetectSpectralLines, FlatFloorHasNoLines) {
    auto s = makeFloor(4000);
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    EXPECT_TRUE(lines.empty());
}

TEST(DetectSpectralLines, FindsSingleNarrowLine) {
    auto s = makeFloor(4000);
    addLine(s, 1234, 2, 10.0f);          // 10x over the floor => ~20 dB
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].bin, 1234);
    EXPECT_NEAR(lines[0].excess_db, 20.0f, 1.0f);
    EXPECT_NEAR(lines[0].baseline, 1.0f, 0.1f);
    EXPECT_LE(lines[0].widthBins(), 24);
}

// The most important negative case. A broad feature is what leakage looks
// like; reporting it as interference would invite the user to filter away the
// very thing they are hunting.
TEST(DetectSpectralLines, RejectsBroadbandFeature) {
    auto s = makeFloor(8000);
    for (int64_t i = 3000; i < 3600; ++i) s[static_cast<size_t>(i)] *= 12.0f;
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    for (const auto& ln : lines)
        EXPECT_FALSE(ln.bin >= 3000 && ln.bin < 3600)
            << "broadband feature reported as a line at bin " << ln.bin;
}

TEST(DetectSpectralLines, WidthLimitIsHonoured) {
    auto s = makeFloor(8000);
    addLine(s, 4000, 40, 10.0f);         // ~81 bins wide, well over the limit
    LineDetectParams p = defaultParams();
    p.max_width_bins = 24;
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, p, lines, nullptr, err)) << err;
    EXPECT_TRUE(lines.empty());

    p.max_width_bins = 200;              // same data, permissive limit
    ASSERT_TRUE(detectSpectralLines(s, p, lines, nullptr, err)) << err;
    EXPECT_EQ(lines.size(), 1u);
}

// ---------------------------------------------------------------------------
// detectSpectralLines — a harmonic comb, the motivating case
// ---------------------------------------------------------------------------

TEST(DetectSpectralLines, FindsHarmonicComb) {
    auto s = makeFloor(16000);
    const int64_t f0 = 1000;
    for (int k = 1; k <= 5; ++k)
        addLine(s, f0 * k, 2, 20.0f / static_cast<float>(k));  // decaying harmonics

    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    ASSERT_GE(lines.size(), 4u);

    std::vector<int64_t> bins;
    for (const auto& ln : lines) bins.push_back(ln.bin);
    std::sort(bins.begin(), bins.end());
    for (int k = 1; k <= 4; ++k) {
        const int64_t want = f0 * k;
        EXPECT_NE(std::find(bins.begin(), bins.end(), want), bins.end())
            << "missed harmonic " << k << " at bin " << want;
    }
}

TEST(DetectSpectralLines, SortedByDescendingExcess) {
    auto s = makeFloor(8000);
    addLine(s, 1000, 2,  4.0f);
    addLine(s, 3000, 2, 30.0f);
    addLine(s, 5000, 2, 10.0f);
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0].bin, 3000);       // strongest first
    EXPECT_EQ(lines[2].bin, 1000);       // weakest last
    for (size_t i = 1; i < lines.size(); ++i)
        EXPECT_GE(lines[i - 1].excess_db, lines[i].excess_db);
}

// One carrier must come back as one line, not one per bin above threshold.
TEST(DetectSpectralLines, ContiguousRunIsOneLine) {
    auto s = makeFloor(4000);
    addLine(s, 2000, 5, 50.0f);
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    EXPECT_EQ(lines.size(), 1u);
}

// ---------------------------------------------------------------------------
// detectSpectralLines — parameters and edges
// ---------------------------------------------------------------------------

TEST(DetectSpectralLines, MinBinSkipsDcRegion) {
    auto s = makeFloor(4000);
    s[0] = 500.0f; s[1] = 300.0f;        // DC offset / drift, not a line
    LineDetectParams p = defaultParams();
    p.min_bin = 4;
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, p, lines, nullptr, err)) << err;
    for (const auto& ln : lines) EXPECT_GE(ln.bin, 4);
}

TEST(DetectSpectralLines, ThresholdGatesWeakLines) {
    auto s = makeFloor(4000);
    addLine(s, 2000, 2, 1.5f);           // ~3.5 dB — below a 6 dB threshold
    LineDetectParams p = defaultParams();
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, p, lines, nullptr, err)) << err;
    EXPECT_TRUE(lines.empty());

    p.min_excess_db = 2.0f;
    ASSERT_TRUE(detectSpectralLines(s, p, lines, nullptr, err)) << err;
    EXPECT_EQ(lines.size(), 1u);
}

// A dB-valued spectrum measures excess by subtraction, so the same physical
// line must be reported with the same excess either way.
TEST(DetectSpectralLines, DbInputMatchesLinearInput) {
    auto lin = makeFloor(4000);
    addLine(lin, 2500, 2, 10.0f);

    std::vector<float> db(lin.size());
    for (size_t i = 0; i < lin.size(); ++i)
        db[i] = 20.0f * std::log10(std::max(lin[i], 1e-20f));

    std::vector<SpectralLine> a, b;
    std::string err;
    LineDetectParams pl = defaultParams();
    LineDetectParams pd = defaultParams();
    pd.input_is_db = true;

    ASSERT_TRUE(detectSpectralLines(lin, pl, a, nullptr, err)) << err;
    ASSERT_TRUE(detectSpectralLines(db,  pd, b, nullptr, err)) << err;
    ASSERT_EQ(a.size(), 1u);
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(a[0].bin, b[0].bin);
    EXPECT_NEAR(a[0].excess_db, b[0].excess_db, 0.5f);
}

TEST(DetectSpectralLines, ExtentCoversLineSkirts) {
    auto s = makeFloor(4000);
    addLine(s, 2000, 6, 40.0f);
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    ASSERT_EQ(lines.size(), 1u);
    // lo/hi expand to half-excess, so they must bracket the peak strictly
    // wider than a single bin and stay inside the planted line.
    EXPECT_LT(lines[0].lo_bin, lines[0].bin);
    EXPECT_GT(lines[0].hi_bin, lines[0].bin);
    EXPECT_GE(lines[0].lo_bin, 2000 - 12);
    EXPECT_LE(lines[0].hi_bin, 2000 + 12);
}

TEST(DetectSpectralLines, BaselineOutIsPopulated) {
    auto s = makeFloor(4000);
    std::vector<SpectralLine> lines;
    std::vector<float> base;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, &base, err)) << err;
    ASSERT_EQ(base.size(), s.size());
    for (float v : base) EXPECT_NEAR(v, 1.0f, 0.2f);
}

TEST(DetectSpectralLines, RejectsTooShortSpectrum) {
    std::vector<float> s(4, 1.0f);
    std::vector<SpectralLine> lines;
    std::string err;
    EXPECT_FALSE(detectSpectralLines(s, defaultParams(), lines, nullptr, err));
    EXPECT_FALSE(err.empty());
}

TEST(DetectSpectralLines, LineAtSpectrumEdgeDoesNotOverrun) {
    auto s = makeFloor(4000);
    addLine(s, 3999, 2, 20.0f);          // hard against the Nyquist end
    std::vector<SpectralLine> lines;
    std::string err;
    ASSERT_TRUE(detectSpectralLines(s, defaultParams(), lines, nullptr, err)) << err;
    for (const auto& ln : lines) {
        EXPECT_GE(ln.lo_bin, 0);
        EXPECT_LT(ln.hi_bin, 4000);
    }
}

// ---------------------------------------------------------------------------
// Unit conversion helpers
// ---------------------------------------------------------------------------

TEST(SpectralUnits, BinToHz) {
    // 1 GS/s, 1024-point FFT => 976562.5 Hz per bin
    EXPECT_NEAR(binToHz(1, 1024, 1e9), 1e9 / 1024.0, 1e-6);
    EXPECT_NEAR(binToHz(512, 1024, 1e9), 5e8, 1e-3);     // Nyquist
    EXPECT_EQ(binToHz(10, 1024, 0.0), 0.0);              // unknown rate
    EXPECT_EQ(binToHz(10, 0, 1e9), 0.0);
}

TEST(SpectralUnits, HzToNormalized) {
    EXPECT_NEAR(hzToNormalized(5e8, 1e9), 1.0, 1e-9);    // Nyquist -> 1.0
    EXPECT_NEAR(hzToNormalized(2.5e8, 1e9), 0.5, 1e-9);
    EXPECT_EQ(hzToNormalized(1e6, 0.0), 0.0);
}

// ===========================================================================
// SpectralExciseTransform
// ===========================================================================

#include "processing.h"

namespace {

// Amplitude of the component at `cycles_per_sample` in x, by direct
// single-bin DFT. Independent of the transform's own FFT, so a bug in one
// can't hide a bug in the other.
double toneAmplitude(const std::vector<float>& x, double cycles_per_sample) {
    double re = 0.0, im = 0.0;
    const int64_t n = static_cast<int64_t>(x.size());
    for (int64_t i = 0; i < n; ++i) {
        const double ph = 2.0 * M_PI * cycles_per_sample * static_cast<double>(i);
        re += static_cast<double>(x[static_cast<size_t>(i)]) * std::cos(ph);
        im -= static_cast<double>(x[static_cast<size_t>(i)]) * std::sin(ph);
    }
    return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(n);
}

// Normalised frequency (fraction of Nyquist) of a cycles-per-sample rate.
constexpr double kNorm = 2.0;

std::vector<float> runExcise(SpectralExciseTransform& tx, std::vector<float> x) {
    tx.reset();
    const int64_t n = tx.apply(x.data(), static_cast<int64_t>(x.size()), 0);
    x.resize(static_cast<size_t>(std::max<int64_t>(0, n)));
    return x;
}

// 4096 samples so both test tones land exactly on FFT bins and leakage
// between them never muddies the measurement.
constexpr int64_t kN = 4096;
constexpr double  kFsignal = 80.0   / 4096.0;   // the "leakage" we must keep
constexpr double  kFinterf = 1024.0 / 4096.0;   // the carrier we must remove

std::vector<float> makeTwoTone(float sig_amp = 0.1f, float int_amp = 1.0f) {
    std::vector<float> x(static_cast<size_t>(kN));
    for (int64_t i = 0; i < kN; ++i) {
        const double t = static_cast<double>(i);
        x[static_cast<size_t>(i)] =
            sig_amp * static_cast<float>(std::sin(2.0 * M_PI * kFsignal * t)) +
            int_amp * static_cast<float>(std::sin(2.0 * M_PI * kFinterf * t));
    }
    return x;
}

} // namespace

TEST(SpectralExcise, DeclaresWholeTraceRequirement) {
    SpectralExciseTransform tx;
    EXPECT_TRUE(tx.requiresWholeTrace());
    EXPECT_TRUE(tx.requiresSequential());
    std::vector<std::shared_ptr<ITransform>> pipe{
        std::make_shared<AbsTransform>(), std::make_shared<SpectralExciseTransform>()};
    EXPECT_TRUE(pipelineRequiresWholeTrace(pipe));
    std::vector<std::shared_ptr<ITransform>> plain{std::make_shared<AbsTransform>()};
    EXPECT_FALSE(pipelineRequiresWholeTrace(plain));
}

TEST(SpectralExcise, NoBandsIsIdentity) {
    SpectralExciseTransform tx;                       // empty band list
    auto in  = makeTwoTone();
    auto out = runExcise(tx, in);
    ASSERT_EQ(out.size(), in.size());
    for (size_t i = 0; i < in.size(); ++i)
        EXPECT_FLOAT_EQ(out[i], in[i]);               // untouched, not merely close
}

// The headline behaviour: kill the carrier, keep the leakage.
TEST(SpectralExcise, RemovesInterfererAndKeepsSignal) {
    auto in = makeTwoTone(0.1f, 1.0f);
    const double amp_sig_before = toneAmplitude(in, kFsignal);
    const double amp_int_before = toneAmplitude(in, kFinterf);
    EXPECT_NEAR(amp_sig_before, 0.1, 0.005);
    EXPECT_NEAR(amp_int_before, 1.0, 0.005);

    SpectralExciseTransform tx({SpectralBand{kFinterf * kNorm, 4.0 / 2048.0}});
    auto out = runExcise(tx, in);
    ASSERT_EQ(out.size(), in.size());

    const double amp_sig_after = toneAmplitude(out, kFsignal);
    const double amp_int_after = toneAmplitude(out, kFinterf);

    // Interferer knocked down by at least 40 dB...
    EXPECT_LT(amp_int_after, amp_int_before / 100.0)
        << "interferer only fell from " << amp_int_before << " to " << amp_int_after;
    // ...while the signal 12 bins away is untouched to within 1%.
    EXPECT_NEAR(amp_sig_after, amp_sig_before, amp_sig_before * 0.01);
}

TEST(SpectralExcise, RandomizePhaseAlsoRemovesInterferer) {
    auto in = makeTwoTone();
    SpectralExciseTransform tx({SpectralBand{kFinterf * kNorm, 4.0 / 2048.0}},
                               SpectralExciseTransform::PhaseMode::Randomize);
    auto out = runExcise(tx, in);
    EXPECT_LT(toneAmplitude(out, kFinterf), 0.05);
    EXPECT_NEAR(toneAmplitude(out, kFsignal), 0.1, 0.005);
}

// Excision must not move the trace vertically — bin 0 is the DC offset, not
// interference, and shifting it would desynchronise a differential analysis
// from every trace that wasn't excised.
TEST(SpectralExcise, PreservesDcOffset) {
    auto in = makeTwoTone();
    for (auto& v : in) v += 7.5f;
    double mean_before = 0.0;
    for (float v : in) mean_before += v;
    mean_before /= static_cast<double>(in.size());

    // A band deliberately straddling DC, to prove bin 0 is protected.
    SpectralExciseTransform tx({SpectralBand{0.0, 8.0 / 2048.0},
                                SpectralBand{kFinterf * kNorm, 4.0 / 2048.0}});
    auto out = runExcise(tx, in);
    double mean_after = 0.0;
    for (float v : out) mean_after += v;
    mean_after /= static_cast<double>(out.size());
    EXPECT_NEAR(mean_after, mean_before, std::abs(mean_before) * 1e-4 + 1e-4);
}

TEST(SpectralExcise, RemovesAWholeHarmonicComb) {
    // Fundamental at bin 256 with harmonics at 512, 768, 1024, 1280.
    const double f0 = 256.0 / 4096.0;
    std::vector<float> x(static_cast<size_t>(kN), 0.0f);
    for (int64_t i = 0; i < kN; ++i) {
        double acc = 0.2 * std::sin(2.0 * M_PI * kFsignal * static_cast<double>(i));
        for (int k = 1; k <= 5; ++k)
            acc += (1.0 / k) * std::sin(2.0 * M_PI * f0 * k * static_cast<double>(i));
        x[static_cast<size_t>(i)] = static_cast<float>(acc);
    }

    auto bands = harmonicComb(f0 * kNorm, 3.0 / 2048.0, 5, /*odd_only=*/false);
    ASSERT_EQ(bands.size(), 5u);
    SpectralExciseTransform tx(bands);
    auto out = runExcise(tx, x);

    for (int k = 1; k <= 5; ++k)
        EXPECT_LT(toneAmplitude(out, f0 * k), 0.02)
            << "harmonic " << k << " survived excision";
    EXPECT_NEAR(toneAmplitude(out, kFsignal), 0.2, 0.01);
}

TEST(SpectralExcise, OddLengthTraceStaysRealAndSameLength) {
    const int64_t n = 4095;                            // odd: no Nyquist bin
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * M_PI * 0.25 * static_cast<double>(i)));
    SpectralExciseTransform tx({SpectralBand{0.5, 8.0 / 2048.0}});
    auto out = runExcise(tx, x);
    ASSERT_EQ(out.size(), static_cast<size_t>(n));
    for (float v : out) EXPECT_TRUE(std::isfinite(v));
}

TEST(SpectralExcise, ShortInputPassesThrough) {
    SpectralExciseTransform tx({SpectralBand{0.5, 0.01}});
    std::vector<float> x{1.f, 2.f, 3.f};
    auto out = runExcise(tx, x);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_FLOAT_EQ(out[0], 1.f);
}

TEST(SpectralExcise, ReusableAcrossTracesOfTheSameLength) {
    SpectralExciseTransform tx({SpectralBand{kFinterf * kNorm, 4.0 / 2048.0}});
    // The same instance is applied to every trace of a set in the real
    // pipeline, so a stale cached FFT plan would show up here.
    for (int rep = 0; rep < 3; ++rep) {
        auto out = runExcise(tx, makeTwoTone());
        EXPECT_LT(toneAmplitude(out, kFinterf), 0.05) << "rep " << rep;
        EXPECT_NEAR(toneAmplitude(out, kFsignal), 0.1, 0.005) << "rep " << rep;
    }
}

TEST(SpectralExcise, HandlesTraceLengthChange) {
    // Eigen's FFT corrupts its plan cache across a size change unless it is
    // rebuilt; this is the regression guard for that. The band is given in
    // fractions of Nyquist, so it covers a different *number* of bins at each
    // length — hence a relative assertion (the tone must drop a lot) rather
    // than an absolute one. How much is length-dependent and is pinned down
    // by NonBinAlignedToneNeedsAWiderBand below.
    SpectralExciseTransform tx({SpectralBand{0.5, 8.0 / 2048.0}});
    for (int64_t n : {1024, 2048, 777, 4096}) {
        std::vector<float> x(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] =
                static_cast<float>(std::sin(2.0 * M_PI * 0.25 * static_cast<double>(i)));
        const double before = toneAmplitude(x, 0.25);
        auto out = runExcise(tx, x);
        ASSERT_EQ(out.size(), static_cast<size_t>(n));
        for (float v : out) ASSERT_TRUE(std::isfinite(v)) << "n=" << n;
        EXPECT_LT(toneAmplitude(out, 0.25), before * 0.25) << "n=" << n;
    }
}

// An interferer that does not land exactly on an FFT bin has its energy
// spread across neighbouring bins by the implicit rectangular window, and
// those sinc skirts decay slowly. Excising only the few bins at the peak
// therefore leaves a visible residue — the fix is a wider band, not a
// different algorithm. Worth pinning down, because it is the one case where
// excision looks like it "didn't work" while behaving exactly as designed.
TEST(SpectralExcise, NonBinAlignedToneNeedsAWiderBand) {
    const int64_t n = 4096;
    const double  f = 1024.5 / 4096.0;          // deliberately half a bin off
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * M_PI * f * static_cast<double>(i)));
    const double before = toneAmplitude(x, f);

    SpectralExciseTransform narrow({SpectralBand{f * kNorm, 2.0 / 2048.0}});
    const double after_narrow = toneAmplitude(runExcise(narrow, x), f);

    SpectralExciseTransform wide({SpectralBand{f * kNorm, 64.0 / 2048.0}});
    const double after_wide = toneAmplitude(runExcise(wide, x), f);

    EXPECT_LT(after_wide, after_narrow)
        << "a wider band must remove more of a non-bin-aligned tone";
    EXPECT_LT(after_wide, before * 0.02);
}

TEST(SpectralExcise, CloneIsIndependent) {
    SpectralExciseTransform tx({SpectralBand{0.5, 0.01}});
    auto c = tx.clone();
    auto* cp = dynamic_cast<SpectralExciseTransform*>(c.get());
    ASSERT_NE(cp, nullptr);
    ASSERT_EQ(cp->bands().size(), 1u);
    cp->setBands({});
    EXPECT_EQ(tx.bands().size(), 1u);      // original unaffected
    EXPECT_EQ(cp->bands().size(), 0u);
}

// ---------------------------------------------------------------------------
// Band encoding (chain.json persistence) and comb generation
// ---------------------------------------------------------------------------

TEST(SpectralBands, EncodeDecodeRoundTrip) {
    std::vector<SpectralBand> in{{0.125, 0.001}, {0.5, 0.0025}, {0.9876543, 1e-5}};
    const std::string enc = encodeSpectralBands(in);
    std::vector<SpectralBand> out;
    ASSERT_TRUE(decodeSpectralBands(enc, out));
    ASSERT_EQ(out.size(), in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i].center,    in[i].center,    1e-9);
        EXPECT_NEAR(out[i].halfwidth, in[i].halfwidth, 1e-12);
    }
}

TEST(SpectralBands, EmptyEncodesAndDecodesToEmpty) {
    EXPECT_EQ(encodeSpectralBands({}), "");
    std::vector<SpectralBand> out{{0.5, 0.1}};
    ASSERT_TRUE(decodeSpectralBands("", out));
    EXPECT_TRUE(out.empty());
}

TEST(SpectralBands, MalformedInputRejected) {
    std::vector<SpectralBand> out;
    EXPECT_FALSE(decodeSpectralBands("0.5", out));           // no colon
    EXPECT_TRUE(out.empty());
    EXPECT_FALSE(decodeSpectralBands("abc:def", out));       // not numbers
    EXPECT_TRUE(out.empty());
}

TEST(SpectralBands, LoHiAccessors) {
    SpectralBand b{0.5, 0.02};
    EXPECT_NEAR(b.lo(), 0.48, 1e-12);
    EXPECT_NEAR(b.hi(), 0.52, 1e-12);
}

TEST(HarmonicComb, GeneratesMultiples) {
    auto c = harmonicComb(0.1, 0.002, 4, false);
    ASSERT_EQ(c.size(), 4u);
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(c[static_cast<size_t>(k)].center, 0.1 * (k + 1), 1e-12);
        EXPECT_NEAR(c[static_cast<size_t>(k)].halfwidth, 0.002, 1e-12);
    }
}

TEST(HarmonicComb, OddOnlySkipsEvenMultiples) {
    auto c = harmonicComb(0.1, 0.002, 3, /*odd_only=*/true);
    ASSERT_EQ(c.size(), 3u);
    EXPECT_NEAR(c[0].center, 0.1, 1e-12);   // 1st
    EXPECT_NEAR(c[1].center, 0.3, 1e-12);   // 3rd
    EXPECT_NEAR(c[2].center, 0.5, 1e-12);   // 5th
}

TEST(HarmonicComb, StopsAtNyquist) {
    // 0.3 of Nyquist: 0.3, 0.6, 0.9 fit; 1.2 does not.
    auto c = harmonicComb(0.3, 0.001, 10, false);
    EXPECT_EQ(c.size(), 3u);
    for (const auto& b : c) EXPECT_LT(b.center, 1.0);
}

TEST(HarmonicComb, RejectsNonsenseParameters) {
    EXPECT_TRUE(harmonicComb(0.0,  0.001, 4, false).empty());
    EXPECT_TRUE(harmonicComb(-0.1, 0.001, 4, false).empty());
    EXPECT_TRUE(harmonicComb(0.1,  0.001, 0, false).empty());
}

// ===========================================================================
// Zero-phase filtering and HarmonicNotchTransform
// ===========================================================================

namespace {

std::vector<float> runTx(ITransform& tx, std::vector<float> x) {
    tx.reset();
    const int64_t n = tx.apply(x.data(), static_cast<int64_t>(x.size()), 0);
    x.resize(static_cast<size_t>(std::max<int64_t>(0, n)));
    return x;
}

// Phase of the component at `cycles_per_sample`, in radians.
double tonePhase(const std::vector<float>& x, double cycles_per_sample) {
    double re = 0.0, im = 0.0;
    const int64_t n = static_cast<int64_t>(x.size());
    for (int64_t i = 0; i < n; ++i) {
        const double ph = 2.0 * M_PI * cycles_per_sample * static_cast<double>(i);
        re += static_cast<double>(x[static_cast<size_t>(i)]) * std::cos(ph);
        im -= static_cast<double>(x[static_cast<size_t>(i)]) * std::sin(ph);
    }
    return std::atan2(im, re);
}

std::vector<float> makeTone(int64_t n, double cycles_per_sample, float amp = 1.0f,
                             float offset = 0.0f) {
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = offset +
            amp * static_cast<float>(std::sin(2.0 * M_PI * cycles_per_sample *
                                              static_cast<double>(i)));
    return x;
}

} // namespace

TEST(ZeroPhaseBiquad, FlagDrivesWholeTraceRequirement) {
    BiquadFilterTransform tx(BiquadFilterTransform::FilterType::Lowpass, 0.2f, 0.707f);
    EXPECT_FALSE(tx.requiresWholeTrace());       // causal default unchanged
    EXPECT_FALSE(tx.zeroPhase());
    tx.setZeroPhase(true);
    EXPECT_TRUE(tx.requiresWholeTrace());
    EXPECT_TRUE(tx.zeroPhase());
}

// The whole point: a causal filter delays a passband tone, a zero-phase one
// does not. This is the property that keeps per-sample analyses honest.
TEST(ZeroPhaseBiquad, PassbandToneIsNotDelayed) {
    const int64_t n = 8192;
    const double  f = 64.0 / 8192.0;             // well inside a 0.2 Nyq passband
    auto in = makeTone(n, f);
    const double phase_in = tonePhase(in, f);

    BiquadFilterTransform causal(BiquadFilterTransform::FilterType::Lowpass, 0.2f, 0.707f);
    const double phase_causal = tonePhase(runTx(causal, in), f);

    BiquadFilterTransform zp(BiquadFilterTransform::FilterType::Lowpass, 0.2f, 0.707f);
    zp.setZeroPhase(true);
    const double phase_zp = tonePhase(runTx(zp, in), f);

    auto wrap = [](double d) {
        while (d >  M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        return std::abs(d);
    };
    EXPECT_GT(wrap(phase_causal - phase_in), 0.05) << "causal filter should shift phase";
    EXPECT_LT(wrap(phase_zp - phase_in), 0.01)     << "zero-phase filter must not";
}

// A high-Q notch has the steepest phase slope of any second-order section, so
// this is where a causal filter does the most damage to sample alignment.
TEST(ZeroPhaseBiquad, HighQNotchLeavesNeighbourPhaseIntact) {
    const int64_t n = 8192;
    const double  f_keep = 100.0 / 8192.0;
    auto in = makeTone(n, f_keep);
    const double phase_in = tonePhase(in, f_keep);

    BiquadFilterTransform zp(BiquadFilterTransform::FilterType::Notch,
                             /*cutoff=*/0.5f, /*q=*/30.0f);
    zp.setZeroPhase(true);
    const double phase_out = tonePhase(runTx(zp, in), f_keep);
    EXPECT_NEAR(phase_out, phase_in, 0.02);
}

// Forward+backward squares the magnitude response. Measured *on the notch
// centre* both passes annihilate the tone and what is left is numerical
// residue, so the comparison has to be made on the notch's shoulder where
// attenuation is partial and squaring is actually observable.
TEST(ZeroPhaseBiquad, ResponseIsSquaredOnTheNotchShoulder) {
    const int64_t n = 8192;
    const double  f = 0.28;                       // 0.56 x Nyquist: on the skirt
    auto in = makeTone(n, f);
    const double a_in = toneAmplitude(in, f);

    BiquadFilterTransform causal(BiquadFilterTransform::FilterType::Notch, 0.5f, 5.0f);
    const double a_causal = toneAmplitude(runTx(causal, in), f);

    BiquadFilterTransform zp(BiquadFilterTransform::FilterType::Notch, 0.5f, 5.0f);
    zp.setZeroPhase(true);
    const double a_zp = toneAmplitude(runTx(zp, in), f);

    // Partial attenuation, or the test proves nothing.
    ASSERT_GT(a_causal, 0.05 * a_in);
    ASSERT_LT(a_causal, 0.95 * a_in);

    EXPECT_LT(a_zp, a_causal);
    // |H|^2 relationship, in amplitude terms: a_zp == a_causal^2 / a_in.
    EXPECT_NEAR(a_zp, a_causal * a_causal / a_in, 0.05 * a_in);
}

// A trace sitting on a large DC offset is the normal case for power traces.
// Without odd-reflected padding and steady-state initialisation, the filter
// treats the first sample as a step and rings inward from both edges.
TEST(ZeroPhaseBiquad, LargeOffsetDoesNotRingAtEdges) {
    const int64_t n = 4096;
    auto in = makeTone(n, 32.0 / 4096.0, /*amp=*/0.01f, /*offset=*/100.0f);

    BiquadFilterTransform zp(BiquadFilterTransform::FilterType::Lowpass, 0.1f, 0.707f);
    zp.setZeroPhase(true);
    auto out = runTx(zp, in);
    ASSERT_EQ(out.size(), in.size());

    // A lowpass must pass the offset through essentially untouched, right up
    // to the first and last sample.
    for (int64_t i : {INT64_C(0), INT64_C(1), INT64_C(5), INT64_C(20),
                      n - 21, n - 6, n - 2, n - 1})
        EXPECT_NEAR(out[static_cast<size_t>(i)], 100.0f, 0.5f)
            << "edge ringing at sample " << i;
}

TEST(ZeroPhaseBiquad, ShortTraceDoesNotCrash) {
    BiquadFilterTransform zp(BiquadFilterTransform::FilterType::Lowpass, 0.2f, 0.707f);
    zp.setZeroPhase(true);
    for (int64_t n : {1, 2, 3, 8}) {
        std::vector<float> x(static_cast<size_t>(n), 1.0f);
        auto out = runTx(zp, x);
        ASSERT_EQ(out.size(), static_cast<size_t>(n));
        for (float v : out) EXPECT_TRUE(std::isfinite(v)) << "n=" << n;
    }
}

TEST(ZeroPhaseBiquad, CausalPathUnchangedByTheNewFlag) {
    // Guards the existing behaviour: with zero-phase off, output must be
    // bit-identical to what the filter produced before the flag existed.
    const int64_t n = 512;
    auto in = makeTone(n, 0.05);
    BiquadFilterTransform a(BiquadFilterTransform::FilterType::Lowpass, 0.3f, 0.707f);
    auto out = runTx(a, in);

    // Reference: the documented Direct-Form-II-Transposed recurrence.
    const BiquadCoeffs c = designBiquad(BiquadFilterTransform::FilterType::Lowpass,
                                         0.3, 0.707);
    double z1 = 0.0, z2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(in[static_cast<size_t>(i)]);
        const double y = c.b0 * x + z1;
        z1 = c.b1 * x + z2 - c.a1 * y;
        z2 = c.b2 * x - c.a2 * y;
        EXPECT_FLOAT_EQ(out[static_cast<size_t>(i)], static_cast<float>(y)) << "i=" << i;
    }
}

TEST(DesignBiquad, MatchesTransformCoefficients) {
    // designBiquad() and BiquadFilterTransform::computeCoeffs() must agree —
    // they are the same RBJ formulas and the cascade path relies on it.
    const int64_t n = 256;
    auto in = makeTone(n, 0.07);
    for (auto type : {BiquadFilterTransform::FilterType::Lowpass,
                      BiquadFilterTransform::FilterType::Highpass,
                      BiquadFilterTransform::FilterType::Bandpass,
                      BiquadFilterTransform::FilterType::Notch}) {
        BiquadFilterTransform tx(type, 0.25f, 1.5f);
        auto out = runTx(tx, in);
        const BiquadCoeffs c = designBiquad(type, 0.25, 1.5);
        double z1 = 0.0, z2 = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double x = static_cast<double>(in[static_cast<size_t>(i)]);
            const double y = c.b0 * x + z1;
            z1 = c.b1 * x + z2 - c.a1 * y;
            z2 = c.b2 * x - c.a2 * y;
            ASSERT_NEAR(out[static_cast<size_t>(i)], static_cast<float>(y), 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// HarmonicNotchTransform
// ---------------------------------------------------------------------------

TEST(HarmonicNotch, NotchCountStopsAtNyquist) {
    HarmonicNotchTransform tx(0.3f, 10, 20.0f, false, true);
    EXPECT_EQ(tx.activeNotches(), 3);        // 0.3, 0.6, 0.9
    HarmonicNotchTransform odd(0.1f, 3, 20.0f, /*odd_only=*/true, true);
    EXPECT_EQ(odd.activeNotches(), 3);       // 0.1, 0.3, 0.5
}

TEST(HarmonicNotch, ZeroPhaseFlagDrivesWholeTraceRequirement) {
    HarmonicNotchTransform zp(0.2f, 3, 20.0f, false, /*zero_phase=*/true);
    EXPECT_TRUE(zp.requiresWholeTrace());
    HarmonicNotchTransform causal(0.2f, 3, 20.0f, false, /*zero_phase=*/false);
    EXPECT_FALSE(causal.requiresWholeTrace());
}

// The motivating case: a fundamental plus harmonics on top of the signal we
// care about, all removed in one stage.
TEST(HarmonicNotch, RemovesCombAndKeepsSignal) {
    const int64_t n  = 16384;
    const double  f0 = 0.05;                   // cycles/sample => 0.1 x Nyquist
    const double  f_signal = 0.004;            // far below the first notch
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        double acc = 0.3 * std::sin(2.0 * M_PI * f_signal * static_cast<double>(i));
        for (int k = 1; k <= 4; ++k)
            acc += std::sin(2.0 * M_PI * f0 * k * static_cast<double>(i));
        x[static_cast<size_t>(i)] = static_cast<float>(acc);
    }

    HarmonicNotchTransform tx(static_cast<float>(f0 * 2.0), 4, 25.0f, false, true);
    ASSERT_EQ(tx.activeNotches(), 4);
    auto out = runTx(tx, x);
    ASSERT_EQ(out.size(), x.size());

    for (int k = 1; k <= 4; ++k)
        EXPECT_LT(toneAmplitude(out, f0 * k), 0.15)
            << "harmonic " << k << " survived the comb";
    EXPECT_NEAR(toneAmplitude(out, f_signal), 0.3, 0.03);
}

TEST(HarmonicNotch, OddOnlyLeavesEvenHarmonicsAlone) {
    const int64_t n  = 16384;
    const double  f0 = 0.05;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int k = 1; k <= 4; ++k)
            acc += std::sin(2.0 * M_PI * f0 * k * static_cast<double>(i));
        x[static_cast<size_t>(i)] = static_cast<float>(acc);
    }
    HarmonicNotchTransform tx(static_cast<float>(f0 * 2.0), 2, 25.0f,
                              /*odd_only=*/true, true);
    auto out = runTx(tx, x);
    EXPECT_LT(toneAmplitude(out, f0 * 1), 0.15);   // 1st notched
    EXPECT_GT(toneAmplitude(out, f0 * 2), 0.80);   // 2nd left alone
    EXPECT_LT(toneAmplitude(out, f0 * 3), 0.15);   // 3rd notched
    EXPECT_GT(toneAmplitude(out, f0 * 4), 0.80);   // 4th left alone
}

// The causal path must give the same answer whether it is handed the trace in
// one call or in chunks — that is what its carried delay registers are for.
TEST(HarmonicNotch, CausalPathIsChunkInvariant) {
    const int64_t n = 4096;
    auto x = makeTone(n, 0.05, 1.0f, 0.0f);

    HarmonicNotchTransform whole(0.1f, 3, 20.0f, false, /*zero_phase=*/false);
    auto ref = runTx(whole, x);

    HarmonicNotchTransform chunked(0.1f, 3, 20.0f, false, /*zero_phase=*/false);
    chunked.reset();
    std::vector<float> got = x;
    const int64_t step = 512;
    for (int64_t off = 0; off < n; off += step)
        chunked.apply(got.data() + off, std::min(step, n - off), off);

    for (int64_t i = 0; i < n; ++i)
        ASSERT_NEAR(got[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], 1e-4)
            << "chunk boundary artefact at " << i;
}

TEST(HarmonicNotch, ResetClearsCausalState) {
    HarmonicNotchTransform tx(0.1f, 2, 20.0f, false, /*zero_phase=*/false);
    auto x = makeTone(1024, 0.05);
    auto a = runTx(tx, x);      // runTx calls reset() first
    auto b = runTx(tx, x);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_FLOAT_EQ(a[i], b[i]);
}

TEST(HarmonicNotch, CloneIsIndependent) {
    HarmonicNotchTransform tx(0.1f, 5, 20.0f, false, true);
    auto c = tx.clone();
    auto* cp = dynamic_cast<HarmonicNotchTransform*>(c.get());
    ASSERT_NE(cp, nullptr);
    EXPECT_EQ(cp->harmonics(), 5);
    cp->setHarmonics(2);
    EXPECT_EQ(tx.harmonics(), 5);
    EXPECT_EQ(cp->harmonics(), 2);
}

TEST(HarmonicNotch, NameMentionsItsConfiguration) {
    HarmonicNotchTransform tx(0.25f, 3, 12.0f, true, true);
    const std::string nm = tx.name();
    EXPECT_NE(nm.find("Harmonic Notch"), std::string::npos);
    EXPECT_NE(nm.find("odd"), std::string::npos);
    EXPECT_NE(nm.find("zero-phase"), std::string::npos);
}

// ===========================================================================
// End-to-end: does removing narrowband interference actually recover leakage?
//
// The unit tests above prove each piece behaves as specified. This section
// proves the thing a user actually cares about: that a t-test which cannot see
// the leakage through a switching-converter comb can see it once the comb is
// excised — and, just as importantly, that excision does not eat the leakage
// when there was no interference to begin with.
// ===========================================================================

#include "ttest.h"

namespace {

constexpr int64_t kSamples   = 1024;
constexpr int32_t kPerGroup  = 1000;
constexpr int64_t kLeakStart = 700;   // where the group-dependent bump sits
constexpr int64_t kLeakLen   = 20;

// The interferer: a comb whose phase is random per trace. Random phase is what
// makes it *noise* rather than a constant offset — a fixed-phase interferer
// would cancel in a difference of means and bother nobody.
constexpr double kInterfF0   = 300.0 / 1024.0;   // cycles/sample
constexpr int    kInterfHarm = 3;
constexpr float  kInterfAmp  = 1.0f;

constexpr float kLeakAmp  = 0.02f;   // 50x smaller than the interferer
constexpr float kNoiseStd = 0.05f;

// One synthetic trace. group 0/1 selects whether the leakage bump is present.
std::vector<float> makeSyntheticTrace(int group, std::mt19937& rng, bool with_interferer) {
    std::normal_distribution<float>       noise(0.0f, kNoiseStd);
    std::uniform_real_distribution<double> phase(0.0, 2.0 * M_PI);

    std::vector<float> t(static_cast<size_t>(kSamples));
    for (int64_t i = 0; i < kSamples; ++i) t[static_cast<size_t>(i)] = noise(rng);

    // Group-dependent leakage: a small raised-cosine bump, i.e. broadband in
    // frequency, which is exactly why a narrow excision cannot remove it.
    if (group == 1) {
        for (int64_t k = 0; k < kLeakLen; ++k) {
            const double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * static_cast<double>(k) /
                                                   static_cast<double>(kLeakLen)));
            t[static_cast<size_t>(kLeakStart + k)] += kLeakAmp * static_cast<float>(w);
        }
    }

    if (with_interferer) {
        for (int hk = 1; hk <= kInterfHarm; ++hk) {
            const double f  = kInterfF0 * hk;
            if (f >= 0.5) break;                       // past Nyquist
            const double ph = phase(rng);
            const float  a  = kInterfAmp / static_cast<float>(hk);
            for (int64_t i = 0; i < kSamples; ++i)
                t[static_cast<size_t>(i)] += a * static_cast<float>(
                    std::sin(2.0 * M_PI * f * static_cast<double>(i) + ph));
        }
    }
    return t;
}

// Largest |t| inside the leakage window — the number an analyst would read off
// the plot and compare against the ±4.5 TVLA threshold.
double peakTInLeakRegion(bool with_interferer,
                          const std::vector<std::shared_ptr<ITransform>>& pipeline)
{
    TTestAccumulator acc(static_cast<int32_t>(kSamples));
    std::mt19937 rng(20260926u);          // fixed: this test must not flake

    for (int32_t i = 0; i < kPerGroup * 2; ++i) {
        const int group = i % 2;
        auto tr = makeSyntheticTrace(group, rng, with_interferer);
        int64_t n = static_cast<int64_t>(tr.size());
        for (const auto& t : pipeline) t->reset();
        for (const auto& t : pipeline) n = t->apply(tr.data(), n, 0);
        acc.addTrace(group, tr.data(), static_cast<int32_t>(n));
    }

    std::vector<float> tv;
    std::string err;
    if (!acc.compute(tv, err)) return 0.0;

    double peak = 0.0;
    for (int64_t i = kLeakStart; i < kLeakStart + kLeakLen &&
                                 i < static_cast<int64_t>(tv.size()); ++i)
        peak = std::max(peak, std::abs(static_cast<double>(tv[static_cast<size_t>(i)])));
    return peak;
}

// The comb, as bands the excision transform understands. Normalised frequency
// is 2x cycles-per-sample.
std::vector<SpectralBand> interfererBands() {
    std::vector<SpectralBand> b;
    for (int hk = 1; hk <= kInterfHarm; ++hk) {
        const double f = kInterfF0 * hk;
        if (f >= 0.5) break;
        b.push_back(SpectralBand{f * 2.0, 6.0 / 512.0});
    }
    return b;
}

} // namespace

// Baseline: with no interferer the leakage is comfortably detectable. If this
// ever fails, the synthetic model is broken and the rest means nothing.
TEST(InterferenceRemovalE2E, LeakageIsDetectableInACleanCapture) {
    const double peak = peakTInLeakRegion(/*with_interferer=*/false, {});
    EXPECT_GT(peak, 4.5) << "clean-capture leakage should be plainly visible";
}

// The problem being solved: a comb 50x the leakage amplitude buries it.
TEST(InterferenceRemovalE2E, InterfererHidesTheLeakage) {
    const double peak = peakTInLeakRegion(/*with_interferer=*/true, {});
    EXPECT_LT(peak, 4.5) << "the interferer was supposed to bury the leakage, "
                             "but |t| reached " << peak;
}

// The payoff. Measured on this fixture: peak |t| in the leakage window goes
// from 2.01 (invisible — below the +-4.5 TVLA threshold) to 9.86 with excision
// and 9.75 with the zero-phase comb, while excising the same three bands from
// an interference-free capture costs 0.04% of the leakage signal.
TEST(InterferenceRemovalE2E, ExcisionRecoversTheLeakage) {
    const double before = peakTInLeakRegion(true, {});
    std::vector<std::shared_ptr<ITransform>> pipe{
        std::make_shared<SpectralExciseTransform>(interfererBands())};
    const double after = peakTInLeakRegion(true, pipe);

    EXPECT_GT(after, before * 3.0)
        << "excision should transform detectability, not nudge it: "
        << before << " -> " << after;
    EXPECT_GT(after, 4.5)
        << "|t| only reached " << after << " after excision";
}

// The same recovery via the streaming-friendly route, so both removal paths are
// known to work on the same data.
TEST(InterferenceRemovalE2E, ZeroPhaseHarmonicNotchRecoversTheLeakage) {
    const double before = peakTInLeakRegion(true, {});
    std::vector<std::shared_ptr<ITransform>> pipe{
        std::make_shared<HarmonicNotchTransform>(
            static_cast<float>(kInterfF0 * 2.0), kInterfHarm, 40.0f,
            /*odd_only=*/false, /*zero_phase=*/true)};
    const double after = peakTInLeakRegion(true, pipe);

    EXPECT_GT(after, before * 3.0) << before << " -> " << after;
    EXPECT_GT(after, 4.5) << "|t| only reached " << after << " after notching";
}

// The failure mode that matters more than any of the above: preprocessing that
// quietly destroys the signal. Excising three narrow bands from a clean capture
// must leave a broadband leakage feature essentially untouched.
TEST(InterferenceRemovalE2E, ExcisionDoesNotEatLeakageInACleanCapture) {
    const double before = peakTInLeakRegion(false, {});
    std::vector<std::shared_ptr<ITransform>> pipe{
        std::make_shared<SpectralExciseTransform>(interfererBands())};
    const double after = peakTInLeakRegion(false, pipe);

    EXPECT_GT(after, before * 0.9)
        << "excision cost " << (100.0 * (1.0 - after / before))
        << "% of the leakage signal in a capture that had no interference";
}

// Likewise for the notch — and this is the one where a *causal* filter would
// show its teeth, by shifting the leakage bump along the sample axis out of the
// window the analyst is looking at.
TEST(InterferenceRemovalE2E, ZeroPhaseNotchKeepsLeakageInPlace) {
    const double before = peakTInLeakRegion(false, {});
    std::vector<std::shared_ptr<ITransform>> zp_pipe{
        std::make_shared<HarmonicNotchTransform>(
            static_cast<float>(kInterfF0 * 2.0), kInterfHarm, 40.0f, false,
            /*zero_phase=*/true)};
    const double after_zp = peakTInLeakRegion(false, zp_pipe);
    EXPECT_GT(after_zp, before * 0.9)
        << "zero-phase notching should not move or shrink the leakage peak";
}

// ===========================================================================
// estimatePeriod / PeriodicSubtractTransform / ComplexDemodulateTransform
// ===========================================================================

TEST(EstimatePeriod, FindsAnIntegerPeriod) {
    const int64_t n = 4096;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * M_PI * static_cast<double>(i) / 64.0));
    double score = 0.0;
    const double T = estimatePeriod(x.data(), n, 8, 512, 0.3, &score);
    EXPECT_NEAR(T, 64.0, 0.5);
    EXPECT_GT(score, 0.9);
}

// A switcher's period almost never lands on a whole number of samples, so the
// fractional refinement is not a nicety.
TEST(EstimatePeriod, RecoversAFractionalPeriod) {
    const int64_t n = 8192;
    const double  T = 73.4;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * M_PI * static_cast<double>(i) / T));
    const double got = estimatePeriod(x.data(), n, 8, 512);
    EXPECT_NEAR(got, T, 0.5) << "got " << got;
}

TEST(EstimatePeriod, SurvivesANonSinusoidalWaveform) {
    // A sawtooth: fundamental plus a long harmonic tail, which is what a
    // switching converter actually looks like.
    const int64_t n = 8192;
    const double  T = 100.0;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] =
            static_cast<float>(std::fmod(static_cast<double>(i), T) / T - 0.5);
    EXPECT_NEAR(estimatePeriod(x.data(), n, 8, 512), T, 0.5);
}

TEST(EstimatePeriod, ReturnsZeroOnNoise) {
    std::mt19937 rng(4242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(4096);
    for (auto& v : x) v = nd(rng);
    // White noise has no period; a high bar must reject it rather than
    // returning whatever lag the autocorrelation happened to favour.
    EXPECT_EQ(estimatePeriod(x.data(), static_cast<int64_t>(x.size()), 8, 512, 0.5), 0.0);
}

TEST(EstimatePeriod, DcOffsetDoesNotFoolIt) {
    const int64_t n = 4096;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = 1000.0f +
            static_cast<float>(std::sin(2.0 * M_PI * static_cast<double>(i) / 50.0));
    EXPECT_NEAR(estimatePeriod(x.data(), n, 8, 512), 50.0, 0.5);
}

TEST(EstimatePeriod, RejectsDegenerateInput) {
    std::vector<float> x(4, 1.0f);
    EXPECT_EQ(estimatePeriod(x.data(), 4, 2, 2), 0.0);
    EXPECT_EQ(estimatePeriod(nullptr, 1000, 2, 100), 0.0);
    std::vector<float> y(1000, 1.0f);
    EXPECT_EQ(estimatePeriod(y.data(), 1000, 400, 100), 0.0);  // min > max
}

// ---------------------------------------------------------------------------

TEST(PeriodicSubtract, DeclaresWholeTraceRequirement) {
    PeriodicSubtractTransform tx(64.0);
    EXPECT_TRUE(tx.requiresWholeTrace());
}

// The motivating case: a non-sinusoidal interferer removed whole, including
// every harmonic, with no notches at all.
TEST(PeriodicSubtract, RemovesASawtoothInterfererEntirely) {
    const int64_t n = 8192;
    const double  T = 100.0;
    const double  f_signal = 7.0 / 8192.0;      // asynchronous with T
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double saw = std::fmod(static_cast<double>(i), T) / T - 0.5;
        x[static_cast<size_t>(i)] = static_cast<float>(
            saw + 0.05 * std::sin(2.0 * M_PI * f_signal * static_cast<double>(i)));
    }

    PeriodicSubtractTransform tx(T);
    auto out = runTx(tx, x);
    ASSERT_EQ(out.size(), x.size());
    EXPECT_NEAR(tx.lastPeriodUsed(), T, 1e-9);

    // Fundamental and harmonics all gone...
    for (int k = 1; k <= 6; ++k)
        EXPECT_LT(toneAmplitude(out, static_cast<double>(k) / T), 0.02)
            << "harmonic " << k << " of the sawtooth survived";
    // ...and the asynchronous signal survives.
    EXPECT_NEAR(toneAmplitude(out, f_signal), 0.05, 0.01);
}

TEST(PeriodicSubtract, HandlesAFractionalPeriod) {
    const int64_t n = 16384;
    const double  T = 137.3;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = static_cast<float>(
            std::fmod(static_cast<double>(i), T) / T - 0.5);

    PeriodicSubtractTransform tx(T);
    auto out = runTx(tx, x);
    for (int k = 1; k <= 4; ++k)
        EXPECT_LT(toneAmplitude(out, static_cast<double>(k) / T), 0.03)
            << "harmonic " << k << " survived at fractional T";
}

TEST(PeriodicSubtract, AutoEstimatesThePeriod) {
    const int64_t n = 8192;
    const double  T = 64.0;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = static_cast<float>(
            std::fmod(static_cast<double>(i), T) / T - 0.5);

    PeriodicSubtractTransform tx(0.0);          // 0 = auto
    tx.setAutoRange(8, 512);
    auto out = runTx(tx, x);
    EXPECT_NEAR(tx.lastPeriodUsed(), T, 0.5);
    EXPECT_LT(toneAmplitude(out, 1.0 / T), 0.03);
}

// Without a convincing period, doing nothing is the only safe answer.
TEST(PeriodicSubtract, LeavesNonPeriodicTracesAlone) {
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(4096);
    for (auto& v : x) v = nd(rng);
    const std::vector<float> before = x;

    PeriodicSubtractTransform tx(0.0);
    tx.setAutoRange(8, 512);
    tx.setAutoMinCorrelation(0.5);
    auto out = runTx(tx, x);
    EXPECT_EQ(tx.lastPeriodUsed(), 0.0);
    for (size_t i = 0; i < before.size(); ++i) EXPECT_FLOAT_EQ(out[i], before[i]);
}

// Subtracting the template must not move the trace vertically; the template's
// mean is the trace's DC level, not part of the interference.
TEST(PeriodicSubtract, PreservesDcOffset) {
    const int64_t n = 8192;
    const double  T = 64.0;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = 250.0f + static_cast<float>(
            std::fmod(static_cast<double>(i), T) / T - 0.5);

    PeriodicSubtractTransform tx(T);
    auto out = runTx(tx, x);
    double mean = 0.0;
    for (float v : out) mean += v;
    mean /= static_cast<double>(out.size());
    EXPECT_NEAR(mean, 250.0, 0.01);
}

TEST(PeriodicSubtract, CloneIsIndependent) {
    PeriodicSubtractTransform tx(64.0);
    auto c = tx.clone();
    auto* cp = dynamic_cast<PeriodicSubtractTransform*>(c.get());
    ASSERT_NE(cp, nullptr);
    cp->setPeriod(128.0);
    EXPECT_DOUBLE_EQ(tx.period(), 64.0);
    EXPECT_DOUBLE_EQ(cp->period(), 128.0);
}

// ---------------------------------------------------------------------------

TEST(ComplexDemodulate, DeclaresWholeTraceRequirement) {
    ComplexDemodulateTransform tx;
    EXPECT_TRUE(tx.requiresWholeTrace());
}

// The screaming-channel case: a carrier amplitude-modulated by a slow envelope,
// where the information is in the envelope and the carrier is just transport.
TEST(ComplexDemodulate, RecoversAnAmplitudeModulatedEnvelope) {
    const int64_t n  = 8192;
    const double  fc = 0.25;                      // carrier, cycles/sample
    const double  fm = 16.0 / 8192.0;             // modulation
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double t   = static_cast<double>(i);
        const double env = 1.0 + 0.5 * std::sin(2.0 * M_PI * fm * t);
        x[static_cast<size_t>(i)] = static_cast<float>(env * std::sin(2.0 * M_PI * fc * t));
    }

    ComplexDemodulateTransform tx(static_cast<float>(fc * 2.0), 0.02f,
                                   ComplexDemodulateTransform::Output::Envelope);
    auto out = runTx(tx, x);
    ASSERT_EQ(out.size(), x.size());

    // The envelope should be 1 + 0.5*sin(2*pi*fm*t): mean 1, modulation 0.5.
    // Edges are excluded — a finite transform cannot know what came before.
    double mean = 0.0;
    const int64_t lo = 512, hi = n - 512;
    for (int64_t i = lo; i < hi; ++i) mean += out[static_cast<size_t>(i)];
    mean /= static_cast<double>(hi - lo);
    EXPECT_NEAR(mean, 1.0, 0.05);

    std::vector<float> mid(out.begin() + lo, out.begin() + hi);
    EXPECT_NEAR(toneAmplitude(mid, fm), 0.5, 0.05);
    // ...and the carrier itself is gone from the output.
    EXPECT_LT(toneAmplitude(mid, fc), 0.05);
}

TEST(ComplexDemodulate, EnvelopeOfAPlainToneIsItsAmplitude) {
    const int64_t n  = 8192;
    const double  fc = 0.3;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = 0.75f * static_cast<float>(
            std::sin(2.0 * M_PI * fc * static_cast<double>(i)));

    ComplexDemodulateTransform tx(static_cast<float>(fc * 2.0), 0.02f);
    auto out = runTx(tx, x);
    for (int64_t i = 1024; i < n - 1024; ++i)
        ASSERT_NEAR(out[static_cast<size_t>(i)], 0.75f, 0.02f) << "at " << i;
}

// A band that contains nothing must demodulate to nothing — this is what makes
// the transform usable to *scan* for which bands carry anything.
TEST(ComplexDemodulate, EmptyBandGivesNearZero) {
    const int64_t n  = 8192;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(2.0 * M_PI * 0.1 * static_cast<double>(i)));
    ComplexDemodulateTransform tx(0.8f, 0.02f);   // nowhere near the tone
    auto out = runTx(tx, x);
    for (int64_t i = 1024; i < n - 1024; ++i)
        ASSERT_LT(std::abs(out[static_cast<size_t>(i)]), 0.02f) << "at " << i;
}

TEST(ComplexDemodulate, BasebandOutputStillOscillates) {
    const int64_t n  = 8192;
    const double  fc = 0.25;
    const double  off = 32.0 / 8192.0;            // 32 bins above the centre
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        x[static_cast<size_t>(i)] = static_cast<float>(
            std::sin(2.0 * M_PI * (fc + off) * static_cast<double>(i)));

    ComplexDemodulateTransform tx(static_cast<float>(fc * 2.0), 0.02f,
                                   ComplexDemodulateTransform::Output::Baseband);
    auto out = runTx(tx, x);
    // Shifted down by fc, the tone should now sit at `off`.
    std::vector<float> mid(out.begin() + 1024, out.end() - 1024);
    EXPECT_GT(toneAmplitude(mid, off), 0.8);
}

TEST(ComplexDemodulate, HandlesTraceLengthChange) {
    ComplexDemodulateTransform tx(0.5f, 0.05f);
    for (int64_t n : {1024, 777, 4096}) {
        std::vector<float> x(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] =
                static_cast<float>(std::sin(2.0 * M_PI * 0.25 * static_cast<double>(i)));
        auto out = runTx(tx, x);
        ASSERT_EQ(out.size(), static_cast<size_t>(n)) << "n=" << n;
        for (float v : out) ASSERT_TRUE(std::isfinite(v)) << "n=" << n;
    }
}

TEST(ComplexDemodulate, CloneIsIndependent) {
    ComplexDemodulateTransform tx(0.5f, 0.05f);
    auto c = tx.clone();
    auto* cp = dynamic_cast<ComplexDemodulateTransform*>(c.get());
    ASSERT_NE(cp, nullptr);
    cp->setCenter(0.25f);
    EXPECT_FLOAT_EQ(tx.center(), 0.5f);
    EXPECT_FLOAT_EQ(cp->center(), 0.25f);
}

// Autocorrelation peaks at every multiple of the true period, so an estimator
// that simply takes the argmax happily returns 5T. These pin down that the
// fundamental wins, without overshooting into sub-harmonics.
TEST(EstimatePeriod, PrefersTheFundamentalOverItsMultiples) {
    const int64_t n = 8192;
    for (double T : {64.0, 73.4, 128.0}) {
        std::vector<float> x(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = static_cast<float>(
                std::sin(2.0 * M_PI * static_cast<double>(i) / T));
        EXPECT_NEAR(estimatePeriod(x.data(), n, 8, 1024), T, 0.5)
            << "pure sinusoid, T=" << T;
    }
}

TEST(EstimatePeriod, DoesNotCollapseToASubHarmonic) {
    // Period T, but with a strong second harmonic — the waveform is *not*
    // T/2-periodic, and returning T/2 would blur the folded template by
    // averaging two different half-cycles together.
    const int64_t n = 8192;
    const double  T = 100.0;
    std::vector<float> x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i);
        x[static_cast<size_t>(i)] = static_cast<float>(
            std::sin(2.0 * M_PI * t / T) + 0.8 * std::sin(4.0 * M_PI * t / T));
    }
    const double got = estimatePeriod(x.data(), n, 8, 512);
    EXPECT_NEAR(got, T, 1.0) << "got " << got << " (T/2 would be " << T / 2 << ")";
}

// ===========================================================================
// Leakage vs Frequency — the reduction behind the analysis
//
// The GUI analysis applies the pipeline, runs an STFT, feeds a t-test and
// collapses the resulting plane to max |t| per bin. The reshape-and-reduce
// step is where an off-by-one silently mislabels every frequency on the axis,
// so it gets pinned down here against a hand-built plane, and the STFT
// geometry it depends on is checked against the transform itself.
// ===========================================================================

namespace {

// Mirror of the reduction in computeSpectralLeakageCore(): collapse time away
// by taking the largest |t| any window reached at each frequency.
std::vector<float> maxAbsPerBin(const std::vector<float>& plane,
                                 int64_t n_windows, int64_t n_bins) {
    std::vector<float> out(static_cast<size_t>(n_bins), 0.0f);
    for (int64_t w = 0; w < n_windows; ++w)
        for (int64_t b = 0; b < n_bins; ++b) {
            const float v = std::abs(plane[static_cast<size_t>(w * n_bins + b)]);
            if (v > out[static_cast<size_t>(b)]) out[static_cast<size_t>(b)] = v;
        }
    return out;
}

} // namespace

TEST(SpectralLeakage, StftGeometryMatchesTheReshape) {
    // n_windows must come out of total/n_bins exactly, or the plane is ragged
    // and every row after the first is offset.
    for (int wlen : {64, 256, 1024}) {
        for (double ov : {0.0, 0.5, 0.75}) {
            const int hop = std::max(1, static_cast<int>(std::llround(wlen * (1.0 - ov))));
            STFTMagnitudeTransform stft(wlen, hop);
            const int64_t n_bins = wlen / 2 + 1;
            for (int64_t n : {4096, 8192, 5000}) {
                const int64_t total = stft.transformedCount(n);
                ASSERT_EQ(total % n_bins, 0)
                    << "wlen=" << wlen << " hop=" << hop << " n=" << n;
                const int64_t n_windows = total / n_bins;
                EXPECT_EQ(n_windows, std::max<int64_t>(0, (n - wlen) / hop + 1));
            }
        }
    }
}

TEST(SpectralLeakage, ReductionTakesTheMaximumAcrossWindows) {
    const int64_t nw = 4, nb = 3;
    // Row-major (window, bin). Bin 1 peaks in window 2; bin 2 is negative.
    const std::vector<float> plane = {
        0.5f,  1.0f, -0.5f,
        0.25f, 2.0f, -1.0f,
        0.10f, 9.0f, -0.25f,
        0.75f, 0.5f, -7.0f,
    };
    const auto got = maxAbsPerBin(plane, nw, nb);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_FLOAT_EQ(got[0], 0.75f);
    EXPECT_FLOAT_EQ(got[1], 9.0f);
    EXPECT_FLOAT_EQ(got[2], 7.0f);   // magnitude, sign discarded
}

TEST(SpectralLeakage, MaxBeatsMeanForAShortEvent) {
    // Leakage confined to one window out of many: a mean over time would
    // divide it away, which is exactly why the reduction takes a maximum.
    const int64_t nw = 50, nb = 2;
    std::vector<float> plane(static_cast<size_t>(nw * nb), 0.0f);
    plane[static_cast<size_t>(17 * nb + 1)] = 12.0f;
    const auto got = maxAbsPerBin(plane, nw, nb);
    EXPECT_FLOAT_EQ(got[1], 12.0f);
    double mean = 0.0;
    for (int64_t w = 0; w < nw; ++w) mean += plane[static_cast<size_t>(w * nb + 1)];
    mean /= static_cast<double>(nw);
    EXPECT_LT(mean, 4.5) << "a time-average would have hidden this event";
}

// The payoff: a leak confined to one frequency band must show up at that bin
// and nowhere else, end to end through a real STFT and a real t-test.
TEST(SpectralLeakage, LocalisesALeakToItsOwnFrequencyBand) {
    constexpr int64_t kN     = 2048;
    constexpr int     kWin   = 256;
    constexpr int     kHop   = 128;
    constexpr int32_t kPer   = 300;
    const double      f_leak = 40.0 / 256.0;   // bin 40 of a 256-point window

    STFTMagnitudeTransform stft(kWin, kHop);
    const int64_t n_bins = kWin / 2 + 1;
    const int64_t total  = stft.transformedCount(kN);
    const int64_t n_win  = total / n_bins;
    ASSERT_GT(n_win, 0);

    TTestAccumulator acc(static_cast<int32_t>(total));
    std::mt19937 rng(20260926u);
    std::normal_distribution<float> noise(0.0f, 0.10f);

    for (int32_t i = 0; i < kPer * 2; ++i) {
        const int group = i % 2;
        std::vector<float> t(static_cast<size_t>(kN));
        for (int64_t k = 0; k < kN; ++k) t[static_cast<size_t>(k)] = noise(rng);
        // Group 1 carries a burst at f_leak in the middle of the trace only.
        if (group == 1)
            for (int64_t k = 700; k < 1000; ++k)
                t[static_cast<size_t>(k)] += 0.05f * static_cast<float>(
                    std::sin(2.0 * M_PI * f_leak * static_cast<double>(k)));

        stft.reset();
        const int64_t n = stft.apply(t.data(), kN, 0);
        ASSERT_EQ(n, total);
        acc.addTrace(group, t.data(), static_cast<int32_t>(total));
    }

    std::vector<float> tv;
    std::string err;
    ASSERT_TRUE(acc.compute(tv, err)) << err;
    const auto per_bin = maxAbsPerBin(tv, n_win, n_bins);

    const int64_t leak_bin = 40;
    EXPECT_GT(per_bin[static_cast<size_t>(leak_bin)], 4.5)
        << "the leaking band should be plainly above the TVLA threshold";

    // Every bin well away from the leak must stay quiet, or the analysis would
    // point at frequencies that do not actually leak.
    for (int64_t b = 2; b < n_bins; ++b) {
        if (std::abs(b - leak_bin) <= 3) continue;
        EXPECT_LT(per_bin[static_cast<size_t>(b)], 4.5)
            << "spurious leakage reported at bin " << b;
    }
}

// ===========================================================================
// Leakage vs Frequency under residual misalignment
//
// STFT magnitude discards phase, and a time shift is a phase ramp, so a shift
// smaller than one window is largely absorbed — that is the whole reason
// frequency-domain analysis is said to tolerate misalignment. These pin down
// where that tolerance actually ends, because the failure is silent: a diluted
// t-statistic reads exactly like "this device does not leak".
// ===========================================================================

namespace {

// Peak |t| over all bins for a leak at f_leak, when each trace is randomly
// shifted by up to +-jitter samples. shift_leak=false keeps the leak at a
// fixed position and jitters only the surrounding trace, which isolates the
// two different things misalignment does.
double leakPeakUnderJitter(int stft_win, int stft_hop, int64_t jitter,
                            uint32_t seed = 20260926u)
{
    constexpr int64_t kN     = 4096;
    constexpr int32_t kPer   = 400;
    constexpr int64_t kLeakAt = 2000;
    constexpr int64_t kLeakLen = 256;
    const double f_leak = 40.0 / 256.0;

    STFTMagnitudeTransform stft(stft_win, stft_hop);
    const int64_t n_bins = stft_win / 2 + 1;
    const int64_t total  = stft.transformedCount(kN);
    if (total <= 0) return 0.0;
    const int64_t n_win = total / n_bins;

    TTestAccumulator acc(static_cast<int32_t>(total));
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    std::uniform_int_distribution<int64_t> jd(-jitter, jitter);

    for (int32_t i = 0; i < kPer * 2; ++i) {
        const int group = i % 2;
        const int64_t sh = (jitter > 0) ? jd(rng) : 0;
        // The buffer must hold the STFT *output*, not just the input: with
        // overlap, n_windows * bins routinely exceeds the sample count, and
        // apply() writes in place. (computeSpectralLeakageCore sizes its own
        // buffer as max(raw_ns, total) for exactly this reason.)
        std::vector<float> t(static_cast<size_t>(std::max<int64_t>(kN, total)), 0.0f);
        for (int64_t k = 0; k < kN; ++k) t[static_cast<size_t>(k)] = noise(rng);
        if (group == 1) {
            for (int64_t k = 0; k < kLeakLen; ++k) {
                const int64_t at = kLeakAt + sh + k;
                if (at < 0 || at >= kN) continue;
                t[static_cast<size_t>(at)] += 0.06f * static_cast<float>(
                    std::sin(2.0 * M_PI * f_leak * static_cast<double>(k)));
            }
        }
        stft.reset();
        const int64_t n = stft.apply(t.data(), kN, 0);
        if (n != total) return 0.0;
        acc.addTrace(group, t.data(), static_cast<int32_t>(total));
    }

    std::vector<float> tv;
    std::string err;
    if (!acc.compute(tv, err)) return 0.0;
    const auto per_bin = maxAbsPerBin(tv, n_win, n_bins);
    return *std::max_element(per_bin.begin(), per_bin.end());
}

} // namespace

// Measured on this fixture, peak |t| over all bins:
//   window 512, jitter 0      -> 172    (aligned control)
//   window 512, jitter +-16   -> 171    (absorbed: jitter << window)
//   window 256, jitter +-8    -> 162
//   window 256, jitter +-400  ->  16    (diluted: jitter >> window)
//   window 128, jitter +-192  ->  29
//   window 1024, jitter +-192 ->  82    (same jitter, longer window)
// The last pair is the actionable one: at fixed jitter, an 8x longer window
// recovers roughly 3x the statistic.

// Perfectly aligned: the leak is plainly there. The control for everything
// below.
TEST(SpectralLeakageJitter, AlignedLeakIsDetected) {
    EXPECT_GT(leakPeakUnderJitter(512, 128, /*jitter=*/0), 6.0);
}

// The tolerance that makes frequency-domain SCA attractive: jitter well inside
// one window barely costs anything.
TEST(SpectralLeakageJitter, ShiftMuchSmallerThanWindowIsAbsorbed) {
    const double aligned = leakPeakUnderJitter(512, 128, 0);
    const double jittered = leakPeakUnderJitter(512, 128, /*jitter=*/16);
    EXPECT_GT(jittered, aligned * 0.6)
        << "±16 samples inside a 512-sample window should cost little: "
        << aligned << " -> " << jittered;
    EXPECT_GT(jittered, 4.5);
}

// ...and where it ends. Jitter comparable to the window pushes the event
// across boundaries, splits its energy and dilutes every statistic.
TEST(SpectralLeakageJitter, ShiftComparableToWindowDilutesTheResult) {
    const double small = leakPeakUnderJitter(256, 64, /*jitter=*/8);
    const double large = leakPeakUnderJitter(256, 64, /*jitter=*/400);
    EXPECT_LT(large, small)
        << "jitter far exceeding the window must cost detectability: "
        << small << " -> " << large;
}

// The mitigation, stated as a property: for the *same* jitter, a longer window
// recovers what a short one loses.
TEST(SpectralLeakageJitter, LongerWindowRecoversDetectionAtTheSameJitter) {
    constexpr int64_t kJitter = 192;
    const double short_win = leakPeakUnderJitter(128,  32,  kJitter);
    const double long_win  = leakPeakUnderJitter(1024, 256, kJitter);
    EXPECT_GT(long_win, short_win)
        << "window 128 -> " << short_win << ", window 1024 -> " << long_win;
}
