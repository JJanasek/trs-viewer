#include <gtest/gtest.h>
#include "chain.h"
#include "processing.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QTemporaryDir>

#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Chain persistence for the narrowband-interference transforms.
//
// A chain is the feature that lets a whole workflow be replayed later, so a
// field that round-trips incorrectly does not fail loudly — it silently
// replays a *different* analysis than the one that was authored. These are the
// guards for the five fields added for zero-phase filtering, harmonic notching
// and spectral excision.
// ---------------------------------------------------------------------------

namespace {

// Saves one step, loads it back, and returns what came out.
ChainStep roundTrip(const ChainStep& in) {
    QTemporaryDir dir;
    EXPECT_TRUE(dir.isValid());
    const QString path = dir.filePath("chain.json");

    QString err;
    EXPECT_TRUE(saveChain(path, {in}, err)) << err.toStdString();

    std::vector<ChainStep> out;
    EXPECT_TRUE(loadChain(path, out, err)) << err.toStdString();
    EXPECT_EQ(out.size(), 1u);
    return out.empty() ? ChainStep{} : out[0];
}

// Writes a chain.json by hand with an AddTransform step that carries only the
// fields that existed before this work, to prove an older file still loads.
QString writeLegacyChain(const QTemporaryDir& dir, int transform_type) {
    QJsonObject step;
    step["kind"]           = "add_transform";
    step["transform_type"] = transform_type;
    step["i_window"]       = 64;
    step["i_stride"]       = 4;
    step["i_hop"]          = 32;
    step["d_overlap"]      = 0.5;
    step["d_value"]        = 1.25;
    step["d_q"]            = 0.707;
    step["d_cutoff"]       = 0.1;
    step["i_choice"]       = 3;
    QJsonArray arr; arr.append(step);
    // Matches saveChain()'s real layout: a top-level object with a "steps"
    // array, as the chain*.json files in the repo are written.
    QJsonObject root;
    root["steps"] = arr;
    const QString path = dir.filePath("legacy.json");
    QFile f(path);
    EXPECT_TRUE(f.open(QIODevice::WriteOnly));
    f.write(QJsonDocument(root).toJson());
    f.close();
    return path;
}

} // namespace

// ---------------------------------------------------------------------------
// Filter (transform_type 10) — the zero-phase flag
// ---------------------------------------------------------------------------

TEST(ChainFilter, ZeroPhaseSurvivesRoundTrip) {
    for (bool zp : {false, true}) {
        ChainStep s;
        s.kind           = ChainStep::Kind::AddTransform;
        s.transform_type = 10;
        s.i_choice       = static_cast<int>(BiquadFilterTransform::FilterType::Notch);
        s.d_cutoff       = 0.42;
        s.d_q            = 12.5;
        s.zero_phase     = zp;

        const ChainStep got = roundTrip(s);
        EXPECT_EQ(got.zero_phase, zp);
        EXPECT_NEAR(got.d_cutoff, 0.42, 1e-9);
        EXPECT_NEAR(got.d_q, 12.5, 1e-9);

        auto tx = buildTransformFromStep(got);
        ASSERT_NE(tx, nullptr);
        auto* f = dynamic_cast<BiquadFilterTransform*>(tx.get());
        ASSERT_NE(f, nullptr);
        EXPECT_EQ(f->zeroPhase(), zp);
        EXPECT_EQ(f->requiresWholeTrace(), zp);
    }
}

TEST(ChainFilter, DescribeThenBuildPreservesZeroPhase) {
    BiquadFilterTransform tx(BiquadFilterTransform::FilterType::Bandpass, 0.3f, 4.0f);
    tx.setZeroPhase(true);
    const ChainStep s = describeTransformStep(10, tx);
    EXPECT_TRUE(s.zero_phase);

    auto rebuilt = buildTransformFromStep(s);
    auto* f = dynamic_cast<BiquadFilterTransform*>(rebuilt.get());
    ASSERT_NE(f, nullptr);
    EXPECT_TRUE(f->zeroPhase());
    EXPECT_EQ(f->type(), BiquadFilterTransform::FilterType::Bandpass);
    EXPECT_NEAR(f->cutoff(), 0.3f, 1e-6);
    EXPECT_NEAR(f->q(), 4.0f, 1e-6);
}

// A chain written before zero-phase existed must keep replaying the causal
// filter it was authored with, not silently switch behaviour.
TEST(ChainFilter, LegacyChainStaysCausal) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString path = writeLegacyChain(dir, 10);

    std::vector<ChainStep> steps;
    QString err;
    ASSERT_TRUE(loadChain(path, steps, err)) << err.toStdString();
    ASSERT_EQ(steps.size(), 1u);
    EXPECT_FALSE(steps[0].zero_phase);

    auto tx = buildTransformFromStep(steps[0]);
    auto* f = dynamic_cast<BiquadFilterTransform*>(tx.get());
    ASSERT_NE(f, nullptr);
    EXPECT_FALSE(f->zeroPhase());
    EXPECT_FALSE(f->requiresWholeTrace());
}

// ---------------------------------------------------------------------------
// Harmonic Notch (transform_type 11)
// ---------------------------------------------------------------------------

TEST(ChainHarmonicNotch, RoundTripsEveryField) {
    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = 11;
    s.d_cutoff       = 0.125;
    s.n_harmonics    = 7;
    s.d_q            = 33.5;
    s.odd_only       = true;
    s.zero_phase     = true;

    const ChainStep got = roundTrip(s);
    EXPECT_NEAR(got.d_cutoff, 0.125, 1e-9);
    EXPECT_EQ(got.n_harmonics, 7);
    EXPECT_NEAR(got.d_q, 33.5, 1e-9);
    EXPECT_TRUE(got.odd_only);
    EXPECT_TRUE(got.zero_phase);

    auto tx = buildTransformFromStep(got);
    auto* n = dynamic_cast<HarmonicNotchTransform*>(tx.get());
    ASSERT_NE(n, nullptr);
    EXPECT_NEAR(n->f0(), 0.125f, 1e-6);
    EXPECT_EQ(n->harmonics(), 7);
    EXPECT_NEAR(n->q(), 33.5f, 1e-4);
    EXPECT_TRUE(n->oddOnly());
    EXPECT_TRUE(n->zeroPhase());
}

TEST(ChainHarmonicNotch, DescribeThenBuildIsIdentity) {
    HarmonicNotchTransform tx(0.2f, 4, 15.0f, /*odd_only=*/false, /*zero_phase=*/false);
    const ChainStep s = describeTransformStep(11, tx);
    auto rebuilt = buildTransformFromStep(s);
    auto* n = dynamic_cast<HarmonicNotchTransform*>(rebuilt.get());
    ASSERT_NE(n, nullptr);
    EXPECT_NEAR(n->f0(), tx.f0(), 1e-6);
    EXPECT_EQ(n->harmonics(), tx.harmonics());
    EXPECT_NEAR(n->q(), tx.q(), 1e-4);
    EXPECT_EQ(n->oddOnly(), tx.oddOnly());
    EXPECT_EQ(n->zeroPhase(), tx.zeroPhase());
    EXPECT_EQ(n->activeNotches(), tx.activeNotches());
}

TEST(ChainHarmonicNotch, SummaryMentionsItsShape) {
    ChainStep s;
    s.kind = ChainStep::Kind::AddTransform;
    s.transform_type = 11;
    s.d_cutoff = 0.1; s.n_harmonics = 3; s.d_q = 20.0;
    s.odd_only = true; s.zero_phase = true;
    const QString sum = s.summary();
    EXPECT_TRUE(sum.contains("Harmonic Notch")) << sum.toStdString();
    EXPECT_TRUE(sum.contains("odd"))            << sum.toStdString();
    EXPECT_TRUE(sum.contains("zero-phase"))     << sum.toStdString();
}

// ---------------------------------------------------------------------------
// Spectral Excise (transform_type 12)
// ---------------------------------------------------------------------------

TEST(ChainSpectralExcise, BandListRoundTrips) {
    const std::vector<SpectralBand> bands{
        {0.125, 0.0012}, {0.25, 0.0012}, {0.5, 0.004}, {0.9876543, 1e-6}};

    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = 12;
    s.bands          = QString::fromStdString(encodeSpectralBands(bands));
    s.phase_mode     = 1;

    const ChainStep got = roundTrip(s);
    EXPECT_EQ(got.phase_mode, 1);

    auto tx = buildTransformFromStep(got);
    auto* e = dynamic_cast<SpectralExciseTransform*>(tx.get());
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(e->bands().size(), bands.size());
    for (size_t i = 0; i < bands.size(); ++i) {
        EXPECT_NEAR(e->bands()[i].center,    bands[i].center,    1e-9);
        EXPECT_NEAR(e->bands()[i].halfwidth, bands[i].halfwidth, 1e-12);
    }
    EXPECT_EQ(e->phaseMode(), SpectralExciseTransform::PhaseMode::Randomize);
    EXPECT_TRUE(e->requiresWholeTrace());
}

TEST(ChainSpectralExcise, DescribeThenBuildIsIdentity) {
    SpectralExciseTransform tx({{0.3, 0.001}, {0.6, 0.002}},
                                SpectralExciseTransform::PhaseMode::Preserve);
    const ChainStep s = describeTransformStep(12, tx);
    EXPECT_EQ(s.phase_mode, 0);
    EXPECT_FALSE(s.bands.isEmpty());

    auto rebuilt = buildTransformFromStep(s);
    auto* e = dynamic_cast<SpectralExciseTransform*>(rebuilt.get());
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(e->bands().size(), 2u);
    EXPECT_NEAR(e->bands()[0].center, 0.3, 1e-9);
    EXPECT_NEAR(e->bands()[1].halfwidth, 0.002, 1e-12);
    EXPECT_EQ(e->phaseMode(), SpectralExciseTransform::PhaseMode::Preserve);
}

// A corrupt band string must degrade to a stage that does nothing, never to a
// stage that excises bands nobody asked for.
TEST(ChainSpectralExcise, CorruptBandListBecomesANoOp) {
    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = 12;
    s.bands          = "not-a-band-list";

    auto tx = buildTransformFromStep(s);
    auto* e = dynamic_cast<SpectralExciseTransform*>(tx.get());
    ASSERT_NE(e, nullptr);
    EXPECT_TRUE(e->bands().empty());

    // ...and a no-op stage really is a no-op on data.
    std::vector<float> x(1024);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i)));
    const std::vector<float> before = x;
    e->reset();
    e->apply(x.data(), static_cast<int64_t>(x.size()), 0);
    for (size_t i = 0; i < x.size(); ++i) EXPECT_FLOAT_EQ(x[i], before[i]);
}

TEST(ChainSpectralExcise, EmptyBandListRoundTrips) {
    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = 12;
    s.bands          = "";
    const ChainStep got = roundTrip(s);
    auto tx = buildTransformFromStep(got);
    auto* e = dynamic_cast<SpectralExciseTransform*>(tx.get());
    ASSERT_NE(e, nullptr);
    EXPECT_TRUE(e->bands().empty());
}

// ---------------------------------------------------------------------------
// Regressions on the pre-existing transform types
// ---------------------------------------------------------------------------

// The new fields are appended, and the type tag is an index into a list that
// must never be reordered — so every old index must still map to the same
// concrete transform.
TEST(ChainCompat, LegacyTypeIndicesStillMapToTheSameTransforms) {
    struct Case { int type; const char* what; };
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    std::vector<std::pair<int, bool>> checks = {
        {0, true}, {1, true}, {2, true}, {3, true}, {4, true},
        {5, true}, {6, true}, {7, true}, {8, true}, {9, true}, {10, true},
    };
    for (auto [type, expect_ok] : checks) {
        const QString path = writeLegacyChain(dir, type);
        std::vector<ChainStep> steps;
        QString err;
        ASSERT_TRUE(loadChain(path, steps, err)) << "type " << type;
        ASSERT_EQ(steps.size(), 1u);
        auto tx = buildTransformFromStep(steps[0]);
        EXPECT_EQ(tx != nullptr, expect_ok) << "type " << type;
    }
}

TEST(ChainCompat, DescribeIndexMatchesBuildIndexForEveryType) {
    // describeTransformStep and buildTransformFromStep share one index space;
    // a mismatch would rebuild a different transform than was authored.
    std::vector<std::pair<int, std::shared_ptr<ITransform>>> all = {
        {0,  std::make_shared<AbsTransform>()},
        {1,  std::make_shared<NegateTransform>()},
        {2,  std::make_shared<MovingAverageTransform>(16)},
        {3,  std::make_shared<WindowResampleTransform>(32, 0.25f)},
        {4,  std::make_shared<StrideResampleTransform>(8)},
        {5,  std::make_shared<OffsetTransform>(2.5f)},
        {6,  std::make_shared<ScaleTransform>(3.5f)},
        {7,  std::make_shared<FFTMagnitudeTransform>(FFTMagnitudeTransform::Window::Hamming)},
        {8,  std::make_shared<STFTMagnitudeTransform>(128, 64)},
        {9,  std::make_shared<GaussianNoiseTransform>(0.25f)},
        {10, std::make_shared<BiquadFilterTransform>(
                 BiquadFilterTransform::FilterType::Highpass, 0.2f, 1.0f)},
        {11, std::make_shared<HarmonicNotchTransform>(0.15f, 3, 18.0f, true, true)},
        {12, std::make_shared<SpectralExciseTransform>(
                 std::vector<SpectralBand>{{0.4, 0.001}})},
        {13, std::make_shared<PeriodicSubtractTransform>(250.5, 16, 8192, 0.4)},
        {14, std::make_shared<ComplexDemodulateTransform>(
                 0.55f, 0.02f, ComplexDemodulateTransform::Output::Envelope)},
    };
    for (auto& [idx, tx] : all) {
        const ChainStep s = describeTransformStep(idx, *tx);
        EXPECT_EQ(s.transform_type, idx);
        auto rebuilt = buildTransformFromStep(s);
        ASSERT_NE(rebuilt, nullptr) << "index " << idx;
        EXPECT_EQ(rebuilt->name(), tx->name()) << "index " << idx;
        EXPECT_FALSE(s.summary().contains("Unknown transform")) << "index " << idx;
    }
}

// ---------------------------------------------------------------------------
// Periodic Subtract (13) and Demodulate (14)
// ---------------------------------------------------------------------------

TEST(ChainPeriodicSubtract, ExplicitPeriodRoundTrips) {
    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = 13;
    s.d_period       = 137.25;          // fractional, as a real switcher would be
    s.period_min     = 16;
    s.period_max     = 4096;
    s.d_period_corr  = 0.55;

    const ChainStep got = roundTrip(s);
    EXPECT_NEAR(got.d_period, 137.25, 1e-9);
    EXPECT_EQ(got.period_min, 16);
    EXPECT_EQ(got.period_max, 4096);
    EXPECT_NEAR(got.d_period_corr, 0.55, 1e-9);

    auto tx = buildTransformFromStep(got);
    auto* t = dynamic_cast<PeriodicSubtractTransform*>(tx.get());
    ASSERT_NE(t, nullptr);
    EXPECT_NEAR(t->period(), 137.25, 1e-9);
    EXPECT_EQ(t->autoMin(), 16);
    EXPECT_EQ(t->autoMax(), 4096);
    EXPECT_NEAR(t->autoMinCorrelation(), 0.55, 1e-9);
    EXPECT_TRUE(t->requiresWholeTrace());
}

TEST(ChainPeriodicSubtract, AutoModeRoundTrips) {
    ChainStep s;
    s.kind           = ChainStep::Kind::AddTransform;
    s.transform_type = 13;
    s.d_period       = 0.0;             // 0 = auto-estimate
    s.period_min     = 8;
    s.period_max     = 2048;

    const ChainStep got = roundTrip(s);
    EXPECT_DOUBLE_EQ(got.d_period, 0.0);
    auto tx = buildTransformFromStep(got);
    auto* t = dynamic_cast<PeriodicSubtractTransform*>(tx.get());
    ASSERT_NE(t, nullptr);
    EXPECT_DOUBLE_EQ(t->period(), 0.0);
    EXPECT_TRUE(got.summary().contains("auto")) << got.summary().toStdString();
}

TEST(ChainDemodulate, RoundTripsBothOutputModes) {
    for (int mode : {0, 1}) {
        ChainStep s;
        s.kind           = ChainStep::Kind::AddTransform;
        s.transform_type = 14;
        s.d_cutoff       = 0.6125;
        s.d_bandwidth    = 0.0375;
        s.i_choice       = mode;

        const ChainStep got = roundTrip(s);
        EXPECT_NEAR(got.d_cutoff, 0.6125, 1e-9);
        EXPECT_NEAR(got.d_bandwidth, 0.0375, 1e-9);
        EXPECT_EQ(got.i_choice, mode);

        auto tx = buildTransformFromStep(got);
        auto* t = dynamic_cast<ComplexDemodulateTransform*>(tx.get());
        ASSERT_NE(t, nullptr);
        EXPECT_NEAR(t->center(), 0.6125f, 1e-6);
        EXPECT_NEAR(t->bandwidth(), 0.0375f, 1e-6);
        EXPECT_EQ(t->output() == ComplexDemodulateTransform::Output::Baseband, mode == 1);
        EXPECT_TRUE(t->requiresWholeTrace());
    }
}

// A chain written by a newer build can name a transform this one has never
// heard of. That must be reported as unavailable, not replayed as something
// arbitrary — and must certainly not fall off the end of the switch.
TEST(ChainCompat, UnknownTransformTypeReturnsNullptr) {
    for (int bogus : {15, 99, -1, 1000}) {
        ChainStep s;
        s.kind           = ChainStep::Kind::AddTransform;
        s.transform_type = bogus;
        EXPECT_EQ(buildTransformFromStep(s), nullptr) << "type " << bogus;
        EXPECT_TRUE(s.summary().contains("Unknown transform"))
            << bogus << " -> " << s.summary().toStdString();
    }
}
