#include <gtest/gtest.h>
#include "align.h"
#include "processing.h"
#include "trs_file.h"

#include <cmath>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto noProgress() {
    return [](int done, int total) { (void)done; (void)total; return true; };
}

// Build a single-impulse trace: all zeros except 1.0 at (base_pos + offsets[t]).
// mem must outlive the TrsFile.
static void makeImpulseDataset(TrsFile& file, std::vector<float>& mem,
                                int n_traces, int n_samples, int base_pos,
                                const std::vector<int>& offsets) {
    mem.assign(static_cast<size_t>(n_traces) * n_samples, 0.f);
    for (int t = 0; t < n_traces; ++t) {
        int pos = base_pos + offsets[static_cast<size_t>(t)];
        if (pos >= 0 && pos < n_samples)
            mem[static_cast<size_t>(t) * n_samples + pos] = 1.0f;
    }
    file.openFromArray(mem.data(), n_traces, n_samples);
}

// Build a triangular-bump trace centred at (base_pos + offsets[t]).
static void makeTriangleDataset(TrsFile& file, std::vector<float>& mem,
                                 int n_traces, int n_samples, int base_pos,
                                 const std::vector<int>& offsets, int width) {
    mem.assign(static_cast<size_t>(n_traces) * n_samples, 0.f);
    for (int t = 0; t < n_traces; ++t) {
        int centre = base_pos + offsets[static_cast<size_t>(t)];
        for (int s = 0; s < n_samples; ++s) {
            int d = std::abs(s - centre);
            if (d < width)
                mem[static_cast<size_t>(t) * n_samples + s] =
                    1.0f - static_cast<float>(d) / width;
        }
    }
    file.openFromArray(mem.data(), n_traces, n_samples);
}

// ---------------------------------------------------------------------------
// alignByPeak — basic correctness
// ---------------------------------------------------------------------------

TEST(AlignByPeak, KnownShifts) {
    const int NT = 5, NS = 40, BASE = 10;
    std::vector<int> offsets = {0, 10, -10, 5, -5};
    TrsFile f;
    std::vector<float> mem;
    makeImpulseDataset(f, mem, NT, NS, BASE, offsets);

    AlignResult res;
    std::string err;
    bool ok = alignByPeak(&f, {}, 0, NT, 0, 0, NS, 15, true, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    ASSERT_EQ(static_cast<int>(res.shifts.size()), NT);

    EXPECT_EQ(res.shifts[0], 0);
    for (int t = 1; t < NT; ++t)
        EXPECT_EQ(res.shifts[static_cast<size_t>(t)], offsets[static_cast<size_t>(t)]);
}

TEST(AlignByPeak, IdenticalTracesZeroShift) {
    const int NT = 6, NS = 32;
    std::vector<int> offsets(NT, 0);
    TrsFile f;
    std::vector<float> mem;
    makeImpulseDataset(f, mem, NT, NS, 15, offsets);

    AlignResult res;
    std::string err;
    bool ok = alignByPeak(&f, {}, 0, NT, 0, 0, NS, 10, true, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    for (int t = 0; t < NT; ++t)
        EXPECT_EQ(res.shifts[static_cast<size_t>(t)], 0);
}

// ---------------------------------------------------------------------------
// alignByPeak — non-zero ref_trace_offset
// ---------------------------------------------------------------------------

TEST(AlignByPeak, NonZeroRefTrace) {
    const int NT = 4, NS = 30, BASE = 10;
    std::vector<int> offsets = {5, 0, -3, 7};  // ref = trace 1
    TrsFile f;
    std::vector<float> mem;
    makeImpulseDataset(f, mem, NT, NS, BASE, offsets);

    AlignResult res;
    std::string err;
    bool ok = alignByPeak(&f, {}, 0, NT, 1, 0, NS, 10, true, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    ASSERT_EQ(static_cast<int>(res.shifts.size()), NT);

    EXPECT_EQ(res.shifts[1], 0);
    EXPECT_EQ(res.shifts[0], offsets[0] - offsets[1]);
    EXPECT_EQ(res.shifts[2], offsets[2] - offsets[1]);
    EXPECT_EQ(res.shifts[3], offsets[3] - offsets[1]);
}

// ---------------------------------------------------------------------------
// alignByPeak — ref_trace_offset out of range → error
// ---------------------------------------------------------------------------

TEST(AlignByPeak, RefTraceOutOfRangeError) {
    const int NT = 5, NS = 20;
    std::vector<int> offsets(NT, 0);
    TrsFile f;
    std::vector<float> mem;
    makeImpulseDataset(f, mem, NT, NS, 10, offsets);

    AlignResult res;
    std::string err;
    bool ok = alignByPeak(&f, {}, 0, NT, NT, 0, NS, 5, true, res, noProgress(), err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}

// ---------------------------------------------------------------------------
// alignByXCorr — basic correctness
// ---------------------------------------------------------------------------

TEST(AlignByXCorr, KnownShifts) {
    // Traces have 120 samples, bump centred at 60 with ±5/±3 shifts.
    // Reference region [40, 40+40) = [40, 80): well inside trace bounds so the
    // search window [40-10, 80+10) = [30, 90) fits without clamping.
    const int NT = 5, NS = 120, BASE = 60;
    std::vector<int> offsets = {0, 5, -5, 3, -3};
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, BASE, offsets, 10);

    AlignResult res;
    std::string err;
    bool ok = alignByXCorr(&f, {}, 0, NT, 0, /*ref_first=*/40, /*ref_num=*/40, 10,
                           -2.0f, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    ASSERT_EQ(static_cast<int>(res.shifts.size()), NT);

    EXPECT_EQ(res.shifts[0], 0);
    for (int t = 1; t < NT; ++t)
        EXPECT_EQ(res.shifts[static_cast<size_t>(t)], offsets[static_cast<size_t>(t)]);
}

TEST(AlignByXCorr, IdenticalTracesZeroShift) {
    const int NT = 6, NS = 80;
    std::vector<int> offsets(NT, 0);
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, 40, offsets, 8);

    AlignResult res;
    std::string err;
    // Reference region [20, 60): well inside trace, leaving room for ±5 search
    bool ok = alignByXCorr(&f, {}, 0, NT, 0, 20, 40, 5, -2.0f, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    for (int t = 0; t < NT; ++t)
        EXPECT_EQ(res.shifts[static_cast<size_t>(t)], 0);
}

// ---------------------------------------------------------------------------
// Cancellation — should not crash
// ---------------------------------------------------------------------------

TEST(AlignByPeak, CancellationNoCrash) {
    const int NT = 10, NS = 40;
    std::vector<int> offsets(NT, 0);
    TrsFile f;
    std::vector<float> mem;
    makeImpulseDataset(f, mem, NT, NS, 15, offsets);

    int calls = 0;
    auto cancel = [&](int, int) -> bool { return ++calls > 2; };

    AlignResult res;
    std::string err;
    alignByPeak(&f, {}, 0, NT, 0, 0, NS, 5, true, res, cancel, err);
    // No assertion — just verify no crash/hang
}

// ---------------------------------------------------------------------------
// Processing pipeline is applied before searching
// ---------------------------------------------------------------------------

// Signed-max peak search on a trough (negative bump): without the pipeline,
// the flat zero baseline is the signed max everywhere, so every trace
// "aligns" to shift 0. Running the same data through an AbsTransform first
// turns the trough into the tallest positive feature, so the search should
// recover the true offsets — proving the pipeline stage actually ran.
TEST(AlignByPeak, PipelineAppliedBeforeSearch) {
    const int NT = 4, NS = 120, BASE = 60, WIDTH = 10;
    std::vector<int> offsets = {0, 5, -5, 3};
    std::vector<float> mem(static_cast<size_t>(NT) * NS, 0.f);
    for (int t = 0; t < NT; ++t) {
        int centre = BASE + offsets[static_cast<size_t>(t)];
        for (int s = 0; s < NS; ++s) {
            int d = std::abs(s - centre);
            if (d < WIDTH)
                mem[static_cast<size_t>(t) * NS + s] =
                    -(1.0f - static_cast<float>(d) / WIDTH);
        }
    }
    TrsFile f;
    f.openFromArray(mem.data(), NT, NS);

    // Without a pipeline, signed-max search can't see the (negative) trough.
    {
        AlignResult res;
        std::string err;
        bool ok = alignByPeak(&f, {}, 0, NT, 0, 0, NS, 15, /*use_abs=*/false,
                              res, noProgress(), err);
        ASSERT_TRUE(ok) << err;
        for (int t = 1; t < NT; ++t)
            EXPECT_NE(res.shifts[static_cast<size_t>(t)], offsets[static_cast<size_t>(t)]);
    }

    // With AbsTransform in the pipeline, the trough becomes the visible peak.
    {
        std::vector<std::shared_ptr<ITransform>> pipeline = {
            std::make_shared<AbsTransform>()
        };
        AlignResult res;
        std::string err;
        bool ok = alignByPeak(&f, pipeline, 0, NT, 0, 0, NS, 15, /*use_abs=*/false,
                              res, noProgress(), err);
        ASSERT_TRUE(ok) << err;
        EXPECT_EQ(res.shifts[0], 0);
        for (int t = 1; t < NT; ++t)
            EXPECT_EQ(res.shifts[static_cast<size_t>(t)], offsets[static_cast<size_t>(t)]);
    }
}

// A pipeline stage that shortens the buffer (decimation) still works: found
// positions are rescaled from the decimated domain back to raw samples, so
// shifts come out close to the true offsets (within the decimation factor's
// rounding resolution) instead of being rejected outright.
TEST(AlignByPeak, DecimatingPipelineRescalesShifts) {
    const int NT = 4, NS = 200, BASE = 100, WIDTH = 15;
    std::vector<int> offsets = {0, 10, -10, 5};
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, BASE, offsets, WIDTH);

    const int window = 5;
    std::vector<std::shared_ptr<ITransform>> pipeline = {
        std::make_shared<WindowResampleTransform>(window)
    };
    AlignResult res;
    std::string err;
    bool ok = alignByPeak(&f, pipeline, 0, NT, 0, 0, NS, 30, /*use_abs=*/true,
                          res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    EXPECT_EQ(res.shifts[0], 0);
    for (int t = 1; t < NT; ++t)
        EXPECT_NEAR(res.shifts[static_cast<size_t>(t)],
                    offsets[static_cast<size_t>(t)], window);
}

TEST(AlignByXCorr, DecimatingPipelineRescalesShifts) {
    const int NT = 4, NS = 200, BASE = 100, WIDTH = 20;
    std::vector<int> offsets = {0, 10, -10, 5};
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, BASE, offsets, WIDTH);

    const int window = 4;
    std::vector<std::shared_ptr<ITransform>> pipeline = {
        std::make_shared<WindowResampleTransform>(window)
    };
    AlignResult res;
    std::string err;
    bool ok = alignByXCorr(&f, pipeline, 0, NT, 0, /*ref_first=*/60, /*ref_num=*/80,
                           30, -2.0f, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    EXPECT_EQ(res.shifts[0], 0);
    for (int t = 1; t < NT; ++t)
        EXPECT_NEAR(res.shifts[static_cast<size_t>(t)],
                    offsets[static_cast<size_t>(t)], window);
}

// ---------------------------------------------------------------------------
// min_correlation threshold: traces with a poor match get discarded
// ---------------------------------------------------------------------------

TEST(AlignByXCorr, ScoresAndThresholdDiscardsPoorMatches) {
    // Traces 0-2 have the usual triangle bump (good match to the reference);
    // trace 3 is left flat/all-zero — a constant patch has zero variance, so
    // its NCC against any window is exactly 0.0 (see the ref_norm>0 && sq>0
    // guard in align.cpp), well below a 0.5 threshold.
    const int NT = 4, NS = 120, BASE = 60;
    std::vector<int> offsets = {0, 5, -5, 0};
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, BASE, offsets, 10);
    std::fill(mem.begin() + 3 * NS, mem.begin() + 4 * NS, 0.0f);  // flatten trace 3
    f.openFromArray(mem.data(), NT, NS);

    AlignResult res;
    std::string err;
    // No threshold: every trace (including the flat one) gets a real shift.
    bool ok = alignByXCorr(&f, {}, 0, NT, 0, 40, 40, 10, -2.0f, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    ASSERT_EQ(static_cast<int>(res.scores.size()), NT);
    EXPECT_FLOAT_EQ(res.scores[0], 1.0f);          // reference always scores 1.0
    EXPECT_GT(res.scores[1], 0.9f);                // clean triangle match
    EXPECT_FLOAT_EQ(res.scores[3], 0.0f);           // flat trace: zero-variance patch
    for (int t = 0; t < NT; ++t)
        EXPECT_NE(res.shifts[static_cast<size_t>(t)], kAlignDiscardShift);

    // With threshold 0.5: only the flat trace should be discarded.
    ok = alignByXCorr(&f, {}, 0, NT, 0, 40, 40, 10, 0.5f, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    EXPECT_NE(res.shifts[0], kAlignDiscardShift);
    EXPECT_NE(res.shifts[1], kAlignDiscardShift);
    EXPECT_NE(res.shifts[2], kAlignDiscardShift);
    EXPECT_EQ(res.shifts[3], kAlignDiscardShift);
    // The score is still reported even though the trace was discarded.
    EXPECT_FLOAT_EQ(res.scores[3], 0.0f);
}

// ---------------------------------------------------------------------------
// A pipeline stage that *expands* the sample count (STFT with heavy window
// overlap) must not overflow the internal load buffer. Regression test for a
// crash: loadProcessed() used to size its buffer to the raw sample count
// only, so an expanding transform (out_n > count) wrote past the end of it.
// ---------------------------------------------------------------------------

TEST(AlignByPeak, ExpandingStftPipelineDoesNotCrash) {
    const int NT = 4, NS = 200, BASE = 100;
    std::vector<int> offsets = {0, 5, -5, 3};
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, BASE, offsets, 10);

    // window=64, hop=4 → heavy overlap → transformedCount(40) and
    // transformedCount(any small search window) are both >> the raw input.
    std::vector<std::shared_ptr<ITransform>> pipeline = {
        std::make_shared<STFTMagnitudeTransform>(
            64, 4, STFTMagnitudeTransform::Window::Hann)
    };

    AlignResult res;
    std::string err;
    bool ok = alignByPeak(&f, pipeline, 0, NT, 0, 60, 80, 15, true,
                          res, noProgress(), err);
    // Whether or not the region is large enough for the STFT window, the
    // important thing is that it doesn't corrupt memory / crash.
    if (ok) EXPECT_EQ(static_cast<int>(res.shifts.size()), NT);
}

TEST(AlignByXCorr, ExpandingStftPipelineDoesNotCrash) {
    const int NT = 4, NS = 200, BASE = 100;
    std::vector<int> offsets = {0, 5, -5, 3};
    TrsFile f;
    std::vector<float> mem;
    makeTriangleDataset(f, mem, NT, NS, BASE, offsets, 10);

    std::vector<std::shared_ptr<ITransform>> pipeline = {
        std::make_shared<STFTMagnitudeTransform>(
            64, 4, STFTMagnitudeTransform::Window::Hann)
    };

    AlignResult res;
    std::string err;
    bool ok = alignByXCorr(&f, pipeline, 0, NT, 0, 60, 80, 15, -2.0f,
                           res, noProgress(), err);
    if (ok) EXPECT_EQ(static_cast<int>(res.shifts.size()), NT);
}
