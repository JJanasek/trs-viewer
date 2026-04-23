#include <gtest/gtest.h>
#include "align.h"
#include "trs_file.h"

#include <cmath>
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
    bool ok = alignByPeak(&f, 0, NT, 0, 0, NS, 15, true, res, noProgress(), err);
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
    bool ok = alignByPeak(&f, 0, NT, 0, 0, NS, 10, true, res, noProgress(), err);
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
    bool ok = alignByPeak(&f, 0, NT, 1, 0, NS, 10, true, res, noProgress(), err);
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
    bool ok = alignByPeak(&f, 0, NT, NT, 0, NS, 5, true, res, noProgress(), err);
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
    bool ok = alignByXCorr(&f, 0, NT, 0, /*ref_first=*/40, /*ref_num=*/40, 10,
                           res, noProgress(), err);
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
    bool ok = alignByXCorr(&f, 0, NT, 0, 20, 40, 5, res, noProgress(), err);
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
    alignByPeak(&f, 0, NT, 0, 0, NS, 5, true, res, cancel, err);
    // No assertion — just verify no crash/hang
}
