#include <gtest/gtest.h>
#include "align.h"
#include "cpa.h"
#include "trs_file.h"
#include "processing.h"
#include "ttest_run.h"   // kTTestMaxThreads

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto noProgress() {
    return [](int32_t, int32_t) { return true; };
}

static int hammingWeight(uint8_t v) {
    int c = 0;
    for (; v; v &= v - 1) ++c;
    return c;
}

// Leakage function: HW(hypothesis XOR data[t][0])
static bool hwLeakage(const std::vector<uint8_t>& data_flat, int data_len,
                      int n_traces, int hypothesis,
                      std::vector<float>& out, std::string& /*err*/) {
    out.resize(static_cast<size_t>(n_traces));
    for (int t = 0; t < n_traces; ++t) {
        uint8_t d = data_flat[static_cast<size_t>(t) * data_len];
        uint8_t xr = static_cast<uint8_t>(hypothesis) ^ d;
        out[static_cast<size_t>(t)] = static_cast<float>(hammingWeight(xr));
    }
    return true;
}

// Build a synthetic dataset where trace[t] leaks `HW(key ^ plain[t])` at `leak_sample`.
// Fills `samples` (row-major), `data_bytes` (plaintexts), opens `file`.
static void buildHWDataset(TrsFile& file,
                            std::vector<float>& samples,
                            std::vector<uint8_t>& data_bytes,
                            int n_traces, int n_samples,
                            int leak_sample, uint8_t key_byte,
                            float leak_scale = 1.0f) {
    // Deterministic plaintexts
    std::vector<uint8_t> plaintexts(static_cast<size_t>(n_traces));
    for (int t = 0; t < n_traces; ++t)
        plaintexts[static_cast<size_t>(t)] = static_cast<uint8_t>((t * 37 + 13) & 0xFF);

    samples.assign(static_cast<size_t>(n_traces) * n_samples, 0.f);
    for (int t = 0; t < n_traces; ++t) {
        uint8_t p = plaintexts[static_cast<size_t>(t)];
        float hw = static_cast<float>(hammingWeight(key_byte ^ p));
        samples[static_cast<size_t>(t) * n_samples + leak_sample] = leak_scale * hw;
        // Tiny orthogonal signal to avoid degenerate matrices
        samples[static_cast<size_t>(t) * n_samples + (leak_sample + 1) % n_samples] =
            0.01f * static_cast<float>(t % 7);
    }

    data_bytes = plaintexts;
    file.openFromArray(samples.data(), n_traces, n_samples, "synthetic",
                       data_bytes.data(), 1);
}

// ---------------------------------------------------------------------------
// Basic: correct key recovered
// ---------------------------------------------------------------------------

TEST(ComputeCpa, CorrectKeyRecovered) {
    const uint8_t KEY = 0xAB;
    const int NT = 200, NS = 20, LS = 10;

    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, NT, NS, LS, KEY, 2.0f);

    CpaResult res;
    std::string err;
    bool ok = computeCpa(&file, 0, NT, 0, 0, 256, {}, {}, hwLeakage,
                         res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    EXPECT_EQ(res.n_hypotheses, 256);
    EXPECT_EQ(res.n_samples,    NS);

    // At leak_sample, the correct key should have max |correlation|
    float best_corr = 0.f;
    int   best_hyp  = -1;
    for (int h = 0; h < 256; ++h) {
        float c = std::abs(res.corr[static_cast<size_t>(h) * res.n_samples + LS]);
        if (c > best_corr) { best_corr = c; best_hyp = h; }
    }
    // HW symmetry: HW(KEY^p) and HW((KEY^0xFF)^p) = 8 - HW(KEY^p) have equal |corr|.
    // Accept either the correct key or its bitwise complement as best hypothesis.
    EXPECT_TRUE(best_hyp == static_cast<int>(KEY) ||
                best_hyp == static_cast<int>(static_cast<uint8_t>(KEY ^ 0xFF)));
    EXPECT_GT(best_corr, 0.5f);
}

// ---------------------------------------------------------------------------
// Result dimensions
// ---------------------------------------------------------------------------

TEST(ComputeCpa, ResultDimensions) {
    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, 50, 16, 5, 0x42, 1.0f);

    CpaResult res;
    std::string err;
    bool ok = computeCpa(&file, 0, 50, 0, 0, 32, {}, {}, hwLeakage,
                         res, noProgress(), err);
    ASSERT_TRUE(ok) << err;
    EXPECT_EQ(res.n_hypotheses, 32);
    EXPECT_EQ(res.n_samples, 16);
    EXPECT_EQ(static_cast<int>(res.corr.size()), 32 * 16);
}

// ---------------------------------------------------------------------------
// Pipeline: ScaleTransform should not change which key is best
// ---------------------------------------------------------------------------

TEST(ComputeCpa, PipelineScaleDoesNotChangeResult) {
    const uint8_t KEY = 0x55;
    const int NT = 200, NS = 20, LS = 5;

    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, NT, NS, LS, KEY, 1.0f);

    std::vector<std::shared_ptr<ITransform>> pipeline;
    pipeline.push_back(std::make_shared<ScaleTransform>(3.0f));

    CpaResult res;
    std::string err;
    bool ok = computeCpa(&file, 0, NT, 0, 0, 256, {}, pipeline, hwLeakage,
                         res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    float best_corr = 0.f;
    int   best_hyp  = -1;
    for (int h = 0; h < 256; ++h) {
        float c = std::abs(res.corr[static_cast<size_t>(h) * res.n_samples + LS]);
        if (c > best_corr) { best_corr = c; best_hyp = h; }
    }
    // HW symmetry: accept KEY or KEY^0xFF as both have equal |corr|
    EXPECT_TRUE(best_hyp == static_cast<int>(KEY) ||
                best_hyp == static_cast<int>(static_cast<uint8_t>(KEY ^ 0xFF)));
}

// ---------------------------------------------------------------------------
// Buffer-sizing regression: size-changing pipeline transforms.
//
// proc_buf must hold max(raw_ns, eff_ns) floats.  Before the fix it was
// sized to raw_ns only, causing an out-of-bounds write (caught by ASan)
// when a transform expanded the trace (STFT with small hop) or a
// downstream read accessed the original N slots after FFT shrank output.
// ---------------------------------------------------------------------------

// FFT: N samples → N/2+1 bins.  proc_buf must accommodate N for the copy.
TEST(ComputeCpa, PipelineFFTShrinkNoCrash) {
    const int NT = 100, NS = 64, LS = 10;
    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, NT, NS, LS, 0xAB, 1.0f);

    std::vector<std::shared_ptr<ITransform>> pipeline;
    pipeline.push_back(
        std::make_shared<FFTMagnitudeTransform>(FFTMagnitudeTransform::Window::Rectangular));

    CpaResult res;
    std::string err;
    bool ok = computeCpa(&file, 0, NT, 0, 0, 16, {}, pipeline, hwLeakage,
                         res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    // Result should span the FFT output: NS/2+1 bins
    EXPECT_EQ(res.n_samples, NS / 2 + 1);
    EXPECT_EQ(res.n_hypotheses, 16);
    EXPECT_EQ(static_cast<int>(res.corr.size()), 16 * (NS / 2 + 1));
}

// STFT with tiny hop: N=64, W=8, H=1 → 285 output samples > 64 input.
// proc_buf must be sized to 285, not 64.
TEST(ComputeCpa, PipelineSTFTExpandNoCrash) {
    const int NT = 100, NS = 64, LS = 5;
    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, NT, NS, LS, 0x55, 1.0f);

    auto stft = std::make_shared<STFTMagnitudeTransform>(
        8, 1, STFTMagnitudeTransform::Window::Rectangular);
    const int64_t expected_ns = stft->transformedCount(NS);
    ASSERT_GT(expected_ns, NS) << "test precondition: STFT must expand output";

    std::vector<std::shared_ptr<ITransform>> pipeline;
    pipeline.push_back(stft);

    CpaResult res;
    std::string err;
    bool ok = computeCpa(&file, 0, NT, 0, 0, 16, {}, pipeline, hwLeakage,
                         res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    EXPECT_EQ(res.n_samples, static_cast<int>(expected_ns));
    EXPECT_EQ(res.n_hypotheses, 16);
}

// ---------------------------------------------------------------------------
// Cancellation: progress callback returning false stops computation
// ---------------------------------------------------------------------------

TEST(ComputeCpa, CancellationNoCrash) {
    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, 200, 20, 5, 0x11, 1.0f);

    int call_count = 0;
    auto cancel_early = [&](int32_t done, int32_t) -> bool {
        ++call_count;
        return done == 0;
    };

    CpaResult res;
    std::string err;
    // Should not crash regardless of whether it returns true or false
    computeCpa(&file, 0, 200, 0, 0, 256, {}, {}, hwLeakage,
               res, cancel_early, err);
}

// ---------------------------------------------------------------------------
// Zero hypotheses → no output samples in result
// ---------------------------------------------------------------------------

TEST(ComputeCpa, ZeroHypothesesGivesEmptyResult) {
    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, 10, 8, 3, 0x01, 1.0f);

    CpaResult res;
    std::string err;
    // 0 hypotheses: either returns false or produces an empty result — should not crash
    computeCpa(&file, 0, 10, 0, 0, 0, {}, {}, hwLeakage, res, noProgress(), err);
    EXPECT_EQ(res.n_hypotheses, 0);
}

// ---------------------------------------------------------------------------
// Subset of traces
// ---------------------------------------------------------------------------

TEST(ComputeCpa, SubsetOfTraces) {
    const uint8_t KEY = 0xCD;
    const int NT = 300, NS = 20, LS = 8;

    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, NT, NS, LS, KEY, 2.0f);

    CpaResult res;
    std::string err;
    bool ok = computeCpa(&file, 50, 200, 0, 0, 256, {}, {}, hwLeakage,
                         res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    float best_corr = 0.f;
    int   best_hyp  = -1;
    for (int h = 0; h < 256; ++h) {
        float c = std::abs(res.corr[static_cast<size_t>(h) * res.n_samples + LS]);
        if (c > best_corr) { best_corr = c; best_hyp = h; }
    }
    // HW symmetry: accept KEY or KEY^0xFF as both have equal |corr|
    EXPECT_TRUE(best_hyp == static_cast<int>(KEY) ||
                best_hyp == static_cast<int>(static_cast<uint8_t>(KEY ^ 0xFF)));
}

// ---------------------------------------------------------------------------
// kAlignDiscardShift: a discarded trace is excluded entirely, not zero-shifted
// ---------------------------------------------------------------------------

TEST(ComputeCpa, DiscardedTraceExcludedFromStats) {
    const uint8_t KEY = 0xAB;
    const int NT = 200, NT_BAD = 20, NS = 20, LS = 10;

    TrsFile file;
    std::vector<float> samples;
    std::vector<uint8_t> data_bytes;
    buildHWDataset(file, samples, data_bytes, NT, NS, LS, KEY, 2.0f);

    // Reference: CPA over the NT clean traces alone.
    CpaResult ref;
    std::string err;
    bool ok = computeCpa(&file, 0, NT, 0, 0, 256, {}, {}, hwLeakage,
                         ref, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    // Append NT_BAD traces with a huge, key-unrelated spike at the leak
    // sample — if these leaked into the stats they'd swamp the real signal.
    samples.resize(samples.size() + static_cast<size_t>(NT_BAD) * NS, 0.f);
    data_bytes.resize(data_bytes.size() + static_cast<size_t>(NT_BAD), 0);
    for (int t = 0; t < NT_BAD; ++t) {
        samples[static_cast<size_t>(NT + t) * NS + LS] = 1e6f;
        data_bytes[static_cast<size_t>(NT + t)] = static_cast<uint8_t>(t);
    }
    TrsFile file2;
    file2.openFromArray(samples.data(), NT + NT_BAD, NS, "synthetic",
                        data_bytes.data(), 1);

    std::vector<int32_t> shifts(NT + NT_BAD, 0);
    for (int t = 0; t < NT_BAD; ++t) shifts[static_cast<size_t>(NT + t)] = kAlignDiscardShift;

    CpaResult res;
    ok = computeCpa(&file2, 0, NT + NT_BAD, 0, 0, 256, shifts, {}, hwLeakage,
                    res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    EXPECT_EQ(res.n_traces, NT);
    ASSERT_EQ(res.corr.size(), ref.corr.size());
    for (size_t i = 0; i < res.corr.size(); ++i)
        EXPECT_NEAR(res.corr[i], ref.corr[i], 1e-5f) << "at index " << i;
}

// ---------------------------------------------------------------------------
// correlateCpa: blocks, peaks and the strongest curves only
// ---------------------------------------------------------------------------

namespace {

// Noise everywhere, HW leakage at two samples (one strong, one weaker), so
// curves differ from hypothesis to hypothesis at many samples.
void buildNoisyDataset(TrsFile& file, std::vector<float>& samples, std::vector<uint8_t>& data,
                       int nt, int ns, uint8_t key) {
    samples.assign(static_cast<size_t>(nt) * ns, 0.0f);
    data.assign(static_cast<size_t>(nt), 0);
    uint32_t x = 12345;
    auto rnd = [&]() { x = x * 1664525u + 1013904223u; return static_cast<float>(x >> 8) / 16777216.0f - 0.5f; };
    for (int t = 0; t < nt; ++t) {
        const uint8_t p = static_cast<uint8_t>((t * 73 + 5) & 0xFF);
        data[static_cast<size_t>(t)] = p;
        const float hw = static_cast<float>(hammingWeight(static_cast<uint8_t>(key ^ p)));
        for (int s = 0; s < ns; ++s) samples[static_cast<size_t>(t) * ns + s] = 3.0f * rnd() + 100.0f;
        samples[static_cast<size_t>(t) * ns + ns / 3]     += 2.0f * hw;
        samples[static_cast<size_t>(t) * ns + 2 * ns / 3] += 0.7f * hw;
    }
    file.openFromArray(samples.data(), nt, ns, "noisy", data.data(), 1);
}

std::vector<float> leakagesFor(TrsFile& file, const std::vector<int32_t>& traces, int m) {
    std::vector<float> l;
    std::string err;
    EXPECT_TRUE(evaluateCpaLeakages(&file, 0, traces, m, hwLeakage, l, nullptr, err)) << err;
    return l;
}

}  // namespace

TEST(CorrelateCpa, ManySmallBlocksMatchOneBlock) {
    const int NT = 300, NS = 1000, M = 32;
    TrsFile file; std::vector<float> s; std::vector<uint8_t> d;
    buildNoisyDataset(file, s, d, NT, NS, 0x2C);
    const auto traces = cpaTraces(NT, {});
    const auto l = leakagesFor(file, traces, M);
    for (const auto& pipeline : {std::vector<std::shared_ptr<ITransform>>{},
                                 std::vector<std::shared_ptr<ITransform>>{std::make_shared<MovingAverageTransform>(8)}}) {
        CpaResult one, many;
        std::string err;
        ASSERT_TRUE(correlateCpa(&file, 0, traces, 0, 0, {}, pipeline, l, M, M, one, nullptr, err)) << err;
        // About 60 samples per block: 17 blocks, each with its own margin.
        ASSERT_TRUE(correlateCpa(&file, 0, traces, 0, 0, {}, pipeline, l, M, M, many, nullptr, err,
                                 60 * (4 * (NT + M) + 16 + 4 * kTTestMaxThreads))) << err;
        ASSERT_EQ(one.corr.size(), many.corr.size());
        for (size_t i = 0; i < one.corr.size(); ++i)
            ASSERT_NEAR(one.corr[i], many.corr[i], 2e-5f) << "at " << i << " with " << pipeline.size() << " stages";
    }
}

TEST(CorrelateCpa, PeaksAreEachCurvesLargestMagnitude) {
    const int NT = 300, NS = 400, M = 64;
    TrsFile file; std::vector<float> s; std::vector<uint8_t> d;
    buildNoisyDataset(file, s, d, NT, NS, 0x31);
    const auto traces = cpaTraces(NT, {});
    CpaResult r;
    std::string err;
    ASSERT_TRUE(correlateCpa(&file, 0, traces, 0, 0, {}, {}, leakagesFor(file, traces, M), M, M, r, nullptr, err));
    ASSERT_TRUE(r.allCurves());
    for (int h = 0; h < M; ++h) {
        const float* row = r.corr.data() + static_cast<size_t>(h) * NS;
        const int best = static_cast<int>(std::max_element(row, row + NS, [](float a, float b) {
                             return std::abs(a) < std::abs(b); }) - row);
        EXPECT_EQ(r.peak_sample[static_cast<size_t>(h)], best) << "h" << h;
        EXPECT_FLOAT_EQ(r.peak_r[static_cast<size_t>(h)], row[best]) << "h" << h;
    }
    EXPECT_EQ(r.peak_sample[0x31], NS / 3);
}

TEST(CorrelateCpa, StrongestCurvesOnlyMatchTheFullResult) {
    const int NT = 300, NS = 600, M = 256, K = 5;
    TrsFile file; std::vector<float> s; std::vector<uint8_t> d;
    buildNoisyDataset(file, s, d, NT, NS, 0x5E);
    const auto traces = cpaTraces(NT, {});
    const auto l = leakagesFor(file, traces, M);
    CpaResult full, top;
    std::string err;
    ASSERT_TRUE(correlateCpa(&file, 0, traces, 0, 0, {}, {}, l, M, M, full, nullptr, err));
    // Small blocks too, so both passes go through several.
    ASSERT_TRUE(correlateCpa(&file, 0, traces, 0, 0, {}, {}, l, M, K, top, nullptr, err,
                             100 * (4 * (NT + M) + 16 + 4 * kTTestMaxThreads)));
    EXPECT_FALSE(top.allCurves());
    ASSERT_EQ(top.curve_hyp.size(), static_cast<size_t>(K));
    ASSERT_EQ(top.corr.size(), static_cast<size_t>(K) * NS);
    // Peaks of every hypothesis are kept either way.
    for (int h = 0; h < M; ++h) {
        EXPECT_NEAR(top.peak_r[static_cast<size_t>(h)], full.peak_r[static_cast<size_t>(h)], 2e-5f);
        EXPECT_EQ(top.peak_sample[static_cast<size_t>(h)], full.peak_sample[static_cast<size_t>(h)]);
    }
    // The curves kept are the K largest peaks, best first, and equal the full
    // ones. (With a HW model a guess and its complement have exactly opposite
    // correlations, so peaks come in equal pairs: compare magnitudes, not ids.)
    std::vector<float> mags(M);
    for (int h = 0; h < M; ++h) mags[static_cast<size_t>(h)] = std::abs(full.peak_r[static_cast<size_t>(h)]);
    std::vector<float> sorted = mags;
    std::sort(sorted.rbegin(), sorted.rend());
    EXPECT_TRUE(top.curve_hyp[0] == 0x5E || top.curve_hyp[0] == (0x5E ^ 0xFF));
    for (int j = 0; j < K; ++j) {
        const int h = top.curve_hyp[static_cast<size_t>(j)];
        EXPECT_NEAR(mags[static_cast<size_t>(h)], sorted[static_cast<size_t>(j)], 2e-5f) << "rank " << j;
        for (int i = 0; i < NS; ++i)
            ASSERT_NEAR(top.corr[static_cast<size_t>(j) * NS + i], full.corr[static_cast<size_t>(h) * NS + i], 2e-5f);
    }
}

TEST(CorrelateCpa, EstimateGrowsWithTheCurvesKept) {
    EXPECT_GT(estimateCpaBytes(1500, 256, 62500000, 256), int64_t(60) << 30);   // every curve: > 60 GB
    EXPECT_LT(estimateCpaBytes(1500, 256, 62500000, 10), int64_t(4) << 30);     // ten: a few GB
}
