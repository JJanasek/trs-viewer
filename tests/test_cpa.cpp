#include <gtest/gtest.h>
#include "cpa.h"
#include "trs_file.h"
#include "processing.h"

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
