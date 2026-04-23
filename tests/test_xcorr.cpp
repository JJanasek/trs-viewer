#include <gtest/gtest.h>
#include "xcorr.h"
#include "trs_file.h"
#include "processing.h"

#include <memory>

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto noProgress() {
    return [](int32_t, int32_t) { return true; };
}

// All traces are scalar multiples of a single ramp pattern.
// mem must outlive the TrsFile.
static void makePropDataset(TrsFile& file, std::vector<float>& mem,
                             int n_traces, int n_samples,
                             const std::vector<float>& scales) {
    std::vector<float> pattern(static_cast<size_t>(n_samples));
    for (int s = 0; s < n_samples; ++s)
        pattern[static_cast<size_t>(s)] = static_cast<float>(s + 1);

    mem.resize(static_cast<size_t>(n_traces) * n_samples);
    for (int t = 0; t < n_traces; ++t)
        for (int s = 0; s < n_samples; ++s)
            mem[static_cast<size_t>(t) * n_samples + s] =
                scales[static_cast<size_t>(t)] * pattern[static_cast<size_t>(s)];

    file.openFromArray(mem.data(), n_traces, n_samples);
}

// ---------------------------------------------------------------------------
// Basic: output dimensions
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, OutputDimensions) {
    const int NT = 20, NS = 32;
    std::vector<float> scales(NT, 1.f);
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    EXPECT_EQ(res.rows, NS);
    EXPECT_EQ(res.cols, NS);
    EXPECT_EQ(static_cast<int>(res.matrix.size()), NS * NS);
    EXPECT_EQ(res.n_traces, NT);
}

// ---------------------------------------------------------------------------
// Proportional traces (varying scale) → all |C[i,j]| ≈ 1
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, PerfectlyCorrelatedTraces) {
    const int NT = 15, NS = 16;
    // Use varying scales so sample columns have non-zero variance across traces.
    // All columns are proportional to the scale vector → |C[i,j]| = 1.
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t) scales[static_cast<size_t>(t)] = 1.0f + 0.5f * t;
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    const int M = res.rows;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            EXPECT_NEAR(std::abs(res.matrix[static_cast<size_t>(i) * M + j]), 1.0f, 0.02f)
                << " at (" << i << "," << j << ")";
}

// ---------------------------------------------------------------------------
// Diagonal = 1.0 (auto-correlation)
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, DiagonalIsOne) {
    const int NT = 20, NS = 16;
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t) scales[static_cast<size_t>(t)] = 1.0f + 0.1f * t;
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    for (int i = 0; i < res.rows; ++i)
        EXPECT_NEAR(res.matrix[static_cast<size_t>(i) * res.cols + i], 1.0f, 0.02f);
}

// ---------------------------------------------------------------------------
// Symmetry: C[i,j] = C[j,i]
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, MatrixIsSymmetric) {
    const int NT = 20, NS = 12;
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t) scales[static_cast<size_t>(t)] = static_cast<float>(t + 1);
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    const int M = res.rows;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            EXPECT_NEAR(res.matrix[static_cast<size_t>(i) * M + j],
                        res.matrix[static_cast<size_t>(j) * M + i], 1e-4f);
}

// ---------------------------------------------------------------------------
// Anti-correlated traces → |C[i,j]| ≈ 1 (just different sign)
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, AntiCorrelatedAbsIsOne) {
    const int NT = 16, NS = 8;
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t)
        scales[static_cast<size_t>(t)] = (t % 2 == 0) ? 1.0f : -1.0f;
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    const int M = res.rows;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            EXPECT_NEAR(std::abs(res.matrix[static_cast<size_t>(i) * M + j]), 1.0f, 0.02f);
}

// ---------------------------------------------------------------------------
// Stride decimation reduces output size
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, StrideReducesSize) {
    const int NT = 20, NS = 32, STRIDE = 4;
    std::vector<float> scales(NT, 1.f);
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, STRIDE, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    EXPECT_EQ(res.rows, (NS + STRIDE - 1) / STRIDE);
    EXPECT_EQ(res.cols, res.rows);
}

// ---------------------------------------------------------------------------
// Too few traces → error
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, ZeroTracesReturnsError) {
    TrsFile f;
    std::vector<float> mem(16, 0.f);
    f.openFromArray(mem.data(), 1, 16);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, 0, 0, 0, 1, XCorrMethod::Baseline,
                           {}, {}, res, noProgress(), err);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(err.empty());
}

// ---------------------------------------------------------------------------
// Naive vs Baseline: results should match
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, NaiveMatchesBaseline) {
    const int NT = 15, NS = 8;
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t) scales[static_cast<size_t>(t)] = 0.5f + 0.3f * (t % 5);
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    XCorrResult res_baseline, res_naive;
    std::string err;
    ASSERT_TRUE(computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                              {}, {}, res_baseline, noProgress(), err)) << err;
    ASSERT_TRUE(computeXCorrNaive(&f, 0, NT, 0, 0, 1,
                                  {}, {}, res_naive, noProgress(), err)) << err;

    ASSERT_EQ(res_baseline.matrix.size(), res_naive.matrix.size());
    for (size_t i = 0; i < res_baseline.matrix.size(); ++i)
        EXPECT_NEAR(res_baseline.matrix[i], res_naive.matrix[i], 0.02f) << " index " << i;
}

// ---------------------------------------------------------------------------
// Pipeline: ScaleTransform should not change correlation values
// ---------------------------------------------------------------------------

TEST(ComputeXCorr, PipelineScaleInvariant) {
    const int NT = 20, NS = 8;
    std::vector<float> scales(NT, 1.f);
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    std::vector<std::shared_ptr<ITransform>> pipeline;
    pipeline.push_back(std::make_shared<ScaleTransform>(10.0f));

    XCorrResult res_plain, res_scaled;
    std::string err;
    ASSERT_TRUE(computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                              {}, {}, res_plain, noProgress(), err)) << err;
    ASSERT_TRUE(computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                              pipeline, {}, res_scaled, noProgress(), err)) << err;

    for (size_t i = 0; i < res_plain.matrix.size(); ++i)
        EXPECT_NEAR(res_plain.matrix[i], res_scaled.matrix[i], 0.02f);
}

// ---------------------------------------------------------------------------
// Buffer-sizing regression: size-changing pipeline transforms.
//
// Before the fix, trace_buf was allocated as raw_ns floats.  When the
// pipeline shrank the output (FFT: N → N/2+1) or expanded it (STFT with
// tiny hop), accessing out-of-bounds memory caused silent corruption or
// an ASan crash.  These tests would be caught by running under ASan
// (cmake -DENABLE_SANITIZERS=ON).
// ---------------------------------------------------------------------------

// FFT: output is N/2+1 < N — buffer must still hold N for the initial read.
TEST(ComputeXCorr, PipelineFFTShrinkNoCrash) {
    const int NT = 20, NS = 64;
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t) scales[static_cast<size_t>(t)] = 1.0f + 0.3f * t;
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    std::vector<std::shared_ptr<ITransform>> pipeline;
    pipeline.push_back(
        std::make_shared<FFTMagnitudeTransform>(FFTMagnitudeTransform::Window::Hann));

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           pipeline, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    // Output should be NS/2+1 samples wide
    const int64_t expected_m = NS / 2 + 1;
    EXPECT_EQ(res.rows, expected_m);
    EXPECT_EQ(res.cols, expected_m);
    EXPECT_EQ(static_cast<int64_t>(res.matrix.size()), expected_m * expected_m);
}

// STFT with small hop: output can be much LARGER than input.
// e.g. N=64, W=8, H=1 → (64-8)/1+1=57 windows × 5 bins = 285 > 64
// buffer must be at least 285 (not just 64).
TEST(ComputeXCorr, PipelineSTFTExpandNoCrash) {
    const int NT = 20, NS = 64;
    std::vector<float> scales(NT);
    for (int t = 0; t < NT; ++t) scales[static_cast<size_t>(t)] = 1.0f + 0.2f * t;
    TrsFile f;
    std::vector<float> mem;
    makePropDataset(f, mem, NT, NS, scales);

    // W=8, H=1 → 57 windows × 5 bins = 285 output samples from 64 input
    auto stft = std::make_shared<STFTMagnitudeTransform>(
        8, 1, STFTMagnitudeTransform::Window::Rectangular);
    ASSERT_GT(stft->transformedCount(NS), NS) << "test precondition: STFT must expand";

    std::vector<std::shared_ptr<ITransform>> pipeline;
    pipeline.push_back(stft);

    XCorrResult res;
    std::string err;
    bool ok = computeXCorr(&f, 0, NT, 0, 0, 1, XCorrMethod::Baseline,
                           pipeline, {}, res, noProgress(), err);
    ASSERT_TRUE(ok) << err;

    const int64_t expected_m = stft->transformedCount(NS);
    EXPECT_EQ(res.rows, expected_m);
    EXPECT_EQ(res.cols, expected_m);
}
