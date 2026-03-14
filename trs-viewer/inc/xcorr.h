#pragma once

#include "processing.h"
#include "trs_file.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

enum class XCorrMethod {
    Baseline,    // direct M×M outer products
    DualMatrix,  // via n×n Gram eigendecomposition
    MPCleaned,   // same as DualMatrix but zeros eigenvalues ≤ λ+
    TwoWindow,   // rectangular search×ref template match
};

struct XCorrResult {
    std::vector<float> matrix;   // row-major M×M correlation matrix C[i*M+j]
    int32_t  M             = 0;  // downsampled sample count
    int32_t  rows          = 0;  // output rows  (= M for square; search M for TwoWindow)
    int32_t  cols          = 0;  // output cols  (= M for square; ref M for TwoWindow)
    int32_t  n_traces      = 0;  // number of traces used
    XCorrMethod method     = XCorrMethod::Baseline;
    double   lambda_plus   = 0.0; // MP upper edge  λ+ = (1 + √(M/n))²
    int32_t  n_signal      = 0;   // # eigenvalues above λ+ (MPCleaned only)
};

// progress(done, total) → return false to cancel
using XCorrProgress = std::function<bool(int32_t, int32_t)>;

// Compute cross-correlation matrix.
// stride controls downsampling after the pipeline: M = ceil(effective_n / stride).
// Returns false and sets error on failure or cancellation.
bool computeXCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        first_sample,
    int64_t        num_samples,   // 0 = all available (raw count)
    int32_t        stride,
    XCorrMethod    method,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error);

// Naive reference implementation: no Eigen, no BLAS, no Welford — purely scalar
// double loops for debugging correlation regressions.  Same signature as computeXCorr
// but without the method parameter (always produces an M×M matrix).
bool computeXCorrNaive(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        first_sample,
    int64_t        num_samples,
    int32_t        stride,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error);

// Compute a rectangular (search × ref) normalised cross-correlation matrix.
// ref window:    [ref_first_sample, ref_first_sample + ref_num_samples)
// search window: [search_first_sample, search_first_sample + search_num_samples)
// stride applies to both windows.  out.matrix is row-major: C[s_row * ref_cols + r_col].
bool computeTwoWindowCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int64_t        search_first_sample,
    int64_t        search_num_samples,
    int32_t        stride,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error);
