#pragma once

#include "align.h"
#include "processing.h"
#include "trs_file.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

enum class XCorrMethod {
    Baseline,      // direct M×M outer products
    DualMatrix,    // via n×n Gram eigendecomposition
    MPCleaned,     // same as DualMatrix but zeros eigenvalues ≤ λ+
    TwoWindow,     // rectangular search×ref template match (full C matrix)
    TemplateMatch, // fixed template, normalised cross-correlation slid across a search window
};

struct XCorrResult {
    std::vector<float> matrix;   // row-major M×M correlation matrix C[i*M+j]
    int32_t  M             = 0;  // downsampled sample count
    int32_t  rows          = 0;  // output rows  (= M for square; search M for TwoWindow; n_traces for TemplateMatch)
    int32_t  cols          = 0;  // output cols  (= M for square; ref M for TwoWindow; n_lags for TemplateMatch)
    int32_t  n_traces      = 0;  // number of traces used
    XCorrMethod method     = XCorrMethod::Baseline;
    double   lambda_plus        = 0.0; // MP upper edge  λ+ = (1 + √(n/M))²
    double   mp_threshold_scale = 1.0; // multiplier applied to λ+ (MPCleaned only)
    int32_t  n_signal      = 0;   // # eigenvalues above threshold (MPCleaned only)

    // TemplateMatch only:
    int64_t  tm_template_len        = 0; // template length after pipeline (samples)
    int64_t  tm_search_first_sample = 0; // absolute sample index where lag 0 starts
    int32_t  tm_lag_stride          = 1; // samples between consecutive lag columns
};

// progress(done, total) → return false to cancel
using XCorrProgress = std::function<bool(int32_t, int32_t)>;

// Compute cross-correlation matrix.
// stride controls downsampling after the pipeline: M = ceil(effective_n / stride).
// shifts[i] == kAlignDiscardShift excludes trace i entirely (not just a
// zero shift) — e.g. traces the align dialog marked below its correlation
// threshold. n_traces in the output reflects the count actually used.
// Returns false and sets error on failure or cancellation.
bool computeXCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        first_sample,
    int64_t        num_samples,        // 0 = all available (raw count)
    int32_t        stride,
    XCorrMethod    method,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const std::vector<int32_t>& shifts,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error,
    double         mp_threshold_scale = 1.0); // multiplier on λ+; >1 keeps fewer eigenvalues

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
    const std::vector<int32_t>& shifts,
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
    const std::vector<int32_t>& shifts,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error);

// Template matching: extract a fixed template from one trace/region, then slide
// it across the search window of each of num_traces traces, computing the
// normalised cross-correlation (NCC) at every lag. out.matrix is row-major
// n_traces × n_lags: matrix[ti * n_lags + lag].
// template_trace is an absolute trace index (need not be inside [first_trace, first_trace+num_traces)).
// lag_stride: step in samples between consecutive lag columns (>=1; 1 = every position).
bool computeTemplateMatch(
    TrsFile*       file,
    int32_t        template_trace,
    int64_t        template_first_sample,
    int64_t        template_num_samples,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        search_first_sample,
    int64_t        search_num_samples,
    int32_t        lag_stride,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const std::vector<int32_t>& shifts,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error);
