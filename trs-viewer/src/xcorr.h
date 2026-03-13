#pragma once

#include "trs_file.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class XCorrMethod {
    Baseline,    // direct M×M: C = X_norm^T X_norm / n  (streaming outer products)
    DualMatrix,  // via n×n Gram matrix G eigendecomposition (mathematically identical)
    MPCleaned,   // same as DualMatrix but zeros eigenvalues ≤ λ+ (Marchenko-Pastur)
};

struct XCorrResult {
    std::vector<float> matrix;   // row-major M×M correlation matrix C[i*M+j]
    int32_t  M             = 0;  // downsampled sample count
    int32_t  n_traces      = 0;  // number of traces used
    XCorrMethod method     = XCorrMethod::Baseline;
    double   lambda_plus   = 0.0; // MP upper edge  λ+ = (1 + √(M/n))²
    int32_t  n_signal      = 0;   // # eigenvalues above λ+ (MPCleaned only)
};

// progress(done, total) → return false to cancel
using XCorrProgress = std::function<bool(int32_t, int32_t)>;

// Compute cross-correlation matrix.
// stride controls downsampling: M = ceil(num_samples / stride).
// Returns false and sets error on failure or cancellation.
bool computeXCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        first_sample,
    int64_t        num_samples,   // 0 = all available
    int32_t        stride,
    XCorrMethod    method,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error);
