#pragma once
// Bivariate second-order t-test: leakage that only shows when two points in
// time are combined — a masked value whose shares are handled at different
// times (software masking). For every pair (i in window A, j in window B)
// each trace gives
//
//     y_ij = (x_i − μ_g,i) · (x_j − μ_g,j)        g = the trace's group
//
// and a Welch t-test runs on y_ij (Schneider & Moradi, CHES 2015, §4). The
// group means come from a first pass over both windows. The result is a
// rows × cols map of t-values; with A = B it is symmetric and its diagonal
// is the univariate 2nd-order test.
//
// The pairs grow as |A|·|B|, so the windows must be short — a pattern on a
// resampled pipeline, or a few thousand samples (kBivariateMaxPairs).

#include "processing.h"
#include "trs_file.h"

#include <QString>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct BivariateWindow {
    QString name;                    // for messages and labels
    int64_t raw_start = 0;           // raw samples
    int64_t raw_len   = 0;
    std::vector<int32_t> shifts;     // empty = none; kAlignDiscardShift excludes a trace
    int32_t shifts_first_trace = 0;
};

struct BivariateResult {
    std::vector<float> t;            // row-major: t[i * cols + j], i in A, j in B
    int32_t rows = 0, cols = 0;      // processed lengths of A and B
    int64_t n0 = 0, n1 = 0;
    bool    symmetric = false;       // A and B were the same window
    float   max_abs = 0.0f;
    int32_t max_row = -1, max_col = -1;
    int64_t n_exceed = 0;            // pairs with |t| > 4.5 (counted once if symmetric)
};

// At most this many pairs (4096 × 4096).
constexpr int64_t kBivariateMaxPairs = int64_t(1) << 24;

// Processed length of a window after `pipeline`.
int64_t processedLength(const std::vector<std::shared_ptr<ITransform>>& pipeline, int64_t raw_len);

// Peak memory: the accumulator (4 doubles per pair) and one pair buffer per
// worker thread (threads capped as in the 1st-order run).
int64_t estimateBivariateBytes(int64_t rows, int64_t cols, int threads);

// sanity_split: group-0 traces only, split in two random halves (see
// TTestOptions). Returns false with err == "Cancelled." on cancel.
bool computeBivariateTTest(TrsFile* file,
                           const std::vector<std::shared_ptr<ITransform>>& pipeline,
                           int32_t first_trace, int32_t trace_count, int32_t byte_idx,
                           const BivariateWindow& a, const BivariateWindow& b,
                           int64_t read_lo, int64_t read_hi,
                           bool sanity_split, bool abs_value,
                           const std::function<bool(int64_t, int64_t)>& progress,
                           BivariateResult& out, QString& warn, QString& err);
