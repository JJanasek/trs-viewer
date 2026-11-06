#pragma once
// Leakage detection with distance correlation (dcor.h), fixed vs random:
//
//  - per sample (computeDcorCurve): a curve like the t-test's, but sensitive
//    to any difference between the groups' distributions at that sample —
//    1st, 2nd or higher order — in one test;
//  - per window (computeWindowDcor): one multivariate test over all samples
//    of a window or pattern together, which also sees leakage spread across
//    samples, such as masked shares handled at different times, without
//    forming pairs. It says whether a window leaks, not where in it.
//
// Both read traces as the t-test does (alignment shifts, read bounds,
// pipeline) and report z = √(n·R*² + 1), which behaves like |t| when nothing
// leaks, so the usual 4.5 threshold and Calc TH apply. Unlike the t-test,
// every trace's values for the samples being tested are held in memory at
// once (kDcorMemoryBytes).

#include "ttest_bivariate.h"   // BivariateWindow: a raw range with its own shifts

#include <QString>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

constexpr int64_t kDcorMemoryBytes = int64_t(1) << 30;

struct DcorCurveResult {
    std::vector<float> z;      // per processed sample of the range
    int64_t n0 = 0, n1 = 0;
};

// Per-sample distance correlation over `range` (raw_start/raw_len, shifts).
// The range is processed in blocks of samples that fit kDcorMemoryBytes;
// with a pipeline, each block starts a little early so filters settle.
// sanity_split: group-0 traces only, split in two random halves.
// memory_bytes: the budget for one block (tests use a small one).
bool computeDcorCurve(TrsFile* file,
                      const std::vector<std::shared_ptr<ITransform>>& pipeline,
                      int32_t first_trace, int32_t trace_count,
                      const BivariateWindow& range, GroupBy groups,
                      int64_t read_lo, int64_t read_hi, bool sanity_split,
                      const std::function<bool(int64_t, int64_t)>& progress,
                      DcorCurveResult& out, QString& warn, QString& err,
                      int64_t memory_bytes = kDcorMemoryBytes);

struct DcorWindowResult {
    std::vector<float>   z;        // per window
    std::vector<double>  r2;       // unbiased R*² per window
    std::vector<int64_t> n0, n1;   // traces per class, per window
};

// The longest window (processed samples) whose traces fit kDcorMemoryBytes.
int64_t maxDcorWindow(int64_t n_traces);

// One multivariate test per window. standardize: scale each sample to unit
// variance first, so every sample counts equally in the distances (by
// default a sample with more spread weighs more).
bool computeWindowDcor(TrsFile* file,
                       const std::vector<std::shared_ptr<ITransform>>& pipeline,
                       int32_t first_trace, int32_t trace_count,
                       const std::vector<BivariateWindow>& windows, GroupBy groups,
                       int64_t read_lo, int64_t read_hi, bool sanity_split, bool standardize,
                       const std::function<bool(int64_t, int64_t)>& progress,
                       DcorWindowResult& out, QString& warn, QString& err);
