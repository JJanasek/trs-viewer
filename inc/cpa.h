#pragma once

#include "trs_file.h"
#include "processing.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct CpaResult {
    std::vector<float> corr;    // row-major M × n_samples: corr[hyp * n_samples + s]
    int32_t n_hypotheses = 0;
    int32_t n_samples    = 0;
    int32_t n_traces     = 0;
};

// Leakage callback: fills leakages_out[0..n_traces-1] for hypothesis h.
// data_flat is n_traces × data_len bytes of per-trace auxiliary data (row-major).
// Returns false on error (sets error string).
using LeakageFn = std::function<bool(
    const std::vector<uint8_t>& data_flat, int data_len,
    int n_traces, int hypothesis,
    std::vector<float>& leakages_out, std::string& error)>;

// Compute CPA: for each hypothesis h in [0, n_hypotheses), call leakage_fn(h)
// to get a model vector, then compute Pearson correlation with each sample column.
// pipeline is applied per-trace (decimation supported).
// shifts[i]: sample offset added when reading trace i (alignment compensation);
//            positive = read from later samples (trace shifts left); empty = no shifts.
// progress(done, total) → return false to cancel.
bool computeCpa(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        n_traces,
    int64_t        first_sample,
    int64_t        n_samples,                   // 0 = all available
    int32_t        n_hypotheses,                // M — user-configurable
    const std::vector<int32_t>& shifts,         // per-trace sample shifts (may be empty)
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const LeakageFn& leakage_fn,
    CpaResult&     out,
    std::function<bool(int32_t, int32_t)> progress,
    std::string&   error);
