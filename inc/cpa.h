#pragma once

#include "align.h"
#include "trs_file.h"
#include "processing.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct CpaResult {
    int32_t n_hypotheses = 0;
    int32_t n_samples    = 0;   // samples per curve (after the pipeline)
    int32_t n_traces     = 0;   // traces actually used
    // Correlation curves, row-major, one row per entry of curve_hyp. When
    // they all fit (allCurves()) that is every hypothesis in order, so
    // corr[h * n_samples + s]; otherwise only the strongest few, best first.
    std::vector<float>   corr;
    std::vector<int32_t> curve_hyp;
    // Per hypothesis: the correlation of largest magnitude (signed), and its sample.
    std::vector<float>   peak_r;
    std::vector<int32_t> peak_sample;

    bool allCurves() const { return static_cast<int32_t>(curve_hyp.size()) == n_hypotheses; }
};

// Leakage callback: fills leakages_out[0..n_traces-1] for hypothesis h.
// data_flat is n_traces × data_len bytes of per-trace auxiliary data (row-major).
// Returns false on error (sets error string).
using LeakageFn = std::function<bool(
    const std::vector<uint8_t>& data_flat, int data_len,
    int n_traces, int hypothesis,
    std::vector<float>& leakages_out, std::string& error)>;

// Working memory of one block in correlateCpa().
constexpr int64_t kCpaBlockBytes = int64_t(512) << 20;

// Every trace's processed samples [p0, p0 + len) into rows of out (N × len);
// L is the pipeline's output length for raw_ns raw samples. progress(traces
// done) → false to cancel.
bool loadSampleBlock(TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
                     int32_t first_trace, const std::vector<int32_t>& traces,
                     const std::vector<int32_t>& shifts, int64_t first_sample, int64_t raw_ns,
                     int64_t L, int64_t p0, int64_t len, float* out,
                     const std::function<bool(int64_t)>& progress);

// The traces of [0, n_traces) to use: all but those whose shift is
// kAlignDiscardShift (e.g. below the alignment's correlation threshold).
std::vector<int32_t> cpaTraces(int32_t n_traces, const std::vector<int32_t>& shifts);

// The model's leakage for every hypothesis and every trace in `traces`
// (offsets from first_trace): row-major n_hypotheses × traces.size().
// Calls leakage_fn on the calling thread — the GUI thread for a Python model.
bool evaluateCpaLeakages(TrsFile* file, int32_t first_trace, const std::vector<int32_t>& traces,
                         int32_t n_hypotheses, const LeakageFn& leakage_fn,
                         std::vector<float>& leakages,
                         const std::function<bool(int32_t, int32_t)>& progress,
                         std::string& error);

// What correlateCpa() needs, in bytes, keeping `curves` full curves.
int64_t estimateCpaBytes(int64_t n_traces, int64_t n_hypotheses, int64_t n_samples, int64_t curves);

// Pearson correlation of every hypothesis's leakages with every sample of
// [first_sample, first_sample + n_samples) (0 = to the end), after the
// pipeline. Goes through the samples in blocks of block_bytes: each trace's
// block is read once for all hypotheses and the block's correlations are one
// matrix product, so any length works.
//
// Keeps every hypothesis's peak. Full curves: all of them if max_curves >=
// n_hypotheses, otherwise the max_curves strongest by peak |r| — found in a
// first pass, their curves computed in a second.
//
// shifts[i] is added to the read position of trace first_trace + i (empty =
// none); `traces` comes from cpaTraces(). progress(done, total) → false to
// cancel. Safe on a worker thread.
bool correlateCpa(TrsFile* file, int32_t first_trace, const std::vector<int32_t>& traces,
                  int64_t first_sample, int64_t n_samples,
                  const std::vector<int32_t>& shifts,
                  const std::vector<std::shared_ptr<ITransform>>& pipeline,
                  const std::vector<float>& leakages, int32_t n_hypotheses, int32_t max_curves,
                  CpaResult& out, const std::function<bool(int64_t, int64_t)>& progress,
                  std::string& error, int64_t block_bytes = kCpaBlockBytes);

// Both steps, keeping every curve:
// for each hypothesis h in [0, n_hypotheses), call leakage_fn(h)
// to get a model vector, then compute Pearson correlation with each sample column.
// pipeline is applied per-trace (decimation supported).
// shifts[i]: sample offset added when reading trace i (alignment compensation);
//            positive = read from later samples (trace shifts left); empty = no shifts.
//            shifts[i] == kAlignDiscardShift excludes trace i entirely (not
//            just a zero shift) — e.g. traces the align dialog marked below
//            its correlation threshold. out.n_traces reflects the count
//            actually used.
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
