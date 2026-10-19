#pragma once
// Parallel Welch t-test over a trace file, with optional live snapshots.
// Shared by the t-test dialog (as a background job) and the Chain
// "Run T-test" step. Kept free of widgets so it can be tested headlessly.

#include "processing.h"
#include "trs_file.h"
#include "ttest.h"

#include <QString>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// One intermediate result, published while a live run is in progress.
struct LiveTTestSnapshot {
    int64_t done  = 0;         // traces processed so far (incl. skipped ones)
    int64_t total = 0;         // traces in the run
    int64_t n0 = 0, n1 = 0;    // traces accumulated per group
    std::vector<float> tstat;  // empty while a group still has < 2 traces
    float   max_abs  = 0.0f;   // max |t| over all samples
    int64_t n_exceed = 0;      // samples with |t| > threshold
    double  traces_per_s = 0.0;
    bool    final = false;     // last snapshot of the run
};

// Optional controls for a live run. Every field may be left at its default;
// with no TTestLive at all, the run behaves exactly like a plain batch run.
struct TTestLive {
    // Processing order as a permutation of [0, count). Empty = file order.
    // See makeInterleavedOrder().
    std::vector<int32_t> order;

    // Publish a snapshot roughly every `snapshot_every` traces, but no more
    // often than every `min_interval_ms`. 0 = no snapshots. Snapshots that
    // turn out expensive (long traces) are further spaced out so they stay
    // around 10% of the run time.
    int32_t snapshot_every  = 0;
    double  min_interval_ms = 400.0;
    float   threshold       = 4.5f;   // for LiveTTestSnapshot::n_exceed

    // Called on the worker thread. Must not touch widgets directly.
    std::function<void(std::shared_ptr<const LiveTTestSnapshot>)> on_snapshot;

    // Set from any thread. stop_keep ends the run early but still returns
    // the result over the traces processed so far; paused holds the run
    // between batches (cancellation still works while paused).
    std::atomic<bool> stop_keep{false};
    std::atomic<bool> paused{false};

    // Output: true when the run ended early through stop_keep.
    bool    stopped_early = false;
    int64_t processed     = 0;
};

// A seeded random permutation of [0, n). Processing traces in this order
// keeps intermediate results unbiased when the file is sorted by group, and
// leaves the final result unchanged (up to floating-point summation order).
std::vector<int32_t> makeInterleavedOrder(int32_t n, uint64_t seed);

// max |t| and the number of samples with |t| > threshold.
void summarizeTStat(const std::vector<float>& t, float threshold,
                    float& max_abs, int64_t& n_exceed);

// Per-trace hooks into computeTTestCore (called on worker threads; must be
// thread-safe and must not touch widgets).
struct TTestHooks {
    // Map a trace's group (0/1 from its data byte) to the group it is
    // accumulated into; return -1 to leave the trace out.
    std::function<int(int group, int32_t trace)> group_map;
    // Transform the processed samples of one trace before accumulation.
    std::function<void(int group, float* buf, int64_t n)> post;
};

// shifts empty => no alignment; shift == kAlignDiscardShift excludes that
// trace. Returns false on failure: err == "Cancelled." for a user cancel,
// otherwise a real failure message from TTestAccumulator::compute().
bool computeTTestCore(TrsFile* file,
                      const std::vector<std::shared_ptr<ITransform>>& pipeline,
                      int32_t eff_first, int32_t eff_count,
                      int64_t eff_first_sample, int64_t eff_n_samples,
                      int32_t byte_idx, const std::vector<int32_t>& shifts,
                      int32_t shifts_first_trace, int64_t read_lo, int64_t read_hi,
                      bool abs_value,
                      const std::function<bool(int64_t, int64_t)>& progress,
                      std::shared_ptr<TTestAccumulator>& acc_out,
                      std::vector<float>& tstat_out, int64_t& n0_out, int64_t& n1_out,
                      QString& warn_out, QString& err,
                      TTestLive* live = nullptr,
                      const TTestHooks* hooks = nullptr);

// ---------------------------------------------------------------------------
// Higher-level runs: order, jitter tolerance and the sanity check.
// ---------------------------------------------------------------------------
struct TTestOptions {
    // 1: Welch t-test on the samples (TVLA first order).
    // 2: univariate second order — t-test on y = (x − μ_group)², i.e. on the
    //    groups' variances (Schneider & Moradi, CHES 2015). Runs two passes:
    //    pass 1 is the first-order test and supplies the group means.
    int     order = 1;
    // Order 2 only: sum y over a centred window of this many samples, so
    // residual misalignment smaller than the window still adds up. 1 = off.
    int32_t integrate = 1;
    // Random-split sanity check: use only group-0 traces, split into two
    // halves by a seeded hash of the trace index. A real setup shows no
    // leakage here; if it does, the preprocessing created it.
    bool    sanity_split = false;
};

// Runs computeTTestCore according to `opt`. For order 2, `first_order_out`
// (optional) receives the pass-1 first-order curve. `live` applies to the
// final pass.
bool computeTTest(TrsFile* file,
                  const std::vector<std::shared_ptr<ITransform>>& pipeline,
                  int32_t eff_first, int32_t eff_count,
                  int64_t eff_first_sample, int64_t eff_n_samples,
                  int32_t byte_idx, const std::vector<int32_t>& shifts,
                  int32_t shifts_first_trace, int64_t read_lo, int64_t read_hi,
                  bool abs_value, const TTestOptions& opt,
                  const std::function<bool(int64_t, int64_t)>& progress,
                  std::shared_ptr<TTestAccumulator>& acc_out,
                  std::vector<float>& tstat_out, int64_t& n0_out, int64_t& n1_out,
                  QString& warn_out, QString& err,
                  TTestLive* live = nullptr,
                  std::vector<float>* first_order_out = nullptr);
