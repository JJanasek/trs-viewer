#include "ttest_run.h"
#include "align.h"

#include <QStringList>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<int32_t> makeInterleavedOrder(int32_t n, uint64_t seed) {
    std::vector<int32_t> order(static_cast<size_t>(std::max(0, n)));
    std::iota(order.begin(), order.end(), 0);
    // Fisher–Yates with our own index draw rather than std::shuffle, whose
    // exact sequence is implementation-defined — the same seed must give the
    // same order on every platform so a run can be reproduced.
    std::mt19937_64 rng(seed);
    for (size_t i = order.size(); i > 1; i--) {
        const size_t j = static_cast<size_t>(rng() % i);
        std::swap(order[i - 1], order[j]);
    }
    return order;
}

void summarizeTStat(const std::vector<float>& t, float threshold,
                    float& max_abs, int64_t& n_exceed) {
    max_abs  = 0.0f;
    n_exceed = 0;
    for (float v : t) {
        const float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
        if (a > threshold) n_exceed++;
    }
}

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
                      TTestLive* live)
{
    const TrsHeader& h = file->header();
    const std::vector<int32_t>& use_shifts = shifts;

    const bool use_order = live && static_cast<int32_t>(live->order.size()) == eff_count;

    // Effective raw sample count for the window
    const int64_t raw_ns = (eff_n_samples == 0)
        ? (h.num_samples - eff_first_sample)
        : std::min<int64_t>(eff_n_samples, h.num_samples - eff_first_sample);

    // Effective sample count after pipeline
    int64_t effective_samples = raw_ns;
    for (const auto& t : pipeline)
        effective_samples = t->transformedCount(effective_samples);

    // Accumulation is parallelised across traces, one TTestAccumulator + one
    // cloned pipeline per worker thread (merged together at the end) — the
    // shared pipeline transforms aren't safe to call apply() on
    // concurrently since they may cache internal state (e.g. FFT plans).
    // Capped at 8 threads since each accumulator costs 4×effective_samples
    // doubles; see confirmTTestMemory() in mainwindow.cpp.
#ifdef _OPENMP
    const int n_threads = std::clamp(omp_get_max_threads(), 1, 8);
#else
    const int n_threads = 1;
#endif

    // --- Accumulation ---
    auto acc_ptr = std::make_shared<TTestAccumulator>(static_cast<int32_t>(effective_samples));

    std::vector<TTestAccumulator> thread_accs(
        static_cast<size_t>(n_threads), TTestAccumulator(static_cast<int32_t>(effective_samples)));
    std::vector<std::vector<std::shared_ptr<ITransform>>> thread_pipelines(
        static_cast<size_t>(n_threads));
    for (auto& tp : thread_pipelines)
        for (const auto& t : pipeline) tp.push_back(t->clone());
    std::vector<std::vector<float>> thread_bufs(
        static_cast<size_t>(n_threads),
        std::vector<float>(static_cast<size_t>(std::max(raw_ns, effective_samples))));
    std::vector<int32_t> thread_skipped(static_cast<size_t>(n_threads), 0);
    std::vector<int32_t> thread_discarded(static_cast<size_t>(n_threads), 0);

    // Live snapshots: the per-thread accumulators are merged into a scratch
    // one between batches (the parallel region is idle then, so no locking),
    // which leaves the running state untouched. Only allocated when needed.
    const bool snapshots = live && live->snapshot_every > 0 && live->on_snapshot;
    std::unique_ptr<TTestAccumulator> scratch;
    if (snapshots)
        scratch = std::make_unique<TTestAccumulator>(static_cast<int32_t>(effective_samples));
    const auto run_t0 = std::chrono::steady_clock::now();
    auto last_snap_t  = run_t0;
    int64_t last_snap_done = 0;

    auto publishSnapshot = [&](int64_t done, bool final) {
        auto snap = std::make_shared<LiveTTestSnapshot>();
        snap->done  = done;
        snap->total = eff_count;
        snap->final = final;
        scratch->reset();
        for (const auto& a : thread_accs) scratch->mergeFrom(a);
        snap->n0 = scratch->countGroup(0);
        snap->n1 = scratch->countGroup(1);
        std::string cerr;
        if (scratch->compute(snap->tstat, cerr)) {
            if (abs_value)
                for (float& v : snap->tstat) v = std::fabs(v);
            summarizeTStat(snap->tstat, live->threshold, snap->max_abs, snap->n_exceed);
        } else {
            snap->tstat.clear();   // a group still has < 2 traces: "waiting"
        }
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - run_t0).count();
        snap->traces_per_s = secs > 0.0 ? static_cast<double>(done) / secs : 0.0;
        live->on_snapshot(std::move(snap));
    };

    // Process in batches so the GUI thread can update the progress bar and
    // check for cancellation between parallel regions (Qt calls are only
    // ever made from the main thread, never from inside the omp for below).
    //
    // The batch size adapts to wall-clock time instead of being a fixed trace
    // count. Between parallel regions is also the only point where this
    // process can answer the window manager's "still alive?" ping, and GNOME
    // raises its "Application is not responding — Wait / Force Quit" dialog
    // when a reply takes more than a few seconds. No fixed count suits both a
    // 1K-sample trace and a 5M-sample one — the same 140-trace batch is
    // milliseconds of work for one and minutes for the other — so aim for
    // roughly kTargetMs of work per batch and let the trace count follow.
    constexpr double kTargetMs = 120.0;
    int32_t batch_size = std::clamp(eff_count / 20, 1, 32);
    // Live runs also cap the batch at the snapshot interval, so snapshots
    // land close to their requested trace counts.
    const int32_t max_batch = snapshots ? std::clamp(live->snapshot_every, 1, 256) : 256;
    batch_size = std::min(batch_size, max_batch);
    bool cancelled = false;
    int32_t processed = 0;

    for (int32_t batch_start = 0; batch_start < eff_count && !cancelled; ) {
        if (live && live->stop_keep.load()) break;
        while (live && live->paused.load() && !live->stop_keep.load()) {
            if (progress && !progress(batch_start, eff_count)) { cancelled = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (cancelled) break;

        int32_t batch_end = std::min(batch_start + batch_size, eff_count);
        const auto batch_t0 = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int32_t ti = batch_start; ti < batch_end; ti++) {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            TTestAccumulator&                        acc_local  = thread_accs[tid];
            std::vector<std::shared_ptr<ITransform>>& pipeline_local = thread_pipelines[tid];
            std::vector<float>&                      trace_buf  = thread_bufs[tid];

            int32_t src_idx = eff_first + (use_order ? live->order[static_cast<size_t>(ti)] : ti);
            auto data_bytes = file->readData(src_idx);
            if (byte_idx >= static_cast<int32_t>(data_bytes.size())) { thread_skipped[tid]++; continue; }
            int group = (data_bytes[byte_idx] != 0) ? 1 : 0;

            // Read window with per-trace shift, zero-pad out of bounds.
            // Looked up by src_idx's position in the *alignment's own*
            // range, not by ti — eff_first may start anywhere (a subset of
            // the aligned traces), so ti==0 does not generally mean
            // "the first aligned trace".
            int32_t shift = 0;
            if (!use_shifts.empty()) {
                int64_t shift_idx = static_cast<int64_t>(src_idx) - shifts_first_trace;
                if (shift_idx < 0 || shift_idx >= static_cast<int64_t>(use_shifts.size())) {
                    thread_discarded[tid]++; continue;   // outside the alignment's coverage
                }
                shift = use_shifts[static_cast<size_t>(shift_idx)];
                if (shift == kAlignDiscardShift) { thread_discarded[tid]++; continue; }
            }
            const int64_t adj_start = eff_first_sample + shift;
            std::fill(trace_buf.begin(), trace_buf.end(), 0.0f);
            // Clamp to read_lo/read_hi (the whole file untiled, one tile's
            // own bounds when tiled) rather than [0, h.num_samples) so a
            // shift can't pull in a neighboring tile's samples.
            if (adj_start < read_hi && adj_start + raw_ns > read_lo) {
                int64_t src_start = std::max<int64_t>(read_lo, adj_start);
                int64_t src_end   = std::min<int64_t>(read_hi, adj_start + raw_ns);
                int64_t dst_off   = src_start - adj_start;
                int64_t got = file->readSamples(src_idx, src_start, src_end - src_start,
                                                      trace_buf.data() + dst_off);
                if (got <= 0) { thread_skipped[tid]++; continue; }
            }
            for (const auto& t : pipeline_local) t->reset();
            int64_t n_out = raw_ns;
            for (const auto& t : pipeline_local)
                n_out = t->apply(trace_buf.data(), n_out, 0);
            acc_local.addTrace(group, trace_buf.data(), static_cast<int32_t>(n_out));
        }

        // Re-aim the next batch at kTargetMs. Clamped per step so one slow
        // batch (a cold page-cache read, say) can't collapse the size to 1 or
        // blow it up to the cap in a single jump.
        const double batch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - batch_t0).count();
        batch_size = (batch_ms > 1.0)
            ? std::clamp(static_cast<int32_t>(batch_size * std::clamp(kTargetMs / batch_ms, 0.5, 2.0)), 1, max_batch)
            : std::min(batch_size * 2, max_batch);
        batch_start = batch_end;   // the new size applies to the *next* batch
        processed   = batch_end;

        if (snapshots && batch_end < eff_count
            && batch_end - last_snap_done >= live->snapshot_every) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double, std::milli>(now - last_snap_t).count()
                    >= live->min_interval_ms) {
                publishSnapshot(batch_end, false);
                last_snap_t    = now;
                last_snap_done = batch_end;
            }
        }

        if (progress && !progress(batch_end, eff_count)) cancelled = true;
    }
    if (cancelled) { err = "Cancelled."; return false; }

    if (live) {
        live->processed     = processed;
        live->stopped_early = processed < eff_count;
    }

    TTestAccumulator& acc = *acc_ptr;
    int32_t skipped = 0, discarded_by_align = 0;
    for (int t = 0; t < n_threads; t++) {
        acc.mergeFrom(thread_accs[static_cast<size_t>(t)]);
        skipped            += thread_skipped[static_cast<size_t>(t)];
        discarded_by_align += thread_discarded[static_cast<size_t>(t)];
    }

    QStringList msg;
    if (live && live->stopped_early)
        msg << QString("Stopped early: result covers %1 of %2 traces.")
                   .arg(processed).arg(eff_count);
    if (skipped > 0)
        msg << QString("%1 traces skipped (data byte out of range).").arg(skipped);
    if (discarded_by_align > 0)
        msg << QString("%1 traces excluded (outside the alignment's range, or below its "
                       "correlation threshold).")
                   .arg(discarded_by_align);
    if (!msg.isEmpty()) warn_out = msg.join("\n");

    // --- Compute ---
    std::vector<float> tstat;
    std::string cerr;
    if (!acc.compute(tstat, cerr)) {
        err = QString::fromStdString(cerr);
        return false;
    }
    if (abs_value)
        for (float& v : tstat) v = std::fabs(v);

    if (snapshots) publishSnapshot(processed, true);

    acc_out   = acc_ptr;
    tstat_out = std::move(tstat);
    n0_out    = acc.countGroup(0);
    n1_out    = acc.countGroup(1);
    return true;
}
