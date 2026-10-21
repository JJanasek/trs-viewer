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

int64_t estimateTTestBytes(int64_t raw_ns, int64_t effective_ns, int order, int threads) {
    int64_t bytes = effective_ns * 4 * static_cast<int64_t>(sizeof(double))
                  + std::clamp(threads, 1, kTTestMaxThreads) * std::max(raw_ns, effective_ns)
                        * static_cast<int64_t>(sizeof(float));
    if (order >= 2)
        bytes += effective_ns * (2 * static_cast<int64_t>(sizeof(double)) + static_cast<int64_t>(sizeof(float)));
    return bytes;
}

TraceLoad loadProcessedTrace(TrsFile* file,
                             const std::vector<std::shared_ptr<ITransform>>& pipeline,
                             int32_t src_idx, int64_t first_sample, int64_t raw_ns,
                             const std::vector<int32_t>& shifts, int32_t shifts_first_trace,
                             int64_t read_lo, int64_t read_hi,
                             std::vector<float>& buf, int64_t& n_out)
{
    // Looked up by src_idx's position in the *alignment's own* range — the
    // run may start anywhere (a subset of the aligned traces).
    int32_t shift = 0;
    if (!shifts.empty()) {
        const int64_t shift_idx = static_cast<int64_t>(src_idx) - shifts_first_trace;
        if (shift_idx < 0 || shift_idx >= static_cast<int64_t>(shifts.size()))
            return TraceLoad::Discarded;   // outside the alignment's coverage
        shift = shifts[static_cast<size_t>(shift_idx)];
        if (shift == kAlignDiscardShift) return TraceLoad::Discarded;
    }
    if (static_cast<int64_t>(buf.size()) < raw_ns) buf.resize(static_cast<size_t>(raw_ns));
    const int64_t adj_start = first_sample + shift;
    std::fill(buf.begin(), buf.end(), 0.0f);
    // Clamp to read_lo/read_hi (the whole file untiled, one tile's own
    // bounds when tiled) rather than [0, num_samples) so a shift can't pull
    // in a neighboring tile's samples. Out of bounds reads as zero.
    if (adj_start < read_hi && adj_start + raw_ns > read_lo) {
        const int64_t src_start = std::max<int64_t>(read_lo, adj_start);
        const int64_t src_end   = std::min<int64_t>(read_hi, adj_start + raw_ns);
        const int64_t got = file->readSamples(src_idx, src_start, src_end - src_start,
                                              buf.data() + (src_start - adj_start));
        if (got <= 0) return TraceLoad::Skipped;
    }
    for (const auto& t : pipeline) t->reset();
    n_out = raw_ns;
    for (const auto& t : pipeline) n_out = t->apply(buf.data(), n_out, 0);
    return TraceLoad::Ok;
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
                      TTestLive* live,
                      const TTestHooks* hooks)
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

    // Accumulation is parallelised in two steps per group of traces: each
    // worker thread reads and transforms one trace into its own slot buffer
    // (with its own cloned pipeline — the shared transforms aren't safe to
    // call apply() on concurrently since they may cache internal state, e.g.
    // FFT plans), then all threads add the slots into the one accumulator,
    // each owning a range of samples. A single accumulator (4 doubles per
    // output sample) instead of one per thread keeps long traces affordable:
    // see estimateTTestBytes().
#ifdef _OPENMP
    const int n_threads = std::clamp(omp_get_max_threads(), 1, kTTestMaxThreads);
#else
    const int n_threads = 1;
#endif

    // --- Accumulation ---
    auto acc_ptr = std::make_shared<TTestAccumulator>(static_cast<int32_t>(effective_samples));
    TTestAccumulator& acc = *acc_ptr;

    std::vector<std::vector<std::shared_ptr<ITransform>>> thread_pipelines(
        static_cast<size_t>(n_threads));
    for (auto& tp : thread_pipelines)
        for (const auto& t : pipeline) tp.push_back(t->clone());
    std::vector<std::vector<float>> slot_bufs(
        static_cast<size_t>(n_threads),
        std::vector<float>(static_cast<size_t>(std::max(raw_ns, effective_samples))));
    std::vector<const float*> slot_ptrs(static_cast<size_t>(n_threads));
    for (int k = 0; k < n_threads; k++) slot_ptrs[static_cast<size_t>(k)] = slot_bufs[static_cast<size_t>(k)].data();
    std::vector<int>     slot_group(static_cast<size_t>(n_threads), -1);
    std::vector<int32_t> slot_count(static_cast<size_t>(n_threads), 0);
    std::vector<int32_t> thread_skipped(static_cast<size_t>(n_threads), 0);
    std::vector<int32_t> thread_discarded(static_cast<size_t>(n_threads), 0);

    // Live snapshots: the t-curve over everything so far is computed straight
    // from the accumulator between batches (the parallel regions are idle
    // then, so no locking), leaving the running state untouched.
    //
    // A snapshot reads 4 doubles per sample, so on long traces it is not
    // free. It is split over the worker threads, and its measured cost throttles
    // how often it runs: the next one waits until at least
    // kSnapshotCostRatio × that cost has passed since it finished, which
    // keeps snapshots at ≲10% of the run whatever snapshot_every says.
    // Snapshots cheaper than kSnapshotCostFloorMs are left to min_interval_ms
    // alone (the default 400 ms already bounds them below that share).
    const bool snapshots = live && live->snapshot_every > 0 && live->on_snapshot;
    constexpr double kSnapshotCostRatio   = 9.0;
    constexpr double kSnapshotCostFloorMs = 20.0;
    const auto run_t0 = std::chrono::steady_clock::now();
    auto last_snap_t    = run_t0;   // when the last snapshot finished
    double last_snap_ms = 0.0;      // and what it cost
    int64_t last_snap_done = 0;

    // final_tstat: the run's own final curve (abs already applied), reused
    // for the last snapshot instead of computing the same thing twice.
    auto publishSnapshot = [&](int64_t done, bool final, const std::vector<float>* final_tstat) {
        auto snap = std::make_shared<LiveTTestSnapshot>();
        snap->done  = done;
        snap->total = eff_count;
        snap->final = final;
        snap->n0 = acc.countGroup(0);
        snap->n1 = acc.countGroup(1);
        std::string cerr;
        if (final_tstat) {
            snap->tstat = *final_tstat;
            summarizeTStat(snap->tstat, live->threshold, snap->max_abs, snap->n_exceed);
        } else if (acc.compute(snap->tstat, cerr, n_threads)) {
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

    auto trace_at = [&](int32_t ti) {
        return eff_first + (use_order ? live->order[static_cast<size_t>(ti)] : ti);
    };
    int32_t prefetched_to = 0;

    for (int32_t batch_start = 0; batch_start < eff_count && !cancelled; ) {
        if (live && live->stop_keep.load()) break;
        while (live && live->paused.load() && !live->stop_keep.load()) {
            if (progress && !progress(batch_start, eff_count)) { cancelled = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (cancelled) break;

        int32_t batch_end = std::min(batch_start + batch_size, eff_count);
        const auto batch_t0 = std::chrono::steady_clock::now();

        // Start reading this batch's windows and the next batch's (assumed
        // the same size) in the background. Each range is advised once: the
        // first batch advises both, later ones only what is new.
        {
            const int32_t from = std::max(batch_start, prefetched_to);
            const int32_t to   = std::min(eff_count, batch_end + batch_size);
            for (int32_t ti = from; ti < to; ti++)
                file->willNeed(trace_at(ti), eff_first_sample, raw_ns);
            prefetched_to = std::max(prefetched_to, to);
        }

        for (int32_t chunk = batch_start; chunk < batch_end; chunk += n_threads) {
            const int m = std::min<int32_t>(n_threads, batch_end - chunk);
            #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
            for (int k = 0; k < m; k++) {
#ifdef _OPENMP
                const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
                const size_t tid = 0;
#endif
                const int32_t ti = chunk + k;
                std::vector<std::shared_ptr<ITransform>>& pipeline_local = thread_pipelines[tid];
                std::vector<float>&                      trace_buf  = slot_bufs[static_cast<size_t>(k)];
                slot_group[static_cast<size_t>(k)] = -1;

                int32_t src_idx = trace_at(ti);
                auto data_bytes = file->readData(src_idx);
                if (byte_idx >= static_cast<int32_t>(data_bytes.size())) { thread_skipped[tid]++; continue; }
                int group = (data_bytes[byte_idx] != 0) ? 1 : 0;
                if (hooks && hooks->group_map) {
                    group = hooks->group_map(group, src_idx);
                    if (group < 0) continue;   // deliberately left out (e.g. sanity split)
                }

                int64_t n_out = 0;
                const TraceLoad ld = loadProcessedTrace(file, pipeline_local, src_idx, eff_first_sample,
                                                        raw_ns, use_shifts, shifts_first_trace,
                                                        read_lo, read_hi, trace_buf, n_out);
                if (ld == TraceLoad::Discarded) { thread_discarded[tid]++; continue; }
                if (ld == TraceLoad::Skipped)   { thread_skipped[tid]++;   continue; }
                if (hooks && hooks->post) hooks->post(group, trace_buf.data(), n_out);
                slot_ptrs[static_cast<size_t>(k)]  = trace_buf.data();
                slot_group[static_cast<size_t>(k)] = group;
                slot_count[static_cast<size_t>(k)] = static_cast<int32_t>(n_out);
            }
            acc.addTraces(slot_group.data(), slot_ptrs.data(), slot_count.data(), m, n_threads);
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
            const double since_ms =
                std::chrono::duration<double, std::milli>(now - last_snap_t).count();
            const double cost_gap_ms =
                last_snap_ms >= kSnapshotCostFloorMs ? kSnapshotCostRatio * last_snap_ms : 0.0;
            if (since_ms >= std::max(live->min_interval_ms, cost_gap_ms)) {
                publishSnapshot(batch_end, false, nullptr);
                last_snap_t    = std::chrono::steady_clock::now();
                last_snap_ms   = std::chrono::duration<double, std::milli>(last_snap_t - now).count();
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

    int32_t skipped = 0, discarded_by_align = 0;
    for (int t = 0; t < n_threads; t++) {
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
    if (!acc.compute(tstat, cerr, n_threads)) {
        err = QString::fromStdString(cerr);
        return false;
    }
    if (abs_value)
        for (float& v : tstat) v = std::fabs(v);

    if (snapshots) publishSnapshot(processed, true, &tstat);

    acc_out   = acc_ptr;
    tstat_out = std::move(tstat);
    n0_out    = acc.countGroup(0);
    n1_out    = acc.countGroup(1);
    return true;
}

// ---------------------------------------------------------------------------
// computeTTest: order / jitter window / sanity split on top of the core.
// ---------------------------------------------------------------------------

namespace {
// SplitMix64 — a well-mixed, platform-independent hash for the sanity split.
uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
}  // namespace

int sanitySplitGroup(int group, int32_t trace) {
    if (group != 0) return -1;
    return static_cast<int>(splitmix(static_cast<uint64_t>(trace) ^ 0x5A17ULL) & 1);
}

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
                  TTestLive* live, std::vector<float>* first_order_out)
{
    TTestHooks hooks;
    if (opt.sanity_split) {
        hooks.group_map = sanitySplitGroup;
    }

    if (opt.order <= 1) {
        return computeTTestCore(file, pipeline, eff_first, eff_count, eff_first_sample,
                                eff_n_samples, byte_idx, shifts, shifts_first_trace,
                                read_lo, read_hi, abs_value, progress, acc_out, tstat_out,
                                n0_out, n1_out, warn_out, err, live, &hooks);
    }

    // ---- order 2, pass 1: first-order test, which gives the group means ----
    auto half = [&](int64_t base) {
        return [&progress, base](int64_t done, int64_t total) {
            return !progress || progress(base * total + done, 2 * total);
        };
    };
    std::shared_ptr<TTestAccumulator> acc1;
    std::vector<float> t1;
    int64_t a0 = 0, a1 = 0;
    QString w1;
    if (!computeTTestCore(file, pipeline, eff_first, eff_count, eff_first_sample,
                          eff_n_samples, byte_idx, shifts, shifts_first_trace,
                          read_lo, read_hi, /*abs=*/false, half(0), acc1, t1, a0, a1, w1, err,
                          nullptr, &hooks))
        return false;
    auto mean0 = std::make_shared<std::vector<double>>();
    auto mean1 = std::make_shared<std::vector<double>>();
    acc1->groupMeans(0, *mean0);
    acc1->groupMeans(1, *mean1);
    acc1.reset();   // pass 2 has its own; don't hold both
    if (first_order_out) {
        if (abs_value) for (float& v : t1) v = std::fabs(v);
        *first_order_out = std::move(t1);
    }

    // ---- pass 2: t-test on the centred squares (optionally windowed) ----
    const int32_t half_w = std::max<int32_t>(0, opt.integrate / 2);
    hooks.post = [mean0, mean1, half_w](int group, float* buf, int64_t n) {
        const std::vector<double>& mu = (group == 1) ? *mean1 : *mean0;
        const int64_t m = std::min<int64_t>(n, static_cast<int64_t>(mu.size()));
        if (half_w == 0) {
            for (int64_t i = 0; i < m; i++) {
                const double d = buf[i] - mu[static_cast<size_t>(i)];
                buf[i] = static_cast<float>(d * d);
            }
            return;
        }
        // Centred window sum, in place: buf[i + half_w] is still the input
        // when buf[i] is written, and a small ring keeps the squares that
        // leave the window (no per-trace copy of the whole trace).
        const size_t w = static_cast<size_t>(2 * half_w + 1);
        thread_local std::vector<double> ring;
        ring.assign(w, 0.0);
        auto sq = [&](int64_t i) {
            const double d = buf[i] - mu[static_cast<size_t>(i)];
            return d * d;
        };
        double sum = 0.0;
        for (int64_t i = 0; i < std::min<int64_t>(half_w, m); i++) {
            const double v = sq(i);
            ring[static_cast<size_t>(i) % w] = v;
            sum += v;
        }
        for (int64_t i = 0; i < m; i++) {
            // out and in share a ring slot (they are w apart): read out first.
            const int64_t out = i - half_w - 1;   // leaves the window
            if (out >= 0) sum -= ring[static_cast<size_t>(out) % w];
            const int64_t in = i + half_w;        // enters it
            if (in < m) {
                const double v = sq(in);
                ring[static_cast<size_t>(in) % w] = v;
                sum += v;
            }
            buf[i] = static_cast<float>(std::max(0.0, sum));
        }
    };
    if (!computeTTestCore(file, pipeline, eff_first, eff_count, eff_first_sample,
                          eff_n_samples, byte_idx, shifts, shifts_first_trace,
                          read_lo, read_hi, abs_value, half(1), acc_out, tstat_out,
                          n0_out, n1_out, warn_out, err, live, &hooks))
        return false;
    return true;
}
