#include "ttest_bivariate.h"
#include "align.h"
#include "ttest.h"
#include "ttest_run.h"

#include <QStringList>

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

int64_t processedLength(const std::vector<std::shared_ptr<ITransform>>& pipeline, int64_t raw_len) {
    int64_t n = raw_len;
    for (const auto& t : pipeline) n = t->transformedCount(n);
    return n;
}

namespace {

// Pass 2 holds this many centred traces at once (each cell's sums are then
// updated once per batch): up to 64, within kBatchBytes.
constexpr int64_t kBatchBytes = int64_t(256) << 20;
int batchTraces(int64_t floats_per_trace) {
    return static_cast<int>(std::clamp<int64_t>(
        kBatchBytes / std::max<int64_t>(1, floats_per_trace * static_cast<int64_t>(sizeof(float))), 1, 64));
}

// The accumulator, the batch of centred windows, and a raw read buffer per thread.
int64_t passBytes(int64_t cells, int64_t raw_a, int64_t len_a, int64_t raw_b, int64_t len_b,
                  bool sym, int threads) {
    const int64_t per_trace = len_a + (sym ? 0 : len_b);
    const int64_t raw = std::max({raw_a, len_a, sym ? 0 : raw_b, sym ? 0 : len_b});
    return cells * 4 * static_cast<int64_t>(sizeof(double))
         + batchTraces(per_trace) * per_trace * static_cast<int64_t>(sizeof(float))
         + std::clamp(threads, 1, kTTestMaxThreads) * raw * static_cast<int64_t>(sizeof(float));
}

// Whether trace src_idx has a usable shift in window w (always, without shifts).
bool usable(const BivariateWindow& w, int32_t src_idx) {
    if (w.shifts.empty()) return true;
    const int64_t i = static_cast<int64_t>(src_idx) - w.shifts_first_trace;
    return i >= 0 && i < static_cast<int64_t>(w.shifts.size())
        && w.shifts[static_cast<size_t>(i)] != kAlignDiscardShift;
}

bool sameWindow(const BivariateWindow& a, const BivariateWindow& b) {
    return a.raw_start == b.raw_start && a.raw_len == b.raw_len
        && a.shifts == b.shifts && a.shifts_first_trace == b.shifts_first_trace;
}

}  // namespace

int64_t estimateBivariateBytes(int64_t rows, int64_t cols, int threads) {
    // Raw ≈ processed here; the caller passes processed lengths.
    return passBytes(rows * cols, rows, rows, cols, cols, false, threads);
}

namespace {

// The two passes shared by the A × B and the fixed-offset test.
//   Pass 1: group means of each window (a first-order run with its own shifts).
//   Pass 2: batches of traces are loaded and centred — ca (rows), cb (cols),
//           zero past what the pipeline produced — and `product(ca, cb, lo,
//           hi, y)` writes their cells [lo, hi) into y on the fly, straight
//           into the accumulator (TTestAccumulator::addGenerated).
// A trace takes part only if both windows can use it, so both passes (and
// both windows' means) see the same traces. sym: A and B are the same
// window (loaded once, cb == ca).
using ProductFn = std::function<void(const float* ca, const float* cb, int32_t lo, int32_t hi, float* y)>;

bool centredProductPasses(TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
                          int32_t first_trace, int32_t trace_count, int32_t byte_idx,
                          const BivariateWindow& a, const BivariateWindow& b, bool sym,
                          int64_t read_lo, int64_t read_hi, bool sanity_split,
                          const std::function<bool(int64_t, int64_t)>& progress,
                          int64_t cells, const ProductFn& product,
                          TTestAccumulator& acc, int& n_threads_out, int32_t& skipped_out,
                          QString& err)
{
    const int64_t rows = processedLength(pipeline, a.raw_len);
    const int64_t cols = processedLength(pipeline, b.raw_len);

    // Progress: pass 1 over A (and B unless the same), then pass 2 (weighted 2×).
    const int64_t n_pass1 = sym ? 1 : 2;
    const int64_t total   = (n_pass1 + 2) * trace_count;
    auto stage = [&](int64_t base, int64_t weight) {
        return [&progress, base, weight, total, trace_count](int64_t done, int64_t) {
            return !progress || progress(base * trace_count + weight * done, total);
        };
    };

    TTestHooks hooks;
    hooks.group_map = [&a, &b, sanity_split](int group, int32_t trace) {
        if (!usable(a, trace) || !usable(b, trace)) return -1;
        return sanity_split ? sanitySplitGroup(group, trace) : group;
    };

    // ---- Pass 1 ----
    std::vector<double> ma0, ma1, mb0, mb1;
    {
        auto run = [&](const BivariateWindow& w, int64_t base, std::vector<double>& m0, std::vector<double>& m1) {
            std::shared_ptr<TTestAccumulator> acc1;
            std::vector<float> t;
            int64_t n0 = 0, n1 = 0;
            QString w_warn;
            if (!computeTTestCore(file, pipeline, first_trace, trace_count, w.raw_start, w.raw_len,
                                  byte_idx, w.shifts, w.shifts_first_trace, read_lo, read_hi, false,
                                  stage(base, 1), acc1, t, n0, n1, w_warn, err, nullptr, &hooks)) {
                if (err != "Cancelled.") err = QString("Window %1: %2").arg(w.name, err);
                return false;
            }
            acc1->groupMeans(0, m0);
            acc1->groupMeans(1, m1);
            return true;
        };
        if (!run(a, 0, ma0, ma1)) return false;
        if (sym) { mb0 = ma0; mb1 = ma1; }
        else if (!run(b, 1, mb0, mb1)) return false;
    }

    // ---- Pass 2 ----
#ifdef _OPENMP
    const int n_threads = std::clamp(omp_get_max_threads(), 1, kTTestMaxThreads);
#else
    const int n_threads = 1;
#endif
    n_threads_out = n_threads;
    std::vector<std::vector<std::shared_ptr<ITransform>>> thread_pipelines(static_cast<size_t>(n_threads));
    for (auto& tp : thread_pipelines)
        for (const auto& t : pipeline) tp.push_back(t->clone());
    const size_t buf_raw = static_cast<size_t>(std::max({a.raw_len, rows, sym ? 0 : b.raw_len, sym ? 0 : cols}));
    std::vector<std::vector<float>> thread_raw(static_cast<size_t>(n_threads), std::vector<float>(buf_raw));
    std::vector<int32_t> thread_skipped(static_cast<size_t>(n_threads), 0);

    const int64_t per_trace = rows + (sym ? 0 : cols);
    const int batch = batchTraces(per_trace);
    std::vector<float> centred(static_cast<size_t>(batch) * static_cast<size_t>(per_trace));
    std::vector<int>   batch_group(static_cast<size_t>(batch), -1);
    auto ca_of = [&](int k) { return centred.data() + static_cast<size_t>(k) * static_cast<size_t>(per_trace); };
    auto cb_of = [&](int k) { return sym ? ca_of(k) : ca_of(k) + rows; };

    // Loads window w of trace src and writes it, centred on the group's
    // means, to dst (len values).
    auto loadCentred = [&](size_t tid, const BivariateWindow& w, int32_t src, int64_t len,
                           const std::vector<double>& mu, float* dst) {
        std::vector<float>& x = thread_raw[tid];
        int64_t n = 0;
        if (loadProcessedTrace(file, thread_pipelines[tid], src, w.raw_start, w.raw_len, w.shifts,
                               w.shifts_first_trace, read_lo, read_hi, x, n) != TraceLoad::Ok)
            return false;
        for (int64_t i = 0; i < len; i++)
            dst[i] = i < n ? static_cast<float>(x[static_cast<size_t>(i)] - mu[static_cast<size_t>(i)]) : 0.0f;
        return true;
    };

    const int64_t base2 = n_pass1;
    for (int32_t chunk = 0; chunk < trace_count; chunk += batch) {
        const int m = std::min<int32_t>(batch, trace_count - chunk);
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int k = 0; k < m; k++) {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            batch_group[static_cast<size_t>(k)] = -1;
            const int32_t src = first_trace + chunk + k;
            const auto data = file->readData(src);
            if (byte_idx >= static_cast<int32_t>(data.size())) { thread_skipped[tid]++; continue; }
            const int group = hooks.group_map(data[static_cast<size_t>(byte_idx)] != 0 ? 1 : 0, src);
            if (group < 0) continue;
            if (!loadCentred(tid, a, src, rows, group ? ma1 : ma0, ca_of(k))
                || (!sym && !loadCentred(tid, b, src, cols, group ? mb1 : mb0, cb_of(k)))) {
                thread_skipped[tid]++; continue;
            }
            batch_group[static_cast<size_t>(k)] = group;
        }
        acc.addGenerated(batch_group.data(), m, n_threads,
                         [&](int k, int32_t lo, int32_t hi, float* y) { product(ca_of(k), cb_of(k), lo, hi, y); });
        if (progress && !progress((base2 * trace_count) + 2 * (chunk + m), total)) {
            err = "Cancelled.";
            return false;
        }
    }
    skipped_out = 0;
    for (int32_t s : thread_skipped) skipped_out += s;
    return true;
}

// The run's warnings: unreadable traces and traces without a usable shift.
QString passWarnings(int32_t trace_count, int64_t n0, int64_t n1, int32_t skipped,
                     bool sanity_split, const QString& where) {
    const int64_t left_out = static_cast<int64_t>(trace_count) - n0 - n1 - skipped;
    QStringList msg;
    if (skipped > 0) msg << QString("%1 traces skipped (data byte out of range or unreadable).").arg(skipped);
    if (left_out > 0 && !sanity_split)
        msg << QString("%1 traces excluded (no usable shift in %2).").arg(left_out).arg(where);
    return msg.join("\n");
}

}  // namespace

bool computeBivariateTTest(TrsFile* file,
                           const std::vector<std::shared_ptr<ITransform>>& pipeline,
                           int32_t first_trace, int32_t trace_count, int32_t byte_idx,
                           const BivariateWindow& a, const BivariateWindow& b,
                           int64_t read_lo, int64_t read_hi,
                           bool sanity_split, bool abs_value,
                           const std::function<bool(int64_t, int64_t)>& progress,
                           BivariateResult& out, QString& warn, QString& err)
{
    out = BivariateResult{};
    const bool sym = sameWindow(a, b);
    const int64_t rows = processedLength(pipeline, a.raw_len);
    const int64_t cols = processedLength(pipeline, b.raw_len);
    if (rows <= 0 || cols <= 0) { err = "A window is empty."; return false; }
    if (rows * cols > kBivariateMaxPairs) {
        err = QString("%1 × %2 = %3 pairs is too many (at most %4). Shorten the windows or "
                      "add a resampling stage to the pipeline.")
                  .arg(rows).arg(cols).arg(rows * cols).arg(kBivariateMaxPairs);
        return false;
    }
    const int32_t R = static_cast<int32_t>(rows), C = static_cast<int32_t>(cols);
    const int32_t pairs = R * C;

    auto product = [C, sym](const float* ca, const float* cb, int32_t lo, int32_t hi, float* y) {
        // Cell c = i·C + j, one row segment at a time.
        for (int32_t c = lo; c < hi;) {
            const int32_t i  = c / C;
            const int32_t j0 = c - i * C;
            const int32_t j1 = std::min(C, j0 + (hi - c));
            const float ai = ca[i];
            float* out = y + (c - lo);
            for (int32_t j = j0; j < j1; j++)
                // Symmetric: the upper triangle only (mirrored at the end).
                out[j - j0] = (sym && j < i) ? 0.0f : ai * cb[j];
            c += j1 - j0;
        }
    };
    TTestAccumulator acc(pairs);
    int n_threads = 1;
    int32_t skipped = 0;
    if (!centredProductPasses(file, pipeline, first_trace, trace_count, byte_idx, a, b, sym,
                              read_lo, read_hi, sanity_split, progress, pairs, product,
                              acc, n_threads, skipped, err))
        return false;

    std::string cerr;
    if (!acc.compute(out.t, cerr, n_threads)) { err = QString::fromStdString(cerr); return false; }
    out.rows = R;
    out.cols = C;
    out.symmetric = sym;
    out.n0 = acc.countGroup(0);
    out.n1 = acc.countGroup(1);
    for (int32_t i = 0; i < R; i++)
        for (int32_t j = 0; j < C; j++) {
            float& v = out.t[static_cast<size_t>(i) * C + j];
            if (sym && j < i) v = out.t[static_cast<size_t>(j) * C + i];
            if (abs_value) v = std::fabs(v);
            if (sym && j < i) continue;   // count each pair once
            const float av = std::fabs(v);
            if (av > out.max_abs) { out.max_abs = av; out.max_row = i; out.max_col = j; }
            if (av > 4.5f) out.n_exceed++;
        }
    warn = passWarnings(trace_count, out.n0, out.n1, skipped, sanity_split,
                        QString("window %1 or %2").arg(a.name, b.name));
    return true;
}

std::vector<int32_t> bivariateLags(int32_t lag_lo, int32_t lag_hi, int32_t lag_step) {
    std::vector<int32_t> lags;
    if (lag_lo < 0 || lag_hi < lag_lo || lag_step < 1) return lags;
    for (int64_t d = lag_lo; d <= lag_hi; d += lag_step) lags.push_back(static_cast<int32_t>(d));
    return lags;
}

int64_t lagExtendedRawLength(const std::vector<std::shared_ptr<ITransform>>& pipeline,
                             int64_t raw_len, int32_t lag_hi, int64_t max_raw)
{
    const int64_t len  = processedLength(pipeline, raw_len);
    const int64_t want = len + lag_hi;
    int64_t raw = std::min(raw_len, max_raw);
    // Grow by the pipeline's ratio until the processed window covers the
    // largest offset (or the trace ends).
    for (int guard = 0; guard < 64 && raw < max_raw; guard++) {
        const int64_t have = processedLength(pipeline, raw);
        if (have >= want) break;
        const int64_t step = std::max<int64_t>(1, (want - have) * std::max<int64_t>(1, raw_len) / std::max<int64_t>(1, len));
        raw = std::min(max_raw, raw + step);
    }
    return raw;
}

int64_t estimateLagBytes(int64_t len, int64_t n_lags, int64_t lag_hi, int threads) {
    // The window read lag_hi past its end (≈ in processed samples).
    return passBytes(len * n_lags, len + lag_hi, len + lag_hi, 0, 0, true, threads);
}

bool computeLagTTest(TrsFile* file,
                     const std::vector<std::shared_ptr<ITransform>>& pipeline,
                     int32_t first_trace, int32_t trace_count, int32_t byte_idx,
                     const BivariateWindow& w, const std::vector<int32_t>& lags,
                     int64_t read_lo, int64_t read_hi,
                     bool sanity_split, bool abs_value,
                     const std::function<bool(int64_t, int64_t)>& progress,
                     LagResult& out, QString& warn, QString& err)
{
    out = LagResult{};
    const int64_t len = processedLength(pipeline, w.raw_len);
    if (len <= 0) { err = "The window is empty."; return false; }
    if (lags.empty()) { err = "No offsets to test."; return false; }
    const int64_t n_lags = static_cast<int64_t>(lags.size());
    if (len * n_lags > kLagMaxCells) {
        err = QString("%1 samples × %2 offsets = %3 values is too many (at most %4). Shorten the "
                      "range or test fewer offsets.")
                  .arg(len).arg(n_lags).arg(len * n_lags).arg(kLagMaxCells);
        return false;
    }

    // The window, extended by the largest offset, is loaded and centred
    // once; sample t is paired with t + Δ inside it.
    BivariateWindow ext = w;
    ext.raw_len = lagExtendedRawLength(pipeline, w.raw_len, *std::max_element(lags.begin(), lags.end()),
                                       std::max<int64_t>(w.raw_len, read_hi - w.raw_start));
    const int64_t ext_len = processedLength(pipeline, ext.raw_len);
    const int32_t L = static_cast<int32_t>(len);
    const int64_t cells = len * n_lags;

    auto product = [&lags, L, ext_len](const float* c, const float*, int32_t lo, int32_t hi, float* y) {
        // Cell = d·L + t, one offset's segment at a time.
        for (int32_t cell = lo; cell < hi;) {
            const int32_t d  = cell / L;
            const int32_t t0 = cell - d * L;
            const int32_t t1 = std::min(L, t0 + (hi - cell));
            const int32_t lag = lags[static_cast<size_t>(d)];
            // Past the end of the trace: no pair (t = 0 for that sample).
            const int32_t n = static_cast<int32_t>(std::clamp<int64_t>(ext_len - lag, t0, t1));
            float* out = y + (cell - lo) - t0;
            for (int32_t t = t0; t < n; t++) out[t] = c[t] * c[t + lag];
            for (int32_t t = n; t < t1; t++) out[t] = 0.0f;
            cell += t1 - t0;
        }
    };
    TTestAccumulator acc(static_cast<int32_t>(cells));
    int n_threads = 1;
    int32_t skipped = 0;
    if (!centredProductPasses(file, pipeline, first_trace, trace_count, byte_idx, ext, ext, true,
                              read_lo, read_hi, sanity_split, progress, cells, product,
                              acc, n_threads, skipped, err))
        return false;

    std::string cerr;
    if (!acc.compute(out.t, cerr, n_threads)) { err = QString::fromStdString(cerr); return false; }
    out.lags = lags;
    out.len  = L;
    out.n0   = acc.countGroup(0);
    out.n1   = acc.countGroup(1);
    out.lag_max_abs.assign(lags.size(), 0.0f);
    for (size_t d = 0; d < lags.size(); d++)
        for (int32_t t = 0; t < L; t++) {
            float& v = out.t[d * static_cast<size_t>(L) + static_cast<size_t>(t)];
            if (abs_value) v = std::fabs(v);
            const float av = std::fabs(v);
            if (av > out.lag_max_abs[d]) out.lag_max_abs[d] = av;
            if (av > out.max_abs) { out.max_abs = av; out.max_lag = static_cast<int32_t>(d); out.max_t = t; }
            if (av > 4.5f) out.n_exceed++;
        }
    out.best_lag.assign(static_cast<size_t>(L), 0);
    out.best.assign(static_cast<size_t>(L), 0.0f);
    std::vector<int64_t> cell(static_cast<size_t>(L));
    for (int32_t t = 0; t < L; t++) {
        size_t bd = 0;
        for (size_t d = 1; d < lags.size(); d++)
            if (std::fabs(out.t[d * L + t]) > std::fabs(out.t[bd * L + t])) bd = d;
        out.best_lag[static_cast<size_t>(t)] = static_cast<int32_t>(bd);
        out.best[static_cast<size_t>(t)]     = out.t[bd * L + t];
        cell[static_cast<size_t>(t)]         = static_cast<int64_t>(bd) * L + t;
    }
    out.best_acc = std::make_shared<TTestAccumulator>(acc.gather(cell));
    warn = passWarnings(trace_count, out.n0, out.n1, skipped, sanity_split,
                        QString("window %1").arg(w.name));
    return true;
}
