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

int64_t estimateBivariateBytes(int64_t rows, int64_t cols, int threads) {
    const int64_t pairs = rows * cols;
    return pairs * 4 * static_cast<int64_t>(sizeof(double))
         + std::clamp(threads, 1, kTTestMaxThreads) * pairs * static_cast<int64_t>(sizeof(float));
}

namespace {

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

    // Progress: pass 1 over A (and B unless the same), then pass 2 (weighted 2×).
    const int64_t n_pass1 = sym ? 1 : 2;
    const int64_t total   = (n_pass1 + 2) * trace_count;
    auto stage = [&](int64_t base, int64_t weight) {
        return [&progress, base, weight, total, trace_count](int64_t done, int64_t) {
            return !progress || progress(base * trace_count + weight * done, total);
        };
    };

    // A trace takes part only if both windows can use it, so both passes
    // (and both windows' means) see the same traces.
    TTestHooks hooks;
    hooks.group_map = [&a, &b, sanity_split](int group, int32_t trace) {
        if (!usable(a, trace) || !usable(b, trace)) return -1;
        return sanity_split ? sanitySplitGroup(group, trace) : group;
    };

    // ---- Pass 1: group means of each window (a first-order run, own shifts) ----
    std::vector<double> ma0, ma1, mb0, mb1;
    {
        auto run = [&](const BivariateWindow& w, int64_t base, std::vector<double>& m0, std::vector<double>& m1) {
            std::shared_ptr<TTestAccumulator> acc;
            std::vector<float> t;
            int64_t n0 = 0, n1 = 0;
            QString w_warn;
            if (!computeTTestCore(file, pipeline, first_trace, trace_count, w.raw_start, w.raw_len,
                                  byte_idx, w.shifts, w.shifts_first_trace, read_lo, read_hi, false,
                                  stage(base, 1), acc, t, n0, n1, w_warn, err, nullptr, &hooks)) {
                if (err != "Cancelled.") err = QString("Window %1: %2").arg(w.name, err);
                return false;
            }
            acc->groupMeans(0, m0);
            acc->groupMeans(1, m1);
            return true;
        };
        if (!run(a, 0, ma0, ma1)) return false;
        if (sym) { mb0 = ma0; mb1 = ma1; }
        else if (!run(b, 1, mb0, mb1)) return false;
    }

    // ---- Pass 2: centred products, accumulated per pair ----
#ifdef _OPENMP
    const int n_threads = std::clamp(omp_get_max_threads(), 1, kTTestMaxThreads);
#else
    const int n_threads = 1;
#endif
    TTestAccumulator acc(pairs);
    std::vector<std::vector<std::shared_ptr<ITransform>>> thread_pipelines(static_cast<size_t>(n_threads));
    for (auto& tp : thread_pipelines)
        for (const auto& t : pipeline) tp.push_back(t->clone());
    const size_t buf_a = static_cast<size_t>(std::max<int64_t>(a.raw_len, rows));
    const size_t buf_b = static_cast<size_t>(std::max<int64_t>(b.raw_len, cols));
    std::vector<std::vector<float>> thread_a(static_cast<size_t>(n_threads), std::vector<float>(buf_a));
    std::vector<std::vector<float>> thread_b(static_cast<size_t>(n_threads), std::vector<float>(buf_b));
    std::vector<std::vector<float>> slot_prod(static_cast<size_t>(n_threads),
                                              std::vector<float>(static_cast<size_t>(pairs)));
    std::vector<const float*> slot_ptrs(static_cast<size_t>(n_threads));
    std::vector<int>     slot_group(static_cast<size_t>(n_threads), -1);
    std::vector<int32_t> slot_count(static_cast<size_t>(n_threads), pairs);
    for (int k = 0; k < n_threads; k++) slot_ptrs[static_cast<size_t>(k)] = slot_prod[static_cast<size_t>(k)].data();
    std::vector<int32_t> thread_skipped(static_cast<size_t>(n_threads), 0);

    const int64_t base2 = n_pass1;
    for (int32_t chunk = 0; chunk < trace_count; chunk += n_threads) {
        const int m = std::min<int32_t>(n_threads, trace_count - chunk);
        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int k = 0; k < m; k++) {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            slot_group[static_cast<size_t>(k)] = -1;
            const int32_t src = first_trace + chunk + k;
            const auto data = file->readData(src);
            if (byte_idx >= static_cast<int32_t>(data.size())) { thread_skipped[tid]++; continue; }
            const int group = hooks.group_map(data[static_cast<size_t>(byte_idx)] != 0 ? 1 : 0, src);
            if (group < 0) continue;

            auto& pl = thread_pipelines[tid];
            std::vector<float>& xa = thread_a[tid];
            std::vector<float>& xb = thread_b[tid];
            int64_t na = 0, nb = 0;
            if (loadProcessedTrace(file, pl, src, a.raw_start, a.raw_len, a.shifts,
                                   a.shifts_first_trace, read_lo, read_hi, xa, na) != TraceLoad::Ok) {
                thread_skipped[tid]++; continue;
            }
            const std::vector<double>& mua = group ? ma1 : ma0;
            for (int32_t i = 0; i < R; i++)
                xa[static_cast<size_t>(i)] = i < na ? static_cast<float>(xa[static_cast<size_t>(i)] - mua[static_cast<size_t>(i)]) : 0.0f;
            const float* cb = xa.data();
            if (!sym) {
                if (loadProcessedTrace(file, pl, src, b.raw_start, b.raw_len, b.shifts,
                                       b.shifts_first_trace, read_lo, read_hi, xb, nb) != TraceLoad::Ok) {
                    thread_skipped[tid]++; continue;
                }
                const std::vector<double>& mub = group ? mb1 : mb0;
                for (int32_t j = 0; j < C; j++)
                    xb[static_cast<size_t>(j)] = j < nb ? static_cast<float>(xb[static_cast<size_t>(j)] - mub[static_cast<size_t>(j)]) : 0.0f;
                cb = xb.data();
            }
            float* y = slot_prod[static_cast<size_t>(k)].data();
            const float* ca = xa.data();
            for (int32_t i = 0; i < R; i++) {
                float* row = y + static_cast<size_t>(i) * C;
                const float ai = ca[i];
                // Symmetric: the upper triangle only (mirrored at the end).
                const int32_t j0 = sym ? i : 0;
                if (sym) std::fill(row, row + j0, 0.0f);
                for (int32_t j = j0; j < C; j++) row[j] = ai * cb[j];
            }
            slot_group[static_cast<size_t>(k)] = group;
        }
        acc.addTraces(slot_group.data(), slot_ptrs.data(), slot_count.data(), m, n_threads);
        if (progress && !progress((base2 * trace_count) + 2 * (chunk + m), total)) {
            err = "Cancelled.";
            return false;
        }
    }

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

    int32_t skipped = 0;
    for (int32_t s : thread_skipped) skipped += s;
    const int64_t left_out = static_cast<int64_t>(trace_count) - out.n0 - out.n1 - skipped;
    QStringList msg;
    if (skipped > 0) msg << QString("%1 traces skipped (data byte out of range or unreadable).").arg(skipped);
    if (left_out > 0 && !sanity_split)
        msg << QString("%1 traces excluded (no usable shift in window %2 or %3).")
                   .arg(left_out).arg(a.name, b.name);
    warn = msg.join("\n");
    return true;
}
