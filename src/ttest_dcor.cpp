#include "ttest_dcor.h"
#include "align.h"
#include "dcor.h"
#include "pattern_align.h"
#include "ttest_run.h"

#include <QStringList>

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int loadThreads() {
#ifdef _OPENMP
    return std::clamp(omp_get_max_threads(), 1, kTTestMaxThreads);
#else
    return 1;
#endif
}

int computeThreads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

bool usable(const BivariateWindow& w, int32_t src_idx) {
    if (w.shifts.empty()) return true;
    const int64_t i = static_cast<int64_t>(src_idx) - w.shifts_first_trace;
    return i >= 0 && i < static_cast<int64_t>(w.shifts.size())
        && w.shifts[static_cast<size_t>(i)] != kAlignDiscardShift;
}

// The traces of the run that take part in window w, with their classes.
struct Classes {
    std::vector<int32_t> traces;
    std::vector<uint8_t> cls;
    int64_t n0 = 0, n1 = 0;
    int32_t no_byte = 0;      // data byte out of range
    int32_t no_shift = 0;     // no usable shift in this window
};

Classes classify(TrsFile* file, int32_t first, int32_t count, int32_t byte_idx,
                 bool sanity_split, const BivariateWindow& w) {
    Classes c;
    for (int32_t t = first; t < first + count; t++) {
        if (!usable(w, t)) { c.no_shift++; continue; }
        const auto data = file->readData(t);
        if (byte_idx < 0 || byte_idx >= static_cast<int32_t>(data.size())) { c.no_byte++; continue; }
        int g = data[static_cast<size_t>(byte_idx)] != 0 ? 1 : 0;
        if (sanity_split) g = sanitySplitGroup(g, t);
        if (g < 0) continue;
        c.traces.push_back(t);
        c.cls.push_back(static_cast<uint8_t>(g));
        (g ? c.n1 : c.n0)++;
    }
    return c;
}

// Rows of `out` (len values each): processed samples [skip, skip + len) of
// the raw range [raw_start, raw_start + raw_len) of each trace, through the
// pipeline with the window's shifts; zeros where a trace runs short or fails
// to read (counted in failed). progress(k) after every batch of traces;
// false from it cancels.
bool loadRows(TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
              const std::vector<int32_t>& traces, const BivariateWindow& w,
              int64_t raw_start, int64_t raw_len, int64_t read_lo, int64_t read_hi,
              int64_t skip, int64_t len, float* out, int32_t& failed,
              const std::function<bool(int64_t)>& progress)
{
    const int threads = loadThreads();
    std::vector<std::vector<std::shared_ptr<ITransform>>> pls(static_cast<size_t>(threads));
    for (auto& pl : pls)
        for (const auto& t : pipeline) pl.push_back(t->clone());
    std::vector<std::vector<float>> bufs(static_cast<size_t>(threads));
    std::vector<int32_t> fails(static_cast<size_t>(threads), 0);
    const int64_t n = static_cast<int64_t>(traces.size());
    const int64_t batch = std::max<int64_t>(threads * 8, 64);
    for (int64_t b0 = 0; b0 < n; b0 += batch) {
        const int64_t b1 = std::min(n, b0 + batch);
        #pragma omp parallel for schedule(dynamic) num_threads(threads)
        for (int64_t k = b0; k < b1; k++) {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            float* row = out + k * len;
            int64_t n_out = 0;
            const TraceLoad ld = loadProcessedTrace(file, pls[tid], traces[static_cast<size_t>(k)],
                                                    raw_start, raw_len, w.shifts, w.shifts_first_trace,
                                                    read_lo, read_hi, bufs[tid], n_out);
            if (ld != TraceLoad::Ok) {
                fails[tid]++;
                std::fill(row, row + len, 0.0f);
                continue;
            }
            const float* src = bufs[tid].data();
            for (int64_t i = 0; i < len; i++)
                row[i] = skip + i < n_out ? src[skip + i] : 0.0f;
        }
        if (progress && !progress(b1)) return false;
    }
    failed = 0;
    for (int32_t f : fails) failed += f;
    return true;
}

QString classWarnings(const Classes& c, int32_t failed, bool sanity_split, const QString& where) {
    QStringList msg;
    if (c.no_byte > 0) msg << QString("%1 traces skipped (data byte out of range).").arg(c.no_byte);
    if (c.no_shift > 0 && !sanity_split)
        msg << QString("%1 traces excluded (no usable shift in %2).").arg(c.no_shift).arg(where);
    if (failed > 0) msg << QString("%1 traces could not be read and count as zeros.").arg(failed);
    return msg.join("\n");
}

bool enoughTraces(const Classes& c, const QString& where, QString& err) {
    if (c.n0 >= 2 && c.n1 >= 2) return true;
    err = QString("Need ≥ 2 traces per group%1 (group 0: %2, group 1: %3).")
              .arg(where.isEmpty() ? QString() : " in " + where).arg(c.n0).arg(c.n1);
    return false;
}

}  // namespace

bool computeDcorCurve(TrsFile* file,
                      const std::vector<std::shared_ptr<ITransform>>& pipeline,
                      int32_t first_trace, int32_t trace_count,
                      const BivariateWindow& range, int32_t byte_idx,
                      int64_t read_lo, int64_t read_hi, bool sanity_split,
                      const std::function<bool(int64_t, int64_t)>& progress,
                      DcorCurveResult& out, QString& warn, QString& err,
                      int64_t memory_bytes)
{
    out = DcorCurveResult{};
    const Classes c = classify(file, first_trace, trace_count, byte_idx, sanity_split, range);
    if (!enoughTraces(c, QString(), err)) return false;
    const int64_t n = static_cast<int64_t>(c.traces.size());
    const int64_t L = processedLength(pipeline, range.raw_len);
    if (L <= 0) { err = "The sample range is empty."; return false; }

    // Blocks of samples whose values for every trace fit the memory budget.
    const int64_t block = std::clamp<int64_t>(memory_bytes / (n * static_cast<int64_t>(sizeof(float))), 1, L);
    const int64_t n_blocks = (L + block - 1) / block;
    if (n_blocks > 1 && pipelineRequiresWholeTrace(pipeline)) {
        err = QString("The pipeline has a stage that needs whole traces, but %1 traces × %2 samples "
                      "don't fit in memory at once. Shorten the range to at most %3 samples (after "
                      "the pipeline), or remove that stage.").arg(n).arg(L).arg(block);
        return false;
    }
    // Raw 8-bit samples with no pipeline: integers −128…127, binned in O(n).
    const bool int8 = pipeline.empty() && file->header().sample_type == SampleType::INT8;
    // A pipeline starts each block early by this many processed samples so
    // filters settle before the block's own samples.
    constexpr int64_t kMargin = 4096;

    out.z.assign(static_cast<size_t>(L), 0.0f);
    out.n0 = c.n0;
    out.n1 = c.n1;
    std::vector<float> mat;
    const int threads = computeThreads();
    const int64_t total = n_blocks * 2 * n;
    int32_t failed_total = 0;
    for (int64_t bi = 0; bi < n_blocks; bi++) {
        const int64_t p0 = bi * block, p1 = std::min(L, p0 + block), len = p1 - p0;
        const int64_t m  = pipeline.empty() ? 0 : std::min(p0, kMargin);
        const int64_t rs = rawStartForProcessedIndex(pipeline, p0 - m, range.raw_len);
        const int64_t re = p1 >= L ? range.raw_len
                         : std::min(range.raw_len, rawCountForProcessedCount(pipeline, p1, range.raw_len));
        mat.assign(static_cast<size_t>(n * len), 0.0f);
        int32_t failed = 0;
        const int64_t base = bi * 2 * n;
        if (!loadRows(file, pipeline, c.traces, range, range.raw_start + rs, re - rs, read_lo, read_hi,
                      m, len, mat.data(), failed,
                      [&](int64_t done) { return !progress || progress(base + done, total); })) {
            err = "Cancelled.";
            return false;
        }
        failed_total += failed;

        // Each sample on its own: its column of n values and the classes.
        const int64_t chunk = std::max<int64_t>(64, len / 64);
        for (int64_t c0 = 0; c0 < len; c0 += chunk) {
            const int64_t c1 = std::min(len, c0 + chunk);
            #pragma omp parallel num_threads(threads)
            {
                std::vector<float> col(static_cast<size_t>(n));
                std::vector<std::pair<float, uint8_t>> scratch;
                std::vector<uint32_t> h0(256), h1(256);
                #pragma omp for schedule(dynamic, 16)
                for (int64_t s = c0; s < c1; s++) {
                    DistanceSums ds;
                    if (int8) {
                        std::fill(h0.begin(), h0.end(), 0u);
                        std::fill(h1.begin(), h1.end(), 0u);
                        for (int64_t k = 0; k < n; k++) {
                            const long v = std::clamp(std::lround(mat[static_cast<size_t>(k * len + s)]) + 128L, 0L, 255L);
                            (c.cls[static_cast<size_t>(k)] ? h1 : h0)[static_cast<size_t>(v)]++;
                        }
                        ds = distanceSumsHistogram(h0.data(), h1.data(), 256, -128.0);
                    } else {
                        for (int64_t k = 0; k < n; k++) col[static_cast<size_t>(k)] = mat[static_cast<size_t>(k * len + s)];
                        ds = distanceSums1D(col.data(), c.cls.data(), n, scratch);
                    }
                    out.z[static_cast<size_t>(p0 + s)] = dcorZ(ds);
                }
            }
            if (progress && !progress(base + n + n * c1 / len, total)) {
                err = "Cancelled.";
                return false;
            }
        }
    }
    warn = classWarnings(c, failed_total, sanity_split, "the alignment");
    return true;
}

int64_t maxDcorWindow(int64_t n_traces) {
    return kDcorMemoryBytes / (std::max<int64_t>(1, n_traces) * static_cast<int64_t>(sizeof(float)));
}

bool computeWindowDcor(TrsFile* file,
                       const std::vector<std::shared_ptr<ITransform>>& pipeline,
                       int32_t first_trace, int32_t trace_count,
                       const std::vector<BivariateWindow>& windows, int32_t byte_idx,
                       int64_t read_lo, int64_t read_hi, bool sanity_split, bool standardize,
                       const std::function<bool(int64_t, int64_t)>& progress,
                       DcorWindowResult& out, QString& warn, QString& err)
{
    out = DcorWindowResult{};
    if (windows.empty()) { err = "No windows to test."; return false; }
    const int64_t nw = static_cast<int64_t>(windows.size());
    const int64_t total = nw * 2 * std::max<int32_t>(1, trace_count);
    QStringList msg;
    std::vector<float> mat;
    for (int64_t wi = 0; wi < nw; wi++) {
        const BivariateWindow& w = windows[static_cast<size_t>(wi)];
        const Classes c = classify(file, first_trace, trace_count, byte_idx, sanity_split, w);
        if (!enoughTraces(c, w.name, err)) return false;
        const int64_t n = static_cast<int64_t>(c.traces.size());
        const int64_t len = processedLength(pipeline, w.raw_len);
        if (len <= 0) { err = QString("Window %1 is empty.").arg(w.name); return false; }
        if (len > maxDcorWindow(n)) {
            err = QString("Window %1: %2 traces × %3 samples don't fit in memory (at most %4 samples "
                          "for this many traces). Use shorter windows or a resampling stage.")
                      .arg(w.name).arg(n).arg(len).arg(maxDcorWindow(n));
            return false;
        }
        mat.assign(static_cast<size_t>(n * len), 0.0f);
        int32_t failed = 0;
        const int64_t base = wi * 2 * trace_count;
        if (!loadRows(file, pipeline, c.traces, w, w.raw_start, w.raw_len, read_lo, read_hi,
                      0, len, mat.data(), failed,
                      [&](int64_t done) { return !progress || progress(base + done, total); })) {
            err = "Cancelled.";
            return false;
        }
        if (standardize) {
            // Unit variance per sample (constant samples are left alone).
            #pragma omp parallel for schedule(static) num_threads(computeThreads())
            for (int64_t s = 0; s < len; s++) {
                double sum = 0, sum2 = 0;
                for (int64_t k = 0; k < n; k++) {
                    const double v = mat[static_cast<size_t>(k * len + s)];
                    sum += v; sum2 += v * v;
                }
                const double mean = sum / static_cast<double>(n);
                const double var  = sum2 / static_cast<double>(n) - mean * mean;
                if (var <= 0) continue;
                const float inv = static_cast<float>(1.0 / std::sqrt(var));
                for (int64_t k = 0; k < n; k++) mat[static_cast<size_t>(k * len + s)] *= inv;
            }
        }
        const DistanceSums ds = distanceSumsRows(mat.data(), n, len, c.cls.data(), computeThreads());
        out.z.push_back(dcorZ(ds));
        out.r2.push_back(dcorSquared(ds));
        out.n0.push_back(c.n0);
        out.n1.push_back(c.n1);
        const QString wmsg = classWarnings(c, failed, sanity_split, "window " + w.name);
        if (!wmsg.isEmpty() && !msg.contains(wmsg)) msg << wmsg;
        if (progress && !progress(base + 2 * trace_count, total)) {
            err = "Cancelled.";
            return false;
        }
    }
    warn = msg.join("\n");
    return true;
}
