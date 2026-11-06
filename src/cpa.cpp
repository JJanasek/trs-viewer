#include "cpa.h"

#include "pattern_align.h"     // rawStartForProcessedIndex
#include "ttest_bivariate.h"   // processedLength
#include "ttest_run.h"         // loadProcessedTrace

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// CPA by sample blocks.
//
// With the leakages centred per hypothesis (Lc, Σ_i Lc[h,i] = 0) and the
// traces centred per sample (Tc), Pearson's r is
//
//     r[h,s] = Σ_i Lc[h,i]·Tc[i,s] / (‖Lc[h]‖ · ‖Tc[·,s]‖)
//
// so a block of samples, with every trace's values for it in memory, gives
// the numerators of all hypotheses at once as one matrix product
// Lc (M × N) · Tc (N × len). Each trace's block is read once for all
// hypotheses, and memory is the block plus whatever curves are kept — not
// M × samples, which for a whole long trace (256 × 62.5M) doesn't fit.
// ---------------------------------------------------------------------------

namespace {

using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int loadThreads() {
#ifdef _OPENMP
    return std::clamp(omp_get_max_threads(), 1, kTTestMaxThreads);
#else
    return 1;
#endif
}

// A pipeline starts each block early by this many processed samples so
// filters settle before the block's own samples (as the dcor test does).
constexpr int64_t kMargin = 4096;

// Centres each column of t (N × len) in place; its norm into norm_t.
void centreColumns(float* t, int64_t n, int64_t len, std::vector<double>& norm_t) {
    std::vector<double> mean(static_cast<size_t>(len), 0.0), sq(static_cast<size_t>(len), 0.0);
    for (int64_t i = 0; i < n; i++) {
        const float* r = t + i * len;
        for (int64_t s = 0; s < len; s++) mean[static_cast<size_t>(s)] += r[s];
    }
    for (double& v : mean) v /= static_cast<double>(n);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        float* r = t + i * len;
        for (int64_t s = 0; s < len; s++) r[s] = static_cast<float>(r[s] - mean[static_cast<size_t>(s)]);
    }
    for (int64_t i = 0; i < n; i++) {
        const float* r = t + i * len;
        for (int64_t s = 0; s < len; s++) sq[static_cast<size_t>(s)] += static_cast<double>(r[s]) * r[s];
    }
    norm_t.resize(static_cast<size_t>(len));
    for (int64_t s = 0; s < len; s++) norm_t[static_cast<size_t>(s)] = std::sqrt(sq[static_cast<size_t>(s)]);
}

}  // namespace

bool loadSampleBlock(TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
                     int32_t first_trace, const std::vector<int32_t>& traces,
                     const std::vector<int32_t>& shifts, int64_t first_sample, int64_t raw_ns,
                     int64_t L, int64_t p0, int64_t len, float* out,
                     const std::function<bool(int64_t)>& progress)
{
    const int64_t m  = pipeline.empty() ? 0 : std::min(p0, kMargin);
    const int64_t rs = rawStartForProcessedIndex(pipeline, p0 - m, raw_ns);
    const int64_t re = p0 + len >= L ? raw_ns
                     : std::min(raw_ns, rawCountForProcessedCount(pipeline, p0 + len, raw_ns));
    const int64_t num_samples = file->header().num_samples;

    const int threads = loadThreads();
    std::vector<std::vector<std::shared_ptr<ITransform>>> pls(static_cast<size_t>(threads));
    for (auto& pl : pls)
        for (const auto& t : pipeline) pl.push_back(t->clone());
    std::vector<std::vector<float>> bufs(static_cast<size_t>(threads));
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
            const TraceLoad ld = loadProcessedTrace(file, pls[tid], first_trace + traces[static_cast<size_t>(k)],
                                                    first_sample + rs, re - rs, shifts, first_trace,
                                                    0, num_samples, bufs[tid], n_out);
            const float* src = bufs[tid].data();
            for (int64_t i = 0; i < len; i++)
                row[i] = (ld == TraceLoad::Ok && m + i < n_out) ? src[m + i] : 0.0f;
        }
        if (progress && !progress(b1)) return false;
    }
    return true;
}

std::vector<int32_t> cpaTraces(int32_t n_traces, const std::vector<int32_t>& shifts) {
    std::vector<int32_t> kept;
    kept.reserve(static_cast<size_t>(std::max(0, n_traces)));
    for (int32_t i = 0; i < n_traces; i++) {
        const int32_t s = i < static_cast<int32_t>(shifts.size()) ? shifts[static_cast<size_t>(i)] : 0;
        if (s != kAlignDiscardShift) kept.push_back(i);
    }
    return kept;
}

bool evaluateCpaLeakages(TrsFile* file, int32_t first_trace, const std::vector<int32_t>& traces,
                         int32_t n_hypotheses, const LeakageFn& leakage_fn,
                         std::vector<float>& leakages,
                         const std::function<bool(int32_t, int32_t)>& progress,
                         std::string& error)
{
    const int32_t N  = static_cast<int32_t>(traces.size());
    const int32_t DL = std::max<int32_t>(0, file->header().data_length);
    std::vector<uint8_t> data(static_cast<size_t>(N) * static_cast<size_t>(DL), 0);
    for (int32_t i = 0; i < N; i++) {
        const auto d = file->readData(first_trace + traces[static_cast<size_t>(i)]);
        const size_t n = std::min<size_t>(d.size(), static_cast<size_t>(DL));
        if (n > 0) std::copy(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(n),
                             data.begin() + static_cast<std::ptrdiff_t>(i) * DL);
    }
    leakages.assign(static_cast<size_t>(std::max(0, n_hypotheses)) * static_cast<size_t>(N), 0.0f);
    std::vector<float> tmp;
    for (int32_t h = 0; h < n_hypotheses; h++) {
        std::string lerr;
        if (!leakage_fn(data, DL, N, h, tmp, lerr)) {
            // Say which guess failed: a model that indexes the data with
            // key_guess runs past the trace data at some M.
            error = "The leakage model failed for key_guess = " + std::to_string(h)
                  + " (M = " + std::to_string(n_hypotheses) + ", guesses 0.."
                  + std::to_string(n_hypotheses - 1) + "; each trace has " + std::to_string(DL)
                  + " data bytes, plaintexts.shape[1]):\n\n" + lerr;
            return false;
        }
        if (static_cast<int32_t>(tmp.size()) != N) {
            error = "The leakage model returned " + std::to_string(tmp.size()) + " values for "
                  + std::to_string(N) + " traces (key_guess = " + std::to_string(h) + ").";
            return false;
        }
        std::copy(tmp.begin(), tmp.end(), leakages.begin() + static_cast<std::ptrdiff_t>(h) * N);
        if (progress && !progress(h + 1, n_hypotheses)) { error.clear(); return false; }
    }
    return true;
}

// Bytes per processed sample of a block: every trace's value, the products of
// `rows` hypotheses, two column statistics, and each loader thread's raw read
// (raw_per_sample raw samples per processed one, more with a decimating
// pipeline).
static int64_t blockBytesPerSample(int64_t n_traces, int64_t rows, int64_t raw_per_sample) {
    return 4 * (n_traces + rows) + 16 + 4 * static_cast<int64_t>(kTTestMaxThreads) * raw_per_sample;
}

int64_t estimateCpaBytes(int64_t n_traces, int64_t n_hypotheses, int64_t n_samples, int64_t curves) {
    const int64_t per_sample = blockBytesPerSample(n_traces, n_hypotheses, 1);
    const int64_t block_len  = std::clamp<int64_t>(kCpaBlockBytes / per_sample, 1, std::max<int64_t>(1, n_samples));
    return per_sample * block_len                // the block
         + 8 * n_hypotheses * n_traces           // the leakages and their centred copy
         + 4 * curves * n_samples;               // the curves kept
}

bool correlateCpa(TrsFile* file, int32_t first_trace, const std::vector<int32_t>& traces,
                  int64_t first_sample, int64_t n_samples_req,
                  const std::vector<int32_t>& shifts,
                  const std::vector<std::shared_ptr<ITransform>>& pipeline,
                  const std::vector<float>& leakages, int32_t n_hypotheses, int32_t max_curves,
                  CpaResult& out, const std::function<bool(int64_t, int64_t)>& progress,
                  std::string& error, int64_t block_bytes)
{
    out = CpaResult{};
    const TrsHeader& h = file->header();
    const int64_t raw_ns = (n_samples_req == 0)
        ? (h.num_samples - first_sample)
        : std::min<int64_t>(n_samples_req, h.num_samples - first_sample);
    if (raw_ns <= 0) { error = "The sample range is empty."; return false; }
    const int64_t L = processedLength(pipeline, raw_ns);
    if (L <= 0) { error = "No samples after pipeline"; return false; }
    const int64_t M = std::max(0, n_hypotheses);
    const int64_t N = static_cast<int64_t>(traces.size());
    out.n_hypotheses = static_cast<int32_t>(M);
    out.n_samples    = static_cast<int32_t>(L);
    out.n_traces     = static_cast<int32_t>(N);
    if (M == 0) return true;
    if (N < 2) { error = "Need at least 2 traces after excluding discarded ones."; return false; }
    if (static_cast<int64_t>(leakages.size()) != M * N) { error = "Leakage matrix has the wrong size."; return false; }

    // Centred leakages and their norms (a constant model has norm 0: r = 0).
    std::vector<float> lc(leakages);
    std::vector<double> norm_l(static_cast<size_t>(M), 0.0);
    for (int64_t k = 0; k < M; k++) {
        float* r = lc.data() + k * N;
        double mean = 0.0;
        for (int64_t i = 0; i < N; i++) mean += r[i];
        mean /= static_cast<double>(N);
        double sq = 0.0;
        for (int64_t i = 0; i < N; i++) {
            r[i] = static_cast<float>(r[i] - mean);
            sq += static_cast<double>(r[i]) * r[i];
        }
        norm_l[static_cast<size_t>(k)] = std::sqrt(sq);
    }

    const bool keep_all = max_curves >= M;
    const int64_t K = keep_all ? M : std::max<int32_t>(0, max_curves);
    // Pass 1 holds the products of every hypothesis (for the peaks).
    const int64_t raw_per_sample = (raw_ns + L - 1) / L;
    const int64_t per_sample = blockBytesPerSample(N, M, raw_per_sample);
    const int64_t block = std::clamp<int64_t>(block_bytes / per_sample, 1, L);
    const int64_t n_blocks = (L + block - 1) / block;
    if (n_blocks > 1 && pipelineRequiresWholeTrace(pipeline)) {
        error = "The pipeline has a stage that needs whole traces, but " + std::to_string(N)
              + " traces × " + std::to_string(L) + " samples don't fit in memory at once. Shorten "
              "the range to at most " + std::to_string(block) + " samples (after the pipeline), or "
              "remove that stage.";
        return false;
    }

    out.peak_r.assign(static_cast<size_t>(M), 0.0f);
    out.peak_sample.assign(static_cast<size_t>(M), 0);
    std::vector<float> best_abs(static_cast<size_t>(M), -1.0f);
    if (keep_all) {
        out.corr.assign(static_cast<size_t>(M) * static_cast<size_t>(L), 0.0f);
        out.curve_hyp.resize(static_cast<size_t>(M));
        std::iota(out.curve_hyp.begin(), out.curve_hyp.end(), 0);
    }

    const int passes = keep_all || K == 0 ? 1 : 2;
    const int64_t total = static_cast<int64_t>(passes) * n_blocks * N;
    std::vector<float> t, prod;
    std::vector<double> norm_t;

    // One pass over every block: products of `rows` hypotheses (rows of lc_p,
    // norms norm_p), handed to sink(row, sample, r) per block.
    auto pass = [&](int pass_idx, const float* lc_p, const std::vector<double>& norm_p, int64_t rows,
                    const std::function<void(int64_t, int64_t, int64_t, const float*)>& sink) -> bool {
        for (int64_t bi = 0; bi < n_blocks; bi++) {
            const int64_t p0 = bi * block, len = std::min(block, L - p0);
            t.assign(static_cast<size_t>(N * len), 0.0f);
            const int64_t base = (static_cast<int64_t>(pass_idx) * n_blocks + bi) * N;
            if (!loadSampleBlock(file, pipeline, first_trace, traces, shifts, first_sample, raw_ns, L, p0, len,
                           t.data(), [&](int64_t done) { return !progress || progress(base + done, total); }))
                return false;
            centreColumns(t.data(), N, len, norm_t);
            prod.resize(static_cast<size_t>(rows * len));
            Eigen::Map<const RowMat> lm(lc_p, rows, N);
            Eigen::Map<const RowMat> tm(t.data(), N, len);
            Eigen::Map<RowMat> pm(prod.data(), rows, len);
            pm.noalias() = lm * tm;
            #pragma omp parallel for schedule(static)
            for (int64_t k = 0; k < rows; k++) {
                float* r = prod.data() + k * len;
                const double nl = norm_p[static_cast<size_t>(k)];
                for (int64_t s = 0; s < len; s++) {
                    const double d = nl * norm_t[static_cast<size_t>(s)];
                    r[s] = d > 0.0 ? static_cast<float>(r[s] / d) : 0.0f;
                }
                sink(k, p0, len, r);
            }
        }
        return true;
    };

    // Pass 1: every hypothesis — its peak, and its curve if all are kept.
    const bool ok1 = pass(0, lc.data(), norm_l, M,
        [&](int64_t k, int64_t p0, int64_t len, const float* r) {
            float& best = best_abs[static_cast<size_t>(k)];
            for (int64_t s = 0; s < len; s++) {
                if (std::abs(r[s]) > best) {
                    best = std::abs(r[s]);
                    out.peak_r[static_cast<size_t>(k)] = r[s];
                    out.peak_sample[static_cast<size_t>(k)] = static_cast<int32_t>(p0 + s);
                }
            }
            if (keep_all)
                std::copy(r, r + len, out.corr.begin() + static_cast<std::ptrdiff_t>(k * L + p0));
        });
    if (!ok1) { error = "Cancelled."; return false; }
    if (passes == 1) return true;

    // Pass 2: the curves of the K strongest.
    std::vector<int32_t> order(static_cast<size_t>(M));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int32_t a, int32_t b) {
        return best_abs[static_cast<size_t>(a)] > best_abs[static_cast<size_t>(b)];
    });
    out.curve_hyp.assign(order.begin(), order.begin() + K);
    std::vector<float> lc_k(static_cast<size_t>(K * N));
    std::vector<double> norm_k(static_cast<size_t>(K));
    for (int64_t j = 0; j < K; j++) {
        const int64_t k = out.curve_hyp[static_cast<size_t>(j)];
        std::copy(lc.begin() + static_cast<std::ptrdiff_t>(k * N), lc.begin() + static_cast<std::ptrdiff_t>((k + 1) * N),
                  lc_k.begin() + static_cast<std::ptrdiff_t>(j * N));
        norm_k[static_cast<size_t>(j)] = norm_l[static_cast<size_t>(k)];
    }
    out.corr.assign(static_cast<size_t>(K) * static_cast<size_t>(L), 0.0f);
    const bool ok2 = pass(1, lc_k.data(), norm_k, K,
        [&](int64_t j, int64_t p0, int64_t len, const float* r) {
            std::copy(r, r + len, out.corr.begin() + static_cast<std::ptrdiff_t>(j * L + p0));
        });
    if (!ok2) { error = "Cancelled."; return false; }
    return true;
}

bool computeCpa(TrsFile* file, int32_t first_trace, int32_t n_traces,
                int64_t first_sample, int64_t n_samples_req, int32_t n_hypotheses,
                const std::vector<int32_t>& shifts,
                const std::vector<std::shared_ptr<ITransform>>& pipeline,
                const LeakageFn& leakage_fn, CpaResult& out,
                std::function<bool(int32_t, int32_t)> progress, std::string& error)
{
    const std::vector<int32_t> traces = cpaTraces(n_traces, shifts);
    if (traces.size() < 2) {
        out = CpaResult{};
        out.n_hypotheses = std::max(0, n_hypotheses);
        error = "Need at least 2 traces after excluding discarded ones.";
        return false;
    }
    // Progress over both steps: one unit per hypothesis, then per trace read.
    const int64_t m = std::max(0, n_hypotheses);
    std::vector<float> leakages;
    if (!evaluateCpaLeakages(file, first_trace, traces, n_hypotheses, leakage_fn, leakages,
                             [&](int32_t done, int32_t) {
                                 return !progress || progress(done, static_cast<int32_t>(m + static_cast<int64_t>(traces.size())));
                             },
                             error)) {
        out = CpaResult{};
        out.n_hypotheses = static_cast<int32_t>(m);
        return false;
    }
    return correlateCpa(file, first_trace, traces, first_sample, n_samples_req, shifts, pipeline,
                        leakages, n_hypotheses, n_hypotheses, out,
                        [&](int64_t done, int64_t total) {
                            if (!progress) return true;
                            const double f = total > 0 ? static_cast<double>(done) / total : 1.0;
                            return progress(static_cast<int32_t>(m + f * static_cast<double>(traces.size())),
                                            static_cast<int32_t>(m + static_cast<int64_t>(traces.size())));
                        },
                        error);
}
