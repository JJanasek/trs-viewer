#include "cpa.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Online (streaming) CPA — O(M·NS + M·N) memory instead of O(N·NS).
//
// For each chunk of hypotheses we stream through every trace once, maintaining:
//   sum_T[s]         = Σ_i  T[i,s]
//   sum_T2[s]        = Σ_i  T[i,s]²          (only filled on first chunk pass)
//   sum_LT[h·NS + s] = Σ_i  L[h,i] · T[i,s]
//   sum_L[h]         = Σ_i  L[h,i]
//   sum_L2[h]        = Σ_i  L[h,i]²
//
// Final Pearson per (h,s):
//   mean_T  = sum_T[s]  / N
//   mean_L  = sum_L[h]  / N
//   var_T   = sum_T2[s] / N - mean_T²
//   var_L   = sum_L2[h] / N - mean_L²
//   cov     = sum_LT[h,s] / N - mean_L·mean_T
//   corr    = cov / sqrt(var_T · var_L)
//
// Chunk size is chosen so that the per-chunk buffers stay ≤ MEM_LIMIT.
// ---------------------------------------------------------------------------

static constexpr size_t MEM_LIMIT_BYTES = 512ULL << 20; // 512 MB per chunk

// Helper: load one trace into proc_buf, applying alignment shift and pipeline.
// Returns the number of valid output samples (≤ NS).
static int64_t loadTrace(
    TrsFile*        file,
    int32_t         trace_idx,
    int64_t         first_sample,
    int64_t         raw_ns,
    int32_t         shift,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    std::vector<float>& raw_buf,
    std::vector<float>& proc_buf)
{
    const TrsHeader& h = file->header();
    const int64_t adj_start = first_sample + shift;

    std::fill(raw_buf.begin(), raw_buf.end(), 0.0f);
    if (adj_start < h.num_samples && adj_start + raw_ns > 0) {
        int64_t src_start = std::max<int64_t>(0, adj_start);
        int64_t src_end   = std::min<int64_t>(h.num_samples, adj_start + raw_ns);
        int64_t dst_off   = src_start - adj_start;
        int64_t count     = src_end - src_start;
        file->readSamples(trace_idx, src_start, count, raw_buf.data() + dst_off);
    }

    int64_t out_count = raw_ns;
    std::copy(raw_buf.begin(), raw_buf.begin() + raw_ns, proc_buf.begin());
    for (const auto& t : pipeline) {
        auto tc = t->clone(); tc->reset();
        out_count = tc->apply(proc_buf.data(), out_count, 0);
    }
    return out_count;
}

bool computeCpa(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        n_traces,
    int64_t        first_sample,
    int64_t        n_samples_req,
    int32_t        n_hypotheses,
    const std::vector<int32_t>& shifts,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const LeakageFn& leakage_fn,
    CpaResult&     out,
    std::function<bool(int32_t, int32_t)> progress,
    std::string&   error)
{
    const TrsHeader& h = file->header();

    int64_t raw_ns = (n_samples_req == 0)
                     ? (h.num_samples - first_sample)
                     : std::min<int64_t>(n_samples_req, h.num_samples - first_sample);
    int64_t eff_ns = raw_ns;
    for (const auto& t : pipeline) eff_ns = t->transformedCount(eff_ns);
    if (eff_ns <= 0) { error = "No samples after pipeline"; return false; }

    const int32_t N  = n_traces;
    const int32_t NS = static_cast<int32_t>(eff_ns);
    const int32_t M  = n_hypotheses;
    const int32_t DL = h.data_length;

    // -----------------------------------------------------------------------
    // Step 1: Load all per-trace data bytes (N × DL). Small — kilobytes range.
    // -----------------------------------------------------------------------
    std::vector<uint8_t> data_flat(static_cast<size_t>(N) * std::max(DL, 0), 0);
    for (int32_t i = 0; i < N; i++) {
        auto d = file->readData(first_trace + i);
        size_t copy_len = std::min<size_t>(d.size(), static_cast<size_t>(std::max(DL, 0)));
        if (copy_len > 0)
            memcpy(data_flat.data() + static_cast<size_t>(i) * DL, d.data(), copy_len);
    }

    // -----------------------------------------------------------------------
    // Step 2: Determine chunk size so per-chunk buffers stay ≤ MEM_LIMIT.
    //   Per hypothesis: N floats (L) + NS doubles (sum_LT).
    // -----------------------------------------------------------------------
    size_t bytes_per_hyp = static_cast<size_t>(N)  * sizeof(float)
                         + static_cast<size_t>(NS) * sizeof(double);
    int32_t chunk_m = static_cast<int32_t>(
        std::max<size_t>(1, MEM_LIMIT_BYTES / bytes_per_hyp));
    chunk_m = std::min(chunk_m, M);

    // Number of passes over the traces (= ceil(M / chunk_m))
    const int32_t n_chunks    = (M + chunk_m - 1) / chunk_m;
    // Progress: M steps for leakage eval + N * n_chunks steps for trace streaming
    const int32_t total_steps = M + N * n_chunks;
    int32_t       progress_done = 0;

    // -----------------------------------------------------------------------
    // Step 3: Allocate output and shared trace-side accumulators.
    // -----------------------------------------------------------------------
    out.n_hypotheses = M;
    out.n_samples    = NS;
    out.n_traces     = N;
    out.corr.assign(static_cast<size_t>(M) * NS, 0.0f);

    // sum_T / sum_T2 are the same for all chunks — compute once on first pass.
    std::vector<double> sum_T(NS, 0.0), sum_T2(NS, 0.0);

    std::vector<float>  raw_buf(static_cast<size_t>(raw_ns));
    std::vector<float>  proc_buf(static_cast<size_t>(raw_ns) + 64);

    // -----------------------------------------------------------------------
    // Step 4: Process hypotheses in chunks.
    // -----------------------------------------------------------------------
    for (int32_t h_start = 0; h_start < M; h_start += chunk_m) {
        const int32_t h_end    = std::min(h_start + chunk_m, M);
        const int32_t cur_m    = h_end - h_start;
        const bool    first_chunk = (h_start == 0);

        // --- 4a. Evaluate leakage for this chunk of hypotheses ---
        std::vector<float>  L_chunk(static_cast<size_t>(cur_m) * N);
        std::vector<double> sum_L(cur_m, 0.0), sum_L2(cur_m, 0.0);

        {
            std::vector<float> tmp(N);
            for (int32_t ci = 0; ci < cur_m; ci++) {
                std::string lerr;
                if (!leakage_fn(data_flat, DL, N, h_start + ci, tmp, lerr)) {
                    error = lerr;
                    return false;
                }
                float* row = L_chunk.data() + static_cast<size_t>(ci) * N;
                double sl = 0, sl2 = 0;
                for (int32_t i = 0; i < N; i++) {
                    row[i] = tmp[i];
                    sl  += tmp[i];
                    sl2 += static_cast<double>(tmp[i]) * tmp[i];
                }
                sum_L[ci]  = sl;
                sum_L2[ci] = sl2;

                if (!progress(++progress_done, total_steps)) return false;
            }
        }

        // --- 4b. Stream traces, accumulate sum_LT (and sum_T/sum_T2 if first chunk) ---
        std::vector<double> sum_LT(static_cast<size_t>(cur_m) * NS, 0.0);

        for (int32_t i = 0; i < N; i++) {
            const int32_t shift = (i < static_cast<int32_t>(shifts.size())) ? shifts[i] : 0;
            int64_t out_count = loadTrace(file, first_trace + i,
                                          first_sample, raw_ns, shift,
                                          pipeline, raw_buf, proc_buf);

            if (first_chunk) {
                for (int32_t s = 0; s < NS && s < out_count; s++) {
                    double v = proc_buf[s];
                    sum_T[s]  += v;
                    sum_T2[s] += v * v;
                }
            }

            // Per-hypothesis accumulation — parallel over hypotheses.
            #pragma omp parallel for schedule(static)
            for (int32_t ci = 0; ci < cur_m; ci++) {
                double l       = L_chunk[static_cast<size_t>(ci) * N + i];
                double* slt    = sum_LT.data() + static_cast<size_t>(ci) * NS;
                int32_t lim    = static_cast<int32_t>(std::min<int64_t>(NS, out_count));
                for (int32_t s = 0; s < lim; s++)
                    slt[s] += l * proc_buf[s];
            }

            if (i % 256 == 0 && !progress(progress_done + i, total_steps)) return false;
        }
        progress_done += N;

        // --- 4c. Compute Pearson for this chunk ---
        #pragma omp parallel for schedule(static)
        for (int32_t ci = 0; ci < cur_m; ci++) {
            int32_t hyp   = h_start + ci;
            double mean_L = sum_L[ci]  / N;
            double var_L  = sum_L2[ci] / N - mean_L * mean_L;
            double rstd_L = (var_L > 0.0) ? 1.0 / std::sqrt(var_L) : 0.0;

            const double* slt_row  = sum_LT.data() + static_cast<size_t>(ci) * NS;
            float*        corr_row = out.corr.data() + static_cast<size_t>(hyp) * NS;

            for (int32_t s = 0; s < NS; s++) {
                double mean_T = sum_T[s]  / N;
                double var_T  = sum_T2[s] / N - mean_T * mean_T;
                double rstd_T = (var_T > 0.0) ? 1.0 / std::sqrt(var_T) : 0.0;
                double cov    = slt_row[s] / N - mean_L * mean_T;
                corr_row[s]   = static_cast<float>(cov * rstd_L * rstd_T);
            }
        }
    }

    return true;
}
