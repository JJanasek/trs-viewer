#include "align.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Load `count` raw samples from trace_idx starting at first_sample into a
// buffer allocated for at least `alloc_size` floats (>= count — scratch
// space for pipeline stages that expand the sample count, e.g. an
// overlapping STFT). Zero-pads the tail if fewer samples are available, and
// zero-fills the region beyond `count` up to `alloc_size`.
static std::vector<float> loadRaw(
    TrsFile* file, int32_t trace_idx,
    int64_t first_sample, int64_t count, int64_t alloc_size)
{
    std::vector<float> buf(static_cast<size_t>(std::max(count, alloc_size)), 0.0f);
    const TrsHeader& h = file->header();
    if (first_sample < 0 || first_sample >= h.num_samples || count <= 0)
        return buf;
    int64_t avail = h.num_samples - first_sample;
    int64_t n     = std::min(count, avail);
    int64_t got   = file->readSamples(trace_idx, first_sample, n, buf.data());
    if (got < n)
        std::fill(buf.begin() + static_cast<size_t>(got),
                  buf.begin() + static_cast<size_t>(n), 0.0f);
    return buf;
}

// Loads a raw window and runs it through the processing pipeline (reset +
// apply, offset 0), mirroring the t-test accumulation path so alignment sees
// the same samples the user sees on the plot. The pipeline may shorten OR
// expand the buffer (a decimating/stride/window-resample stage shortens it;
// an overlapping STFT can expand it) — the buffer is pre-sized for whichever
// is larger, same contract as every other pipeline call site (t-test, CPA,
// xcorr, main plot). The caller maps positions found in the returned buffer
// back to raw offsets with rescaleIndex().
static std::vector<float> loadProcessed(
    TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int32_t trace_idx, int64_t first_sample, int64_t count)
{
    int64_t expected_out = count;
    for (const auto& t : pipeline) expected_out = t->transformedCount(expected_out);

    std::vector<float> buf = loadRaw(file, trace_idx, first_sample, count, expected_out);
    for (const auto& t : pipeline) t->reset();
    int64_t n_out = count;
    for (const auto& t : pipeline)
        n_out = t->apply(buf.data(), n_out, 0);
    buf.resize(static_cast<size_t>(std::max<int64_t>(0, n_out)));
    return buf;
}

// Loads and pipeline-processes ref_num_samples starting at ref_first_sample
// from ref_trace_count consecutive traces starting at
// first_trace+ref_trace_offset, and returns their elementwise average — the
// same processed length loadProcessed() would give for a single trace, since
// every one of them shares the same raw window and pipeline. A trace that
// comes back empty (e.g. loadProcessed found nothing) is skipped rather than
// corrupting the average. ref_trace_count == 1 degenerates to loadProcessed()
// exactly, byte-for-byte (dividing by the single count of 1 changes nothing).
static std::vector<float> loadProcessedAveraged(
    TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int32_t first_trace, int32_t ref_trace_offset, int32_t ref_trace_count,
    int64_t ref_first_sample, int64_t ref_num_samples)
{
    std::vector<float> acc;
    int32_t used = 0;
    for (int32_t k = 0; k < ref_trace_count; k++) {
        auto buf = loadProcessed(file, pipeline, first_trace + ref_trace_offset + k,
                                  ref_first_sample, ref_num_samples);
        if (buf.empty()) continue;
        if (acc.empty()) {
            acc.assign(buf.size(), 0.0f);
        } else if (buf.size() != acc.size()) {
            continue; // pipeline should never vary output length by trace; guard anyway
        }
        for (size_t i = 0; i < buf.size(); i++) acc[i] += buf[i];
        used++;
    }
    if (used > 1) {
        const float inv = 1.0f / static_cast<float>(used);
        for (auto& v : acc) v *= inv;
    }
    return acc;
}

// Rounds num/den to the nearest integer (ties away from zero). den > 0.
static int64_t roundedDiv(int64_t num, int64_t den)
{
    if (num >= 0) return (num + den / 2) / den;
    return -(((-num) + den / 2) / den);
}

// Rescales `idx` from a domain of length `from_len` into a domain of length
// `to_len` (both spanning the same underlying window). Used to move offsets
// and lags between raw sample space and the (possibly decimated) space a
// pipeline stage produces. Identity when from_len == to_len.
static int64_t rescaleIndex(int64_t idx, int64_t from_len, int64_t to_len)
{
    if (from_len <= 0) return 0;
    return roundedDiv(idx * to_len, from_len);
}

// Index of the peak in buf[0..n-1].
// use_abs=true  → argmax |v|
// use_abs=false → argmax  v
static int64_t argPeak(const float* buf, int64_t n, bool use_abs)
{
    if (n <= 0) return 0;
    int64_t best   = 0;
    float   best_v = use_abs ? std::abs(buf[0]) : buf[0];
    for (int64_t i = 1; i < n; i++) {
        float v = use_abs ? std::abs(buf[i]) : buf[i];
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Peak alignment
// ---------------------------------------------------------------------------

bool alignByPeak(
    TrsFile*       file,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,
    int32_t        ref_trace_count,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int32_t        search_half,
    bool           use_abs,
    AlignResult&   out,
    AlignProgress  progress,
    std::string&   error)
{
    out.shifts.assign(static_cast<size_t>(num_traces), 0);

    const TrsHeader& h = file->header();
    if (ref_trace_offset < 0 || ref_trace_offset >= num_traces) {
        error = "Reference trace offset out of range.";
        return false;
    }
    ref_trace_count = std::clamp(ref_trace_count, 1, num_traces - ref_trace_offset);

    ref_first_sample = std::max<int64_t>(0, ref_first_sample);
    ref_num_samples  = std::min(ref_num_samples,
                                h.num_samples - ref_first_sample);
    if (ref_num_samples <= 0) {
        error = "Reference region is empty or outside trace bounds.";
        return false;
    }

    // Find peak in the reference template — a single trace, or the
    // elementwise average of ref_trace_count consecutive traces starting at
    // ref_trace_offset (see loadProcessedAveraged).
    auto ref_buf = loadProcessedAveraged(file, pipeline, first_trace, ref_trace_offset,
                                          ref_trace_count, ref_first_sample, ref_num_samples);
    if (ref_buf.empty()) {
        error = "Reference region produced no samples after the processing pipeline.";
        return false;
    }
    int64_t ref_peak_proc = argPeak(ref_buf.data(),
                                    static_cast<int64_t>(ref_buf.size()), use_abs);
    int64_t ref_peak_local = rescaleIndex(
        ref_peak_proc, static_cast<int64_t>(ref_buf.size()), ref_num_samples);
    int64_t ref_peak_pos   = ref_first_sample + ref_peak_local;

    for (int ti = 0; ti < num_traces; ti++) {
        if (progress && !progress(ti, num_traces)) {
            error = "Cancelled.";
            return false;
        }

        // With a single-trace reference, trace ref_trace_offset *is* the
        // template — shift 0 by definition, no search needed. Once averaging
        // more than one trace, no individual trace (including ones folded
        // into the average) is guaranteed to already match it, so every
        // trace is searched uniformly.
        if (ref_trace_count == 1 && ti == ref_trace_offset) {
            out.shifts[static_cast<size_t>(ti)] = 0;
            continue;
        }

        int64_t s_start = std::max<int64_t>(0,
                              ref_peak_pos - search_half);
        int64_t s_end   = std::min<int64_t>(h.num_samples,
                              ref_peak_pos + search_half + 1);
        int64_t s_len   = s_end - s_start;
        if (s_len <= 0) continue;

        auto sbuf = loadProcessed(file, pipeline, first_trace + ti, s_start, s_len);
        if (sbuf.empty()) continue;
        int64_t local_peak_proc = argPeak(sbuf.data(),
                                          static_cast<int64_t>(sbuf.size()), use_abs);
        int64_t local_peak = rescaleIndex(
            local_peak_proc, static_cast<int64_t>(sbuf.size()), s_len);
        int64_t trace_peak = s_start + local_peak;

        // Positive shift: trace feature is later than reference → advance.
        out.shifts[static_cast<size_t>(ti)] =
            static_cast<int32_t>(trace_peak - ref_peak_pos);
    }

    if (progress) progress(num_traces, num_traces);
    return true;
}

// ---------------------------------------------------------------------------
// Cross-correlation alignment
// ---------------------------------------------------------------------------

bool alignByXCorr(
    TrsFile*       file,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,
    int32_t        ref_trace_count,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int32_t        search_half,
    float          min_correlation,
    AlignResult&   out,
    AlignProgress  progress,
    std::string&   error)
{
    out.shifts.assign(static_cast<size_t>(num_traces), 0);
    out.scores.assign(static_cast<size_t>(num_traces), 0.0f);

    const TrsHeader& h = file->header();
    if (ref_trace_offset < 0 || ref_trace_offset >= num_traces) {
        error = "Reference trace offset out of range.";
        return false;
    }
    ref_trace_count = std::clamp(ref_trace_count, 1, num_traces - ref_trace_offset);

    ref_first_sample = std::max<int64_t>(0, ref_first_sample);
    ref_num_samples  = std::min(ref_num_samples,
                                h.num_samples - ref_first_sample);
    if (ref_num_samples < 2) {
        error = "Reference region too short (need ≥ 2 samples).";
        return false;
    }

    const int64_t M = ref_num_samples;

    // Build mean-centred reference template — a single trace, or the
    // elementwise average of ref_trace_count consecutive traces starting at
    // ref_trace_offset (see loadProcessedAveraged). `M_proc` is its length
    // after the pipeline — may be shorter than M if the pipeline decimates.
    auto ref_raw = loadProcessedAveraged(file, pipeline, first_trace, ref_trace_offset,
                                          ref_trace_count, ref_first_sample, M);
    const int64_t M_proc = static_cast<int64_t>(ref_raw.size());
    if (M_proc < 2) {
        error = "Reference region too short after the processing pipeline "
                "(need ≥ 2 samples).";
        return false;
    }

    double ref_sum = 0.0;
    for (int64_t i = 0; i < M_proc; i++) ref_sum += ref_raw[i];
    double ref_mean = ref_sum / M_proc;

    double ref_sq = 0.0;
    std::vector<float> ref_c(static_cast<size_t>(M_proc));
    for (int64_t i = 0; i < M_proc; i++) {
        double v = ref_raw[i] - ref_mean;
        ref_c[static_cast<size_t>(i)] = static_cast<float>(v);
        ref_sq += v * v;
    }
    const double ref_norm = std::sqrt(ref_sq);

    // Search buffer covers [ref_first - search_half, ref_first + M + search_half)
    int64_t sbuf_first = std::max<int64_t>(0,
                             ref_first_sample - search_half);
    int64_t sbuf_end   = std::min<int64_t>(h.num_samples,
                             ref_first_sample + M + search_half);
    int64_t sbuf_len   = sbuf_end - sbuf_first;

    // Actual lag range after boundary clamping, in *raw* samples.
    int64_t neg_half_raw = ref_first_sample - sbuf_first;
    int64_t pos_half_raw = sbuf_end - (ref_first_sample + M);

    // Traces are independent, so the search is parallelised across them —
    // one cloned pipeline per worker thread, since a shared transform
    // instance (e.g. STFT, which caches its FFT plan) isn't safe to call
    // apply() on concurrently. Capped at 8 threads to keep memory bounded.
#ifdef _OPENMP
    const int n_threads = std::clamp(omp_get_max_threads(), 1, 8);
#else
    const int n_threads = 1;
#endif
    std::vector<std::vector<std::shared_ptr<ITransform>>> thread_pipelines(
        static_cast<size_t>(n_threads));
    for (auto& tp : thread_pipelines)
        for (const auto& t : pipeline) tp.push_back(t->clone());

    // Processed in batches so `progress` (which may touch a GUI, e.g. the
    // caller's QProgressDialog) is only ever invoked from this thread,
    // between parallel regions — never from a worker thread. The batch size
    // is capped at 64 for parallel efficiency, but also capped so there are
    // at least ~20 progress updates regardless of trace count — otherwise a
    // run with fewer traces than the batch size reports progress exactly
    // once, at the very end, which looks indistinguishable from a hang.
    const int32_t kBatchSize = std::clamp(num_traces / 20, 1, 64);
    bool cancelled = false;

    for (int32_t batch_start = 0; batch_start < num_traces && !cancelled;
         batch_start += kBatchSize) {
        int32_t batch_end = std::min(batch_start + kBatchSize, num_traces);

        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int32_t ti = batch_start; ti < batch_end; ti++) {
            // See the matching comment in alignByPeak: only a true
            // single-trace reference makes ref_trace_offset's own shift/score
            // trivially known in advance.
            if (ref_trace_count == 1 && ti == ref_trace_offset) {
                out.shifts[static_cast<size_t>(ti)] = 0;
                out.scores[static_cast<size_t>(ti)] = 1.0f;
                continue;
            }

#ifdef _OPENMP
            auto& pipeline_local = thread_pipelines[static_cast<size_t>(omp_get_thread_num())];
#else
            auto& pipeline_local = thread_pipelines[0];
#endif
            auto sbuf = loadProcessed(file, pipeline_local, first_trace + ti, sbuf_first, sbuf_len);
            const int64_t L_proc = static_cast<int64_t>(sbuf.size());

            float   local_best_ncc = -2.0f;   // < any real NCC; stays if no window fits
            int64_t best_k_raw      = 0;

            if (L_proc >= M_proc) {
                // Map the raw lag range and the reference-window offset into
                // the (possibly decimated) processed domain of this buffer.
                int64_t neg_half = rescaleIndex(neg_half_raw, sbuf_len, L_proc);
                int64_t pos_half = rescaleIndex(pos_half_raw, sbuf_len, L_proc);
                int64_t best_k   = 0;

                // Sliding-window NCC: off = neg_half + k starts at 0 (k =
                // -neg_half, always in range since neg_half >= 0) and grows
                // by exactly 1 per step, so the patch's sum/sum-of-squares
                // are updated in O(1) per lag instead of rescanning all
                // M_proc samples — only the dot product against the fixed
                // reference template still needs a fresh O(M_proc) pass.
                double sum = 0.0, sumsq = 0.0;
                bool   window_init = false;

                for (int64_t k = -neg_half; k <= pos_half; k++) {
                    int64_t off = neg_half + k;
                    if (off + M_proc > L_proc) break;   // off only grows — nothing further fits

                    if (!window_init) {
                        for (int64_t j = 0; j < M_proc; j++) {
                            double v = sbuf[static_cast<size_t>(off + j)];
                            sum   += v;
                            sumsq += v * v;
                        }
                        window_init = true;
                    }

                    double patch_mean = sum / M_proc;
                    double sq = sumsq - sum * patch_mean;  // Σ(v-mean)² = Σv² - (Σv)·mean
                    if (sq < 0.0) sq = 0.0;                // guard tiny FP rounding

                    double dot = 0.0;
                    for (int64_t j = 0; j < M_proc; j++)
                        dot += ref_c[static_cast<size_t>(j)] *
                               (static_cast<double>(sbuf[static_cast<size_t>(off + j)]) - patch_mean);

                    float ncc = (ref_norm > 0.0 && sq > 0.0)
                                ? static_cast<float>(dot / (ref_norm * std::sqrt(sq)))
                                : 0.0f;

                    if (ncc > local_best_ncc) { local_best_ncc = ncc; best_k = k; }

                    // Slide the window by one sample, if there is a next iteration.
                    if (off + 1 + M_proc <= L_proc) {
                        double outgoing = sbuf[static_cast<size_t>(off)];
                        double incoming = sbuf[static_cast<size_t>(off + M_proc)];
                        sum   += incoming - outgoing;
                        sumsq += incoming * incoming - outgoing * outgoing;
                    }
                }

                // Positive best_k: feature is later in this trace than in the
                // reference. Map the processed-domain lag back to raw samples.
                best_k_raw = rescaleIndex(best_k, L_proc, sbuf_len);
            }

            // No window fit at all (e.g. trace ran out of samples) → treat as
            // the worst possible match rather than an unset sentinel value.
            float score = (local_best_ncc > -2.0f) ? local_best_ncc : -1.0f;
            out.scores[static_cast<size_t>(ti)] = score;
            out.shifts[static_cast<size_t>(ti)] = (score < min_correlation)
                ? kAlignDiscardShift
                : static_cast<int32_t>(best_k_raw);
        }

        if (progress && !progress(batch_end, num_traces)) cancelled = true;
    }

    if (cancelled) {
        error = "Cancelled.";
        return false;
    }
    if (progress) progress(num_traces, num_traces);
    return true;
}
