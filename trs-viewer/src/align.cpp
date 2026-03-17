#include "align.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Load `count` raw samples from trace_idx starting at first_sample.
// Zero-pads the tail if fewer samples are available.
static std::vector<float> loadRaw(
    TrsFile* file, int32_t trace_idx,
    int64_t first_sample, int64_t count)
{
    std::vector<float> buf(static_cast<size_t>(count), 0.0f);
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
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,
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

    ref_first_sample = std::max<int64_t>(0, ref_first_sample);
    ref_num_samples  = std::min(ref_num_samples,
                                h.num_samples - ref_first_sample);
    if (ref_num_samples <= 0) {
        error = "Reference region is empty or outside trace bounds.";
        return false;
    }

    // Find peak in reference trace
    int32_t ref_abs       = first_trace + ref_trace_offset;
    auto    ref_buf       = loadRaw(file, ref_abs,
                                    ref_first_sample, ref_num_samples);
    int64_t ref_peak_local = argPeak(ref_buf.data(), ref_num_samples, use_abs);
    int64_t ref_peak_pos   = ref_first_sample + ref_peak_local;

    for (int ti = 0; ti < num_traces; ti++) {
        if (progress && !progress(ti, num_traces)) {
            error = "Cancelled.";
            return false;
        }

        if (ti == ref_trace_offset) {
            out.shifts[static_cast<size_t>(ti)] = 0;
            continue;
        }

        int64_t s_start = std::max<int64_t>(0,
                              ref_peak_pos - search_half);
        int64_t s_end   = std::min<int64_t>(h.num_samples,
                              ref_peak_pos + search_half + 1);
        int64_t s_len   = s_end - s_start;
        if (s_len <= 0) continue;

        auto    sbuf       = loadRaw(file, first_trace + ti, s_start, s_len);
        int64_t local_peak = argPeak(sbuf.data(), s_len, use_abs);
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
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int32_t        search_half,
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

    ref_first_sample = std::max<int64_t>(0, ref_first_sample);
    ref_num_samples  = std::min(ref_num_samples,
                                h.num_samples - ref_first_sample);
    if (ref_num_samples < 2) {
        error = "Reference region too short (need ≥ 2 samples).";
        return false;
    }

    const int64_t M = ref_num_samples;

    // Build mean-centred reference template
    int32_t ref_abs = first_trace + ref_trace_offset;
    auto    ref_raw = loadRaw(file, ref_abs, ref_first_sample, M);

    double ref_sum = 0.0;
    for (int64_t i = 0; i < M; i++) ref_sum += ref_raw[i];
    double ref_mean = ref_sum / M;

    double ref_sq = 0.0;
    std::vector<float> ref_c(static_cast<size_t>(M));
    for (int64_t i = 0; i < M; i++) {
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

    // Actual lag range after boundary clamping
    int neg_half = static_cast<int>(ref_first_sample - sbuf_first);
    int pos_half = static_cast<int>(sbuf_end - (ref_first_sample + M));

    for (int ti = 0; ti < num_traces; ti++) {
        if (progress && !progress(ti, num_traces)) {
            error = "Cancelled.";
            return false;
        }

        if (ti == ref_trace_offset) {
            out.shifts[static_cast<size_t>(ti)] = 0;
            continue;
        }

        auto sbuf = loadRaw(file, first_trace + ti, sbuf_first, sbuf_len);

        float best_ncc = -2.0f;
        int   best_k   = 0;

        for (int k = -neg_half; k <= pos_half; k++) {
            int64_t off = static_cast<int64_t>(neg_half + k);
            if (off < 0 || off + M > sbuf_len) continue;

            // Compute patch mean, then NCC in a single pass
            double sum = 0.0;
            for (int64_t j = 0; j < M; j++)
                sum += sbuf[static_cast<size_t>(off + j)];
            double patch_mean = sum / M;

            double sq = 0.0, dot = 0.0;
            for (int64_t j = 0; j < M; j++) {
                double v = sbuf[static_cast<size_t>(off + j)] - patch_mean;
                sq  += v * v;
                dot += ref_c[static_cast<size_t>(j)] * v;
            }

            float ncc = (ref_norm > 0.0 && sq > 0.0)
                        ? static_cast<float>(dot / (ref_norm * std::sqrt(sq)))
                        : 0.0f;

            if (ncc > best_ncc) { best_ncc = ncc; best_k = k; }
        }

        // Positive best_k: feature is later in this trace than in the reference.
        out.shifts[static_cast<size_t>(ti)] = best_k;
    }

    if (progress) progress(num_traces, num_traces);
    return true;
}
