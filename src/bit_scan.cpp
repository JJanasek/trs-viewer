#include "bit_scan.h"
#include "cpa.h"       // loadSampleBlock
#include "ttest.h"     // kTTestMaxThreads
#include "ttest_run.h"
#include "ttest_bivariate.h"

#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int scanThreads() {
#ifdef _OPENMP
    return std::clamp(omp_get_max_threads(), 1, kTTestMaxThreads);
#else
    return 1;
#endif
}

// One block's raw memory per processed sample: every trace's value, and each
// loader thread's raw read.
int64_t blockBytesPerSample(int64_t n_traces, int64_t raw_per_sample) {
    return 4 * n_traces + 4 * static_cast<int64_t>(kTTestMaxThreads) * raw_per_sample;
}

}  // namespace

int64_t estimateBitScanBytes(int64_t n_traces, int64_t n_bits, int64_t n_samples) {
    const int64_t per_sample = blockBytesPerSample(n_traces, 1);
    const int64_t block = std::clamp<int64_t>(kBitScanBlockBytes / per_sample, 1, std::max<int64_t>(1, n_samples));
    return per_sample * block + n_traces * n_bits / 8 + 40 * n_bits
         + 32 * block * static_cast<int64_t>(kTTestMaxThreads);
}

bool computeBitScan(TrsFile* file, int32_t first_trace, const std::vector<int32_t>& traces,
                    int64_t first_sample, int64_t n_samples_req, const std::vector<int32_t>& shifts,
                    const std::vector<std::shared_ptr<ITransform>>& pipeline,
                    int32_t first_byte, int32_t n_bytes, bool keep_curves, BitScanResult& out,
                    const std::function<bool(int64_t, int64_t)>& progress, std::string& error,
                    int64_t block_bytes)
{
    out = BitScanResult{};
    const TrsHeader& h = file->header();
    if (first_byte < 0 || n_bytes <= 0 || first_byte + n_bytes > h.data_length) {
        error = "Those data bytes are outside the file's per-trace data.";
        return false;
    }
    const int64_t raw_ns = n_samples_req == 0
        ? h.num_samples - first_sample
        : std::min<int64_t>(n_samples_req, h.num_samples - first_sample);
    if (raw_ns <= 0) { error = "The sample range is empty."; return false; }
    const int64_t L = processedLength(pipeline, raw_ns);
    if (L <= 0) { error = "No samples after pipeline"; return false; }
    const int64_t N = static_cast<int64_t>(traces.size());
    if (N < 4) { error = "Need at least 4 traces after excluding discarded ones."; return false; }
    const int64_t M = int64_t(n_bytes) * 8;

    const int64_t block = std::clamp<int64_t>(
        block_bytes / blockBytesPerSample(N, (raw_ns + L - 1) / L), 1, L);
    const int64_t n_blocks = (L + block - 1) / block;
    if (n_blocks > 1 && pipelineRequiresWholeTrace(pipeline)) {
        error = "The pipeline has a stage that needs whole traces, but " + std::to_string(N)
              + " traces × " + std::to_string(L) + " samples don't fit in memory at once. Shorten "
              "the range to at most " + std::to_string(block) + " samples (after the pipeline), or "
              "remove that stage.";
        return false;
    }

    // Which traces are in each bit's group 1.
    std::vector<std::vector<int32_t>> members(static_cast<size_t>(M));
    for (int64_t i = 0; i < N; i++) {
        const auto d = file->readData(first_trace + traces[static_cast<size_t>(i)]);
        for (int32_t b = 0; b < n_bytes; b++) {
            const size_t at = static_cast<size_t>(first_byte + b);
            const uint8_t v = at < d.size() ? d[at] : uint8_t(0);
            for (int bit = 0; bit < 8; bit++)
                if ((v >> bit) & 1) members[static_cast<size_t>(b * 8 + bit)].push_back(static_cast<int32_t>(i));
        }
    }

    out.n_bits     = static_cast<int32_t>(M);
    out.n_samples  = static_cast<int32_t>(L);
    out.n_traces   = static_cast<int32_t>(N);
    out.first_byte = first_byte;
    out.n0.resize(static_cast<size_t>(M));
    out.n1.resize(static_cast<size_t>(M));
    out.peak_t.assign(static_cast<size_t>(M), 0.0f);
    out.peak_sample.assign(static_cast<size_t>(M), 0);
    out.n_over.assign(static_cast<size_t>(M), 0);
    for (int64_t k = 0; k < M; k++) {
        out.n1[static_cast<size_t>(k)] = static_cast<int64_t>(members[static_cast<size_t>(k)].size());
        out.n0[static_cast<size_t>(k)] = N - out.n1[static_cast<size_t>(k)];
    }
    if (keep_curves) out.t.assign(static_cast<size_t>(M) * static_cast<size_t>(L), 0.0f);

    const int threads = scanThreads();
    const int64_t bit_batch = std::max<int64_t>(threads, 8);   // between cancel checks
    const int64_t per_block = N + M;
    const int64_t total = n_blocks * per_block;
    std::vector<float> x;
    std::vector<double> sum_all, sq_all;
    for (int64_t bi = 0; bi < n_blocks; bi++) {
        const int64_t p0 = bi * block, len = std::min(block, L - p0);
        x.assign(static_cast<size_t>(N * len), 0.0f);
        const int64_t base = bi * per_block;
        if (!loadSampleBlock(file, pipeline, first_trace, traces, shifts, first_sample, raw_ns, L, p0, len,
                             x.data(), [&](int64_t done) { return !progress || progress(base + done, total); })) {
            error = "Cancelled.";
            return false;
        }
        sum_all.assign(static_cast<size_t>(len), 0.0);
        sq_all.assign(static_cast<size_t>(len), 0.0);
        for (int64_t i = 0; i < N; i++) {
            const float* r = x.data() + i * len;
            for (int64_t s = 0; s < len; s++) {
                const double v = r[s];
                sum_all[static_cast<size_t>(s)] += v;
                sq_all[static_cast<size_t>(s)]  += v * v;
            }
        }
        for (int64_t b0 = 0; b0 < M; b0 += bit_batch) {
            const int64_t b1 = std::min(M, b0 + bit_batch);
            #pragma omp parallel num_threads(threads)
            {
                std::vector<double> sum1(static_cast<size_t>(len)), sq1(static_cast<size_t>(len));
                #pragma omp for schedule(dynamic)
                for (int64_t k = b0; k < b1; k++) {
                    const double n1 = static_cast<double>(out.n1[static_cast<size_t>(k)]);
                    const double n0 = static_cast<double>(out.n0[static_cast<size_t>(k)]);
                    if (n0 < 2 || n1 < 2) continue;
                    std::fill(sum1.begin(), sum1.end(), 0.0);
                    std::fill(sq1.begin(), sq1.end(), 0.0);
                    for (int32_t i : members[static_cast<size_t>(k)]) {
                        const float* r = x.data() + static_cast<int64_t>(i) * len;
                        for (int64_t s = 0; s < len; s++) {
                            const double v = r[s];
                            sum1[static_cast<size_t>(s)] += v;
                            sq1[static_cast<size_t>(s)]  += v * v;
                        }
                    }
                    float* curve = keep_curves ? out.t.data() + k * L + p0 : nullptr;
                    float& peak = out.peak_t[static_cast<size_t>(k)];
                    for (int64_t s = 0; s < len; s++) {
                        const double s1 = sum1[static_cast<size_t>(s)];
                        const double s0 = sum_all[static_cast<size_t>(s)] - s1;
                        const double q1 = sq1[static_cast<size_t>(s)];
                        const double q0 = sq_all[static_cast<size_t>(s)] - q1;
                        const double var1 = std::max(0.0, (q1 - s1 * s1 / n1) / (n1 - 1));
                        const double var0 = std::max(0.0, (q0 - s0 * s0 / n0) / (n0 - 1));
                        const double denom = std::sqrt(var0 / n0 + var1 / n1);
                        const float t = denom > 0.0 ? static_cast<float>((s0 / n0 - s1 / n1) / denom) : 0.0f;
                        if (curve) curve[s] = t;
                        if (std::fabs(t) > std::fabs(peak)) {
                            peak = t;
                            out.peak_sample[static_cast<size_t>(k)] = static_cast<int32_t>(p0 + s);
                        }
                        if (std::fabs(t) > 4.5f) out.n_over[static_cast<size_t>(k)]++;
                    }
                }
            }
            if (progress && !progress(base + N + b1, total)) {
                error = "Cancelled.";
                return false;
            }
        }
    }
    return true;
}
