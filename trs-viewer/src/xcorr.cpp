#include "xcorr.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#ifdef __linux__
#  include <fstream>
#  include <string>
#endif

// Returns available system memory in MB, or -1 if unknown.
static double availableMemMB() {
#ifdef __linux__
    std::ifstream f("/proc/meminfo");
    std::string key;
    long val;
    while (f >> key >> val) {
        if (key == "MemAvailable:") return val / 1024.0;
        std::string unit; f >> unit;
    }
#endif
    return -1.0;  // unknown — fall back to fixed limit
}

#ifdef _OPENMP
#  include <omp.h>
#endif

// Number of rows/columns processed per block when upcasting float→double for BLAS calls.
// Controls the trade-off between peak memory and BLAS efficiency.
static constexpr int XCORR_BLOCK = 512;

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
bool computeXCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        first_sample,
    int64_t        num_samples,
    int32_t        stride,
    XCorrMethod    method,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const std::vector<int32_t>& shifts,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error)
{
    out = XCorrResult{};
    const TrsHeader& h = file->header();

    // Clamp ranges
    if (first_trace + num_traces > h.num_traces)
        num_traces = h.num_traces - first_trace;
    if (num_traces < 2) { error = "Need at least 2 traces."; return false; }

    if (num_samples <= 0 || first_sample + num_samples > h.num_samples)
        num_samples = h.num_samples - first_sample;
    if (num_samples <= 0) { error = "No samples in range."; return false; }

    if (stride < 1) stride = 1;

    // Effective sample count after pipeline transforms
    int64_t effective_n = num_samples;
    for (const auto& t : pipeline)
        effective_n = t->transformedCount(effective_n);
    if (effective_n <= 0) { error = "Pipeline produces 0 samples."; return false; }

    const int32_t M = static_cast<int32_t>((effective_n + stride - 1) / stride);
    if (M < 2) { error = "Stride too large, fewer than 2 output samples."; return false; }

    const int n = num_traces;

    // Memory guard.
    // Baseline peak: C (M×M doubles) + output (M×M floats).
    // Dual/MP: A stored as float (M×n×4). Chunked double blocks (XCORR_BLOCK rows) used for G and AV.
    //   Phase G:   A_float(M*n*4) + G(n*n*8) + block_tmp(XCORR_BLOCK*n*8)
    //   Phase AV:  A_float(M*n*4) + AV(M*k*8, k≤n) + block_tmp(XCORR_BLOCK*n*8)
    //   Phase C:   AV(M*n*8, worst) + C_eig(M*M*8) + output(M*M*4)  [A freed before C]
    {
        double peak_mb;
        if (method == XCorrMethod::Baseline) {
            peak_mb = (static_cast<double>(M) * M * 8.0   // C doubles
                     + static_cast<double>(M) * M * 4.0)  // output floats
                    / (1024.0 * 1024.0);
        } else {
            double phase_G  = static_cast<double>(M)*n*4 + static_cast<double>(n)*n*8
                            + XCORR_BLOCK*static_cast<double>(n)*8;
            double phase_AV = static_cast<double>(M)*n*4 + static_cast<double>(M)*n*8
                            + XCORR_BLOCK*static_cast<double>(n)*8;
            double phase_C  = static_cast<double>(M)*n*8   // AV (k≤n worst case)
                            + static_cast<double>(M)*M*8    // C_eig doubles
                            + static_cast<double>(M)*M*4;   // output floats
            peak_mb = std::max({phase_G, phase_AV, phase_C}) / (1024.0 * 1024.0);
        }
        // Use 80% of actual available memory as limit; fall back to 4 GB on unknown platforms.
        double avail = availableMemMB();
        double limit = (avail > 0) ? avail * 0.80 : 4096.0;
        if (peak_mb > limit) {
            int peak_i  = static_cast<int>(peak_mb);
            int avail_i = (avail > 0) ? static_cast<int>(avail) : -1;
            error = "Estimated peak memory " + std::to_string(peak_i) + " MB exceeds "
                  + (avail_i > 0 ? std::to_string(static_cast<int>(limit)) + " MB (80% of "
                                   + std::to_string(avail_i) + " MB available)"
                                 : "4 GB limit")
                  + ". Increase stride or reduce trace/sample count.";
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // loadTrace: read raw samples for trace ti, apply pipeline, sub-sample
    //            at xcorr stride → M floats written to out_m[0..M-1].
    // -----------------------------------------------------------------------
    std::vector<float> raw_full(static_cast<size_t>(num_samples));
    auto loadTrace = [&](int ti, float* out_m) {
        int32_t src   = first_trace + ti;
        int32_t shift = (ti < static_cast<int>(shifts.size())) ? shifts[ti] : 0;
        const int64_t total_samples = h.num_samples;
        const int64_t adj_start = first_sample + shift;
        std::fill(raw_full.begin(), raw_full.end(), 0.0f);
        if (adj_start < total_samples && adj_start + num_samples > 0) {
            int64_t src_start = std::max<int64_t>(0, adj_start);
            int64_t src_end   = std::min<int64_t>(total_samples, adj_start + num_samples);
            int64_t dst_off   = src_start - adj_start;
            file->readSamples(src, src_start, src_end - src_start, raw_full.data() + dst_off);
        }
        for (const auto& t : pipeline) t->reset();
        int64_t n_out = num_samples;
        for (const auto& t : pipeline)
            n_out = t->apply(raw_full.data(), n_out, 0);
        for (int j = 0; j < M; j++) {
            int64_t idx = static_cast<int64_t>(j) * stride;
            out_m[j] = (idx < n_out) ? raw_full[static_cast<size_t>(idx)] : 0.0f;
        }
    };

    const int total_phases = (method == XCorrMethod::Baseline) ? 2 : 4;
    int phase_done = 0;

    std::vector<float> raw(static_cast<size_t>(M));

    // -----------------------------------------------------------------------
    // Phase 1: Welford's online algorithm for per-sample mean and variance.
    // Avoids catastrophic cancellation from E[X²]-E[X]² when DC offset is large.
    // -----------------------------------------------------------------------
    std::vector<double> wf_mean(static_cast<size_t>(M), 0.0);
    std::vector<double> wf_M2  (static_cast<size_t>(M), 0.0);

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(phase_done * n + ti, total_phases * n)) {
            error = "Cancelled."; return false;
        }
        loadTrace(ti, raw.data());
        for (int j = 0; j < M; j++) {
            double x     = static_cast<double>(raw[static_cast<size_t>(j)]);
            double delta = x - wf_mean[static_cast<size_t>(j)];
            wf_mean[static_cast<size_t>(j)] += delta / (ti + 1);
            wf_M2  [static_cast<size_t>(j)] += delta * (x - wf_mean[static_cast<size_t>(j)]);
        }
    }
    phase_done++;

    Eigen::VectorXd mean_v(M), inv_std_v(M);
    for (int j = 0; j < M; j++) {
        double var   = wf_M2[static_cast<size_t>(j)] / n;
        mean_v[j]    = wf_mean[static_cast<size_t>(j)];
        inv_std_v[j] = (var > 1e-30) ? 1.0 / std::sqrt(var) : 0.0;
    }

    // -----------------------------------------------------------------------
    // Phase 2a (Baseline): per-trace rank-1 update.
    // C accumulated in double; mirrored via triangularView; cast to float at output.
    // -----------------------------------------------------------------------
    if (method == XCorrMethod::Baseline) {
        try {
            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(M, M);

            for (int ti = 0; ti < n; ti++) {
                if (progress && !progress(phase_done * n + ti, total_phases * n)) {
                    error = "Cancelled."; return false;
                }
                loadTrace(ti, raw.data());
                Eigen::Map<Eigen::VectorXf> rv(raw.data(), M);
                Eigen::VectorXd xn = (rv.cast<double>() - mean_v).cwiseProduct(inv_std_v);
                C.selfadjointView<Eigen::Lower>().rankUpdate(xn, 1.0 / n);
            }

            C.triangularView<Eigen::StrictlyUpper>() = C.transpose();

            out.matrix.resize(static_cast<size_t>(M) * static_cast<size_t>(M));
            Eigen::Map<Eigen::MatrixXf>(out.matrix.data(), M, M) = C.cast<float>();
        } catch (const std::bad_alloc&) {
            error = "Out of memory. Increase stride or reduce sample/trace count.";
            return false;
        }
        out.M        = M;
        out.rows     = M;
        out.cols     = M;
        out.n_traces = n;
        out.method   = method;
        return true;
    }

    // -----------------------------------------------------------------------
    // Phase 2b (Dual / MPCleaned): load all traces into A (M×n, col = trace).
    // Stored as float to halve memory; double precision used only for the
    // Welford-normalised per-column fill.
    // -----------------------------------------------------------------------
    Eigen::MatrixXf A_eig;
    try { A_eig.resize(M, n); }
    catch (const std::bad_alloc&) {
        error = "Out of memory allocating trace matrix. Increase stride or reduce count.";
        return false;
    }

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(phase_done * n + ti, total_phases * n)) {
            error = "Cancelled."; return false;
        }
        loadTrace(ti, raw.data());
        Eigen::Map<Eigen::VectorXf> rv(raw.data(), M);
        A_eig.col(ti) = ((rv.cast<double>() - mean_v).cwiseProduct(inv_std_v)).cast<float>();
    }
    phase_done++;

    // -----------------------------------------------------------------------
    // Phase 3a: Gram matrix G = A^T A / M
    // Computed in double via 512-row blocks to avoid materialising a full
    // double copy of A (peak extra = XCORR_BLOCK × n × 8 bytes).
    // -----------------------------------------------------------------------
    Eigen::MatrixXd G_eig = Eigen::MatrixXd::Zero(n, n);
    for (int r = 0; r < M; r += XCORR_BLOCK) {
        int len = std::min(XCORR_BLOCK, M - r);
        Eigen::MatrixXd blk = A_eig.middleRows(r, len).cast<double>();
        G_eig.noalias() += blk.transpose() * blk;
    }
    G_eig /= static_cast<double>(M);

    if (progress && !progress(phase_done * n, total_phases * n)) {
        error = "Cancelled."; return false;
    }

    // -----------------------------------------------------------------------
    // Phase 3b: Eigendecompose G
    // -----------------------------------------------------------------------
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(G_eig);
    if (solver.info() != Eigen::Success) {
        error = "Eigendecomposition failed."; return false;
    }
    G_eig.resize(0, 0);   // free memory
    phase_done++;

    if (progress && !progress(phase_done * n, total_phases * n)) {
        error = "Cancelled."; return false;
    }

    // Eigenvalues ascending; compute Marchenko-Pastur upper edge
    double gamma       = static_cast<double>(n) / M;
    double lambda_plus = (1.0 + std::sqrt(gamma)) * (1.0 + std::sqrt(gamma));

    int n_signal = 0;
    for (int k = n - 1; k >= 0; k--)
        if (solver.eigenvalues()[k] > lambda_plus) n_signal++;

    // -----------------------------------------------------------------------
    // Phase 4: C = (A * V_sel) * (A * V_sel)^T / n  — two DGEMMs
    // -----------------------------------------------------------------------
    int k_select;
    Eigen::MatrixXd V_sel;
    if (method == XCorrMethod::MPCleaned) {
        k_select = n_signal;
        if (k_select == 0) {
            out.matrix.assign(static_cast<size_t>(M) * static_cast<size_t>(M), 0.0f);
            out.M = M; out.rows = M; out.cols = M;
            out.n_traces = n; out.method = method;
            out.lambda_plus = lambda_plus; out.n_signal = 0;
            return true;
        }
        V_sel = solver.eigenvectors().rightCols(k_select);
    } else {
        k_select = 0;
        for (int k = 0; k < n; k++)
            if (solver.eigenvalues()[k] > 0.0) k_select++;
        V_sel = solver.eigenvectors().rightCols(k_select);
    }

    Eigen::MatrixXd AV, C_eig;
    try {
        // AV: same block strategy — avoids a full double copy of A_eig.
        AV.resize(M, k_select);
        for (int r = 0; r < M; r += XCORR_BLOCK) {
            int len = std::min(XCORR_BLOCK, M - r);
            AV.middleRows(r, len).noalias() =
                A_eig.middleRows(r, len).cast<double>() * V_sel;
        }
        A_eig.resize(0, 0);   // free float A before allocating C_eig

        C_eig = AV * AV.transpose() / static_cast<double>(n);
        out.matrix.resize(static_cast<size_t>(M) * static_cast<size_t>(M));
        Eigen::Map<Eigen::MatrixXf>(out.matrix.data(), M, M) = C_eig.cast<float>();
    } catch (const std::bad_alloc&) {
        error = "Out of memory during correlation matrix computation. Increase stride.";
        return false;
    }

    if (progress && !progress(total_phases * n - 1, total_phases * n)) {
        error = "Cancelled."; return false;
    }
    out.M           = M;
    out.rows        = M;
    out.cols        = M;
    out.n_traces    = n;
    out.method      = method;
    out.lambda_plus = lambda_plus;
    out.n_signal    = n_signal;
    return true;
}

// ---------------------------------------------------------------------------
// Naive scalar MxM Pearson correlation — reference / debug implementation.
// No Eigen, no BLAS, no OpenMP, no Welford.
// Phase 1: load all traces into a plain double matrix (M rows × n cols).
// Phase 2: compute per-sample sum1/sum2 → mean/inv_std via E[X²]-E[X]².
// Phase 3: normalise in-place (subtract mean, multiply inv_std).
// Phase 4: explicit i/j double loop: C[i,j] = dot(A[i,:], A[j,:]) / n.
// ---------------------------------------------------------------------------
bool computeXCorrNaive(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        first_sample,
    int64_t        num_samples,
    int32_t        stride,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const std::vector<int32_t>& shifts,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error)
{
    out = XCorrResult{};
    const TrsHeader& h = file->header();

    if (first_trace + num_traces > h.num_traces)
        num_traces = h.num_traces - first_trace;
    if (num_traces < 2) { error = "Need at least 2 traces."; return false; }

    if (num_samples <= 0 || first_sample + num_samples > h.num_samples)
        num_samples = h.num_samples - first_sample;
    if (num_samples <= 0) { error = "No samples in range."; return false; }

    if (stride < 1) stride = 1;

    // Effective sample count after pipeline
    int64_t effective_n = num_samples;
    for (const auto& t : pipeline)
        effective_n = t->transformedCount(effective_n);
    if (effective_n <= 0) { error = "Pipeline produces 0 samples."; return false; }

    const int32_t M = static_cast<int32_t>((effective_n + stride - 1) / stride);
    if (M < 2) { error = "Stride too large, fewer than 2 output samples."; return false; }

    const int n = num_traces;

    // Memory guard: A (M×n floats) + C (M×M doubles) + output (M×M floats)
    double mem_mb = (static_cast<double>(M) * n  * 4.0
                   + static_cast<double>(M) * M  * 8.0
                   + static_cast<double>(M) * M  * 4.0) / (1024.0 * 1024.0);
    if (mem_mb > 4096.0) {
        error = "Estimated working memory " + std::to_string(static_cast<int>(mem_mb))
              + " MB exceeds 4 GB. Reduce trace count or increase stride.";
        return false;
    }

    // Reuse the same loadTrace lambda pattern (local raw buffer)
    std::vector<float> raw_full(static_cast<size_t>(num_samples));
    std::vector<float> raw(static_cast<size_t>(M));

    auto loadTrace = [&](int ti, float* out_m) {
        int32_t src   = first_trace + ti;
        int32_t shift = (ti < static_cast<int>(shifts.size())) ? shifts[ti] : 0;
        const int64_t total_samples = h.num_samples;
        const int64_t adj_start = first_sample + shift;
        std::fill(raw_full.begin(), raw_full.end(), 0.0f);
        if (adj_start < total_samples && adj_start + num_samples > 0) {
            int64_t src_start = std::max<int64_t>(0, adj_start);
            int64_t src_end   = std::min<int64_t>(total_samples, adj_start + num_samples);
            int64_t dst_off   = src_start - adj_start;
            file->readSamples(src, src_start, src_end - src_start, raw_full.data() + dst_off);
        }
        for (const auto& t : pipeline) t->reset();
        int64_t n_out = num_samples;
        for (const auto& t : pipeline)
            n_out = t->apply(raw_full.data(), n_out, 0);
        for (int j = 0; j < M; j++) {
            int64_t idx = static_cast<int64_t>(j) * stride;
            out_m[j] = (idx < n_out) ? raw_full[static_cast<size_t>(idx)] : 0.0f;
        }
    };

    // -----------------------------------------------------------------------
    // Phase 1: load all traces into A[j * n + ti]  (row j = sample j, col ti = trace ti)
    // Stored as float; double precision used only during normalization and dot products.
    // -----------------------------------------------------------------------
    std::vector<float> A(static_cast<size_t>(M) * static_cast<size_t>(n), 0.0f);

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(ti, 4 * n)) { error = "Cancelled."; return false; }
        loadTrace(ti, raw.data());
        for (int j = 0; j < M; j++)
            A[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(ti)]
                = raw[static_cast<size_t>(j)];
    }

    // -----------------------------------------------------------------------
    // Phase 2: per-sample mean and variance via naive sum1/sum2 (E[X²]-E[X]²)
    // -----------------------------------------------------------------------
    std::vector<double> mean_v(static_cast<size_t>(M), 0.0);
    std::vector<double> inv_std(static_cast<size_t>(M), 0.0);

    for (int j = 0; j < M; j++) {
        if (progress && !progress(n + j * n / M, 4 * n)) { error = "Cancelled."; return false; }
        double sum1 = 0.0, sum2 = 0.0;
        for (int ti = 0; ti < n; ti++) {
            double v = static_cast<double>(
                A[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(ti)]);
            sum1 += v;
            sum2 += v * v;
        }
        double mean = sum1 / n;
        double var  = sum2 / n - mean * mean;
        mean_v[static_cast<size_t>(j)] = mean;
        inv_std[static_cast<size_t>(j)] = (var > 1e-30) ? 1.0 / std::sqrt(var) : 0.0;
    }

    // -----------------------------------------------------------------------
    // Phase 3: normalise A in-place: A[j][ti] = (A[j][ti] - mean[j]) * inv_std[j]
    // -----------------------------------------------------------------------
    for (int j = 0; j < M; j++) {
        double mu = mean_v[static_cast<size_t>(j)];
        double is = inv_std[static_cast<size_t>(j)];
        for (int ti = 0; ti < n; ti++) {
            size_t idx = static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(ti);
            A[idx] = static_cast<float>((static_cast<double>(A[idx]) - mu) * is);
        }
    }

    if (progress && !progress(2 * n, 4 * n)) { error = "Cancelled."; return false; }

    // -----------------------------------------------------------------------
    // Phase 4: C[i,j] = (1/n) * sum_t A[i][t] * A[j][t]   (explicit scalar loop)
    // Only lower triangle computed; mirrored manually.
    // -----------------------------------------------------------------------
    std::vector<double> C(static_cast<size_t>(M) * static_cast<size_t>(M), 0.0);

    for (int i = 0; i < M; i++) {
        if (progress && !progress(2 * n + i * n / M, 4 * n)) { error = "Cancelled."; return false; }
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            const float* row_i = A.data() + static_cast<size_t>(i) * static_cast<size_t>(n);
            const float* row_j = A.data() + static_cast<size_t>(j) * static_cast<size_t>(n);
            for (int ti = 0; ti < n; ti++)
                dot += static_cast<double>(row_i[ti]) * static_cast<double>(row_j[ti]);
            double val = dot / n;
            C[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(j)] = val;
            C[static_cast<size_t>(j) * static_cast<size_t>(M) + static_cast<size_t>(i)] = val;
        }
    }

    if (progress && !progress(4 * n - 1, 4 * n)) { error = "Cancelled."; return false; }

    out.matrix.resize(static_cast<size_t>(M) * static_cast<size_t>(M));
    for (size_t k = 0; k < out.matrix.size(); k++)
        out.matrix[k] = static_cast<float>(C[k]);
    out.M        = M;
    out.rows     = M;
    out.cols     = M;
    out.n_traces = n;
    out.method   = XCorrMethod::Baseline;
    return true;
}

// ---------------------------------------------------------------------------
// Two-window normalised cross-correlation: C (M_search × M_ref)
// ---------------------------------------------------------------------------

// Load one window from a trace, apply pipeline, sub-sample at stride.
static void loadWindow(TrsFile* file, int32_t first_trace,
                       const std::vector<std::shared_ptr<ITransform>>& pipeline,
                       int ti, int64_t first_sample, int64_t num_samples,
                       int32_t stride, int32_t M_out,
                       float* work_buf, float* out_buf,
                       int32_t shift = 0)
{
    const int64_t total_samples = file->header().num_samples;
    int32_t src = first_trace + ti;
    const int64_t adj_start = first_sample + shift;
    std::fill(work_buf, work_buf + num_samples, 0.0f);
    if (adj_start < total_samples && adj_start + num_samples > 0) {
        int64_t src_start = std::max<int64_t>(0, adj_start);
        int64_t src_end   = std::min<int64_t>(total_samples, adj_start + num_samples);
        int64_t dst_off   = src_start - adj_start;
        file->readSamples(src, src_start, src_end - src_start, work_buf + dst_off);
    }
    for (const auto& t : pipeline) t->reset();
    int64_t n_out = num_samples;
    for (const auto& t : pipeline) n_out = t->apply(work_buf, n_out, 0);
    for (int j = 0; j < M_out; j++) {
        int64_t idx = static_cast<int64_t>(j) * stride;
        out_buf[j] = (idx < n_out) ? work_buf[static_cast<size_t>(idx)] : 0.0f;
    }
}

bool computeTwoWindowCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int64_t        search_first_sample,
    int64_t        search_num_samples,
    int32_t        stride,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    const std::vector<int32_t>& shifts,
    XCorrResult&   out,
    XCorrProgress  progress,
    std::string&   error)
{
    out = XCorrResult{};
    const TrsHeader& h = file->header();

    if (first_trace + num_traces > h.num_traces)
        num_traces = h.num_traces - first_trace;
    if (num_traces < 2) { error = "Need at least 2 traces."; return false; }

    if (stride < 1) stride = 1;

    // Effective sample counts after pipeline
    int64_t ref_eff    = ref_num_samples;
    int64_t search_eff = search_num_samples;
    for (const auto& t : pipeline) {
        ref_eff    = t->transformedCount(ref_eff);
        search_eff = t->transformedCount(search_eff);
    }
    if (ref_eff <= 0 || search_eff <= 0) {
        error = "Pipeline produces 0 samples."; return false;
    }

    const int32_t M_ref    = static_cast<int32_t>((ref_eff    + stride - 1) / stride);
    const int32_t M_search = static_cast<int32_t>((search_eff + stride - 1) / stride);
    const int n = num_traces;

    // Memory check:
    //   A_ref (M_ref×n floats) + A_search (M_search×n floats)
    //   + block_tmp (512×max(M_ref,M_search) doubles) + C (M_search×M_ref doubles)
    {
        double mem_mb = (static_cast<double>(M_ref)    * n * 4.0
                       + static_cast<double>(M_search) * n * 4.0
                       + 512.0 * std::max(M_ref, M_search) * 8.0
                       + static_cast<double>(M_search) * M_ref * 8.0) / (1024.0 * 1024.0);
        if (mem_mb > 4096.0) {
            error = "Estimated memory " + std::to_string(static_cast<int>(mem_mb))
                  + " MB exceeds 4 GB. Reduce ranges or increase stride.";
            return false;
        }
    }

    // Per-trace work buffers
    std::vector<float> work_r(static_cast<size_t>(ref_num_samples));
    std::vector<float> work_s(static_cast<size_t>(search_num_samples));
    std::vector<float> raw_r(static_cast<size_t>(M_ref));
    std::vector<float> raw_s(static_cast<size_t>(M_search));

    // -----------------------------------------------------------------------
    // Phase 1: Welford's online mean+variance for both windows.
    // -----------------------------------------------------------------------
    std::vector<double> ref_wf_mean(M_ref,    0.0), ref_wf_M2(M_ref,    0.0);
    std::vector<double> sea_wf_mean(M_search, 0.0), sea_wf_M2(M_search, 0.0);

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(ti, 3 * n)) { error = "Cancelled."; return false; }

        int32_t shift = (ti < static_cast<int>(shifts.size())) ? shifts[ti] : 0;
        loadWindow(file, first_trace, pipeline, ti,
                   ref_first_sample, ref_num_samples, stride, M_ref,
                   work_r.data(), raw_r.data(), shift);
        for (int j = 0; j < M_ref; j++) {
            double x     = static_cast<double>(raw_r[j]);
            double delta = x - ref_wf_mean[j];
            ref_wf_mean[j] += delta / (ti + 1);
            ref_wf_M2  [j] += delta * (x - ref_wf_mean[j]);
        }

        loadWindow(file, first_trace, pipeline, ti,
                   search_first_sample, search_num_samples, stride, M_search,
                   work_s.data(), raw_s.data(), shift);
        for (int j = 0; j < M_search; j++) {
            double x     = static_cast<double>(raw_s[j]);
            double delta = x - sea_wf_mean[j];
            sea_wf_mean[j] += delta / (ti + 1);
            sea_wf_M2  [j] += delta * (x - sea_wf_mean[j]);
        }
    }

    Eigen::VectorXd ref_mean(M_ref),    ref_inv_std(M_ref);
    Eigen::VectorXd sea_mean(M_search), sea_inv_std(M_search);
    for (int j = 0; j < M_ref; j++) {
        double var     = ref_wf_M2[j] / n;
        ref_mean[j]    = ref_wf_mean[j];
        ref_inv_std[j] = (var > 1e-30) ? 1.0 / std::sqrt(var) : 0.0;
    }
    for (int j = 0; j < M_search; j++) {
        double var     = sea_wf_M2[j] / n;
        sea_mean[j]    = sea_wf_mean[j];
        sea_inv_std[j] = (var > 1e-30) ? 1.0 / std::sqrt(var) : 0.0;
    }

    // -----------------------------------------------------------------------
    // Phase 2: build A_ref (M_ref × n) and A_search (M_search × n) as float.
    // -----------------------------------------------------------------------
    Eigen::MatrixXf A_ref(M_ref, n), A_search(M_search, n);

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(n + ti, 3 * n)) { error = "Cancelled."; return false; }

        int32_t shift = (ti < static_cast<int>(shifts.size())) ? shifts[ti] : 0;
        loadWindow(file, first_trace, pipeline, ti,
                   ref_first_sample, ref_num_samples, stride, M_ref,
                   work_r.data(), raw_r.data(), shift);
        Eigen::Map<Eigen::VectorXf> vr(raw_r.data(), M_ref);
        A_ref.col(ti) = ((vr.cast<double>() - ref_mean).cwiseProduct(ref_inv_std)).cast<float>();

        loadWindow(file, first_trace, pipeline, ti,
                   search_first_sample, search_num_samples, stride, M_search,
                   work_s.data(), raw_s.data(), shift);
        Eigen::Map<Eigen::VectorXf> vs(raw_s.data(), M_search);
        A_search.col(ti) = ((vs.cast<double>() - sea_mean).cwiseProduct(sea_inv_std)).cast<float>();
    }

    if (progress && !progress(2 * n, 3 * n)) { error = "Cancelled."; return false; }

    // -----------------------------------------------------------------------
    // Phase 3: C = A_search * A_ref^T / n  (DGEMM, M_search × M_ref)
    // Accumulated in double via 512-column (trace) blocks to avoid a full
    // double copy of either A matrix.
    // -----------------------------------------------------------------------
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(M_search, M_ref);
    for (int c = 0; c < n; c += XCORR_BLOCK) {
        int len = std::min(XCORR_BLOCK, n - c);
        C.noalias() += A_search.middleCols(c, len).cast<double>()
                     * A_ref.middleCols(c, len).cast<double>().transpose();
    }
    C /= static_cast<double>(n);

    if (progress && !progress(3 * n - 1, 3 * n)) { error = "Cancelled."; return false; }

    out.matrix.resize(static_cast<size_t>(M_search) * M_ref);
    // Row-major storage: row = search sample, col = ref sample; cast to float at boundary.
    Eigen::Map<Eigen::MatrixXf>(out.matrix.data(), M_search, M_ref) = C.cast<float>();
    out.M        = 0;
    out.rows     = M_search;
    out.cols     = M_ref;
    out.n_traces = n;
    out.method   = XCorrMethod::TwoWindow;
    return true;
}
