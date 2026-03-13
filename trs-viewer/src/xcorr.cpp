#include "xcorr.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition of n×n matrix G (stored row-major, symmetric).
// Because G is symmetric, treating row-major data as column-major gives G^T = G,
// so Eigen's column-major solver sees the same matrix.
// Output: vals (descending), vecs (rows — row k = k-th eigenvector).
// ---------------------------------------------------------------------------
static bool eigenSolveG(int n, std::vector<double>& G_data,
                         std::vector<double>& vals,
                         std::vector<double>& vecs)
{
    Eigen::Map<Eigen::MatrixXd> Gm(G_data.data(), n, n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Gm);
    if (solver.info() != Eigen::Success) return false;

    // Eigen returns eigenvalues ascending; we need descending.
    vals.resize(static_cast<size_t>(n));
    vecs.resize(static_cast<size_t>(n * n));
    for (int i = 0; i < n; i++) {
        int di = n - 1 - i;
        vals[static_cast<size_t>(i)] = solver.eigenvalues()[di];
        // Store as row i: vecs[i*n + r] = r-th component of eigenvector i
        for (int r = 0; r < n; r++)
            vecs[static_cast<size_t>(i * n + r)] = solver.eigenvectors()(r, di);
    }
    return true;
}

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
    const int32_t M = static_cast<int32_t>((num_samples + stride - 1) / stride);
    if (M < 2) { error = "Stride too large, fewer than 2 output samples."; return false; }

    const int n = num_traces;

    // Memory guard for Dual/MP modes: A is M×n doubles, G is n×n doubles
    if (method != XCorrMethod::Baseline) {
        double mem_mb = (static_cast<double>(M) * n
                       + static_cast<double>(n) * n) * 8.0 / (1024.0 * 1024.0);
        if (mem_mb > 4096.0) {
            error = "Estimated working memory " + std::to_string(static_cast<int>(mem_mb))
                  + " MB exceeds 4 GB. Reduce trace count or increase stride.";
            return false;
        }
    }

    // -----------------------------------------------------------------------
    // Phase 1: per-sample mean and inverse-std (serial — file I/O bound)
    // -----------------------------------------------------------------------
    const int total_phases = (method == XCorrMethod::Baseline) ? 2 : 4;
    int phase_done = 0;

    std::vector<float>  raw(static_cast<size_t>(M));
    std::vector<double> sum1(static_cast<size_t>(M), 0.0);
    std::vector<double> sum2(static_cast<size_t>(M), 0.0);

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(phase_done * n + ti, total_phases * n)) {
            error = "Cancelled."; return false;
        }
        int32_t src = first_trace + ti;
        for (int j = 0; j < M; j++) {
            int64_t s = first_sample + static_cast<int64_t>(j) * stride;
            raw[static_cast<size_t>(j)] = file->readSample(src, s);
        }
        for (int j = 0; j < M; j++) {
            double x = raw[static_cast<size_t>(j)];
            sum1[static_cast<size_t>(j)] += x;
            sum2[static_cast<size_t>(j)] += x * x;
        }
    }
    phase_done++;

    std::vector<double> mean(static_cast<size_t>(M)), inv_std(static_cast<size_t>(M));
    for (int j = 0; j < M; j++) {
        mean[static_cast<size_t>(j)] = sum1[static_cast<size_t>(j)] / n;
        double var = sum2[static_cast<size_t>(j)] / n
                   - mean[static_cast<size_t>(j)] * mean[static_cast<size_t>(j)];
        inv_std[static_cast<size_t>(j)] = (var > 1e-30) ? 1.0 / std::sqrt(var) : 0.0;
    }

    // -----------------------------------------------------------------------
    // Phase 2a (Baseline): streaming outer-product accumulation  C = Xnorm^T Xnorm / n
    // -----------------------------------------------------------------------
    if (method == XCorrMethod::Baseline) {
        std::vector<double> xn(static_cast<size_t>(M));
        std::vector<double> C(static_cast<size_t>(M) * static_cast<size_t>(M), 0.0);

        for (int ti = 0; ti < n; ti++) {
            if (progress && !progress(phase_done * n + ti, total_phases * n)) {
                error = "Cancelled."; return false;
            }
            int32_t src = first_trace + ti;
            for (int j = 0; j < M; j++) {
                int64_t s = first_sample + static_cast<int64_t>(j) * stride;
                raw[static_cast<size_t>(j)] = file->readSample(src, s);
            }
            for (int j = 0; j < M; j++)
                xn[static_cast<size_t>(j)] = (raw[static_cast<size_t>(j)]
                                             - mean[static_cast<size_t>(j)])
                                           * inv_std[static_cast<size_t>(j)];

            // Rank-1 update: C += xn * xn^T  (lower triangle, rows are independent)
#pragma omp parallel for schedule(static)
            for (int r = 0; r < M; r++) {
                double xr = xn[static_cast<size_t>(r)];
                double* Cr = C.data() + static_cast<size_t>(r) * M;
                for (int c2 = 0; c2 <= r; c2++)
                    Cr[c2] += xr * xn[static_cast<size_t>(c2)];
            }
        }

        // Normalise and mirror (rows are independent)
        double inv_n = 1.0 / n;
#pragma omp parallel for schedule(static)
        for (int r = 0; r < M; r++) {
            double* Cr = C.data() + static_cast<size_t>(r) * M;
            for (int c2 = 0; c2 <= r; c2++) {
                double v = Cr[c2] * inv_n;
                Cr[c2] = v;
                C[static_cast<size_t>(c2) * M + r] = v;
            }
        }

        out.matrix.resize(static_cast<size_t>(M) * static_cast<size_t>(M));
#pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < static_cast<int64_t>(M) * M; k++)
            out.matrix[static_cast<size_t>(k)] = static_cast<float>(C[static_cast<size_t>(k)]);

        out.M        = M;
        out.n_traces = n;
        out.method   = method;
        return true;
    }

    // -----------------------------------------------------------------------
    // Phase 2b (Dual / MPCleaned): load all traces into A (M×n, col = trace)
    //   A[j*n + ti] = Xnorm[ti, j]
    // -----------------------------------------------------------------------
    std::vector<double> A(static_cast<size_t>(M) * static_cast<size_t>(n), 0.0);

    for (int ti = 0; ti < n; ti++) {
        if (progress && !progress(phase_done * n + ti, total_phases * n)) {
            error = "Cancelled."; return false;
        }
        int32_t src = first_trace + ti;
        for (int j = 0; j < M; j++) {
            int64_t s = first_sample + static_cast<int64_t>(j) * stride;
            raw[static_cast<size_t>(j)] = file->readSample(src, s);
        }
        for (int j = 0; j < M; j++)
            A[static_cast<size_t>(j * n + ti)] =
                (raw[static_cast<size_t>(j)] - mean[static_cast<size_t>(j)])
                * inv_std[static_cast<size_t>(j)];
    }
    phase_done++;

    // -----------------------------------------------------------------------
    // Phase 3a: Gram matrix G = A^T A / M  (n×n, parallel over rows of G)
    // -----------------------------------------------------------------------
    std::vector<double> G(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);

#pragma omp parallel for schedule(dynamic, 4)
    for (int ti = 0; ti < n; ti++) {
        const double* col_ti = A.data() + ti;  // column ti of A (stride n)
        for (int tk = 0; tk <= ti; tk++) {
            const double* col_tk = A.data() + tk;
            double dot = 0.0;
            for (int j = 0; j < M; j++)
                dot += col_ti[static_cast<size_t>(j) * n]
                     * col_tk[static_cast<size_t>(j) * n];
            G[static_cast<size_t>(ti * n + tk)] =
            G[static_cast<size_t>(tk * n + ti)] = dot / M;
        }
    }

    if (progress && !progress(phase_done * n, total_phases * n)) {
        error = "Cancelled."; return false;
    }

    // -----------------------------------------------------------------------
    // Phase 3b: Eigendecompose G using Eigen (O(n²√n), replaces Jacobi O(n³))
    // -----------------------------------------------------------------------
    std::vector<double> eigenvalues, eigenvecs;
    if (!eigenSolveG(n, G, eigenvalues, eigenvecs)) {
        error = "Eigendecomposition failed."; return false;
    }
    G.clear(); G.shrink_to_fit();  // free Gram matrix memory
    phase_done++;

    if (progress && !progress(phase_done * n, total_phases * n)) {
        error = "Cancelled."; return false;
    }

    // Marchenko-Pastur upper edge: λ+ = (1 + √(n/M))²
    double gamma       = static_cast<double>(n) / M;
    double lambda_plus = (1.0 + std::sqrt(gamma)) * (1.0 + std::sqrt(gamma));

    int n_signal = 0;
    for (int k = 0; k < n; k++)
        if (eigenvalues[static_cast<size_t>(k)] > lambda_plus) n_signal++;

    // -----------------------------------------------------------------------
    // Phase 4: Reconstruct C = sum_{k signal} (A v_k)(A v_k)^T / n
    // -----------------------------------------------------------------------
    std::vector<double> C(static_cast<size_t>(M) * static_cast<size_t>(M), 0.0);
    std::vector<double> Av(static_cast<size_t>(M));

    const int n_selected = (method == XCorrMethod::MPCleaned) ? n_signal : n;
    int processed = 0;

    for (int k = 0; k < n; k++) {
        double lam = eigenvalues[static_cast<size_t>(k)];
        if (method == XCorrMethod::MPCleaned && lam <= lambda_plus) continue;
        if (lam <= 0.0) continue;

        // Av[i] = A[row i] · v_k   (A is M×n, row i has stride 1 in n-direction)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++) {
            const double* row_i = A.data() + static_cast<size_t>(i) * n;
            const double* vk    = eigenvecs.data() + static_cast<size_t>(k) * n;
            double dot = 0.0;
            for (int col = 0; col < n; col++) dot += row_i[col] * vk[col];
            Av[static_cast<size_t>(i)] = dot;
        }

        // C += (Av ⊗ Av) / n  (lower triangle, rows independent)
        double scale = 1.0 / n;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++) {
            double avi = Av[static_cast<size_t>(i)] * scale;
            double* Ci = C.data() + static_cast<size_t>(i) * M;
            for (int j = 0; j <= i; j++)
                Ci[j] += avi * Av[static_cast<size_t>(j)];
        }

        processed++;
        if (progress) {
            int pv = phase_done * n + processed * n / std::max(1, n_selected);
            if (!progress(std::min(pv, total_phases * n - 1), total_phases * n)) {
                error = "Cancelled."; return false;
            }
        }
    }

    // Mirror lower triangle to upper (rows independent)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < i; j++)
            C[static_cast<size_t>(j) * M + i] = C[static_cast<size_t>(i) * M + j];

    out.matrix.resize(static_cast<size_t>(M) * static_cast<size_t>(M));
#pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < static_cast<int64_t>(M) * M; k++)
        out.matrix[static_cast<size_t>(k)] = static_cast<float>(C[static_cast<size_t>(k)]);

    out.M          = M;
    out.n_traces   = n;
    out.method     = method;
    out.lambda_plus = lambda_plus;
    out.n_signal   = n_signal;
    return true;
}
