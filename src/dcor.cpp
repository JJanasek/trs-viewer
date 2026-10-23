#include "dcor.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

double dcorSquared(const DistanceSums& s) {
    const double n0 = static_cast<double>(s.n0), n1 = static_cast<double>(s.n1);
    const double n  = n0 + n1;
    if (n < 4 || n0 < 1 || n1 < 1) return 0.0;
    // The U-centred estimator Ω_n (eq. 4) for X×class, X×X and class×class.
    // With b_ij = 1 for pairs of different classes: Σ_{i≠j} a_ij b_ij =
    // 2·cross, b_i· = n1 (class 0) or n0 (class 1), b·· = 2·n0·n1, and the
    // row sums of a per class are 2·within + cross.
    const double d1 = n * (n - 3);
    const double d2 = n * (n - 2) * (n - 3);
    const double d3 = n * (n - 1) * (n - 2) * (n - 3);
    const double a_all = 2.0 * (s.within0 + s.within1 + s.cross);   // a··
    const double r0 = 2.0 * s.within0 + s.cross;                      // Σ_{i∈G0} a_i·
    const double r1 = 2.0 * s.within1 + s.cross;
    const double b_all = 2.0 * n0 * n1;
    const double oxy = 2.0 * s.cross / d1 - 2.0 * (n1 * r0 + n0 * r1) / d2 + a_all * b_all / d3;
    const double oyy = b_all / d1 - 2.0 * n0 * n1 * n / d2 + b_all * b_all / d3;
    const double oxx = s.sq / d1 - 2.0 * s.row_sq / d2 + a_all * a_all / d3;
    if (!(oxx > 0.0) || !(oyy > 0.0)) return 0.0;
    return oxy / std::sqrt(oxx * oyy);
}

float dcorZ(const DistanceSums& s) {
    const double n = static_cast<double>(s.n0 + s.n1);
    return static_cast<float>(std::sqrt(std::max(0.0, n * dcorSquared(s) + 1.0)));
}

double dcorPValue(double z) {
    return std::erfc(z / std::sqrt(2.0));
}

DistanceSums distanceSums1D(const float* x, const uint8_t* g, int64_t n,
                            std::vector<std::pair<float, uint8_t>>& v)
{
    DistanceSums s;
    v.resize(static_cast<size_t>(n));
    double total = 0.0;
    for (int64_t i = 0; i < n; i++) {
        v[static_cast<size_t>(i)] = {x[i], static_cast<uint8_t>(g[i] ? 1 : 0)};
        total += x[i];
    }
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    // One pass in sorted order: everything below x_k is smaller, so
    // Σ_j |x_k − x_j| = x_k·k − (sum below) + (sum above) − x_k·(above).
    const double mean = total / static_cast<double>(std::max<int64_t>(1, n));
    double below = 0.0, all_pairs = 0.0, dev2 = 0.0;
    double cnt[2] = {0, 0}, sum[2] = {0, 0}, within[2] = {0, 0};
    for (int64_t k = 0; k < n; k++) {
        const double xk = v[static_cast<size_t>(k)].first;
        const int    c  = v[static_cast<size_t>(k)].second;
        const double dk = static_cast<double>(k);
        const double row = xk * dk - below + (total - below - xk) - xk * (static_cast<double>(n - 1) - dk);
        s.row_sq += row * row;
        all_pairs += xk * dk - below;
        within[c] += xk * cnt[c] - sum[c];
        below += xk;
        cnt[c] += 1.0;
        sum[c] += xk;
        dev2 += (xk - mean) * (xk - mean);
    }
    s.n0 = static_cast<int64_t>(cnt[0]);
    s.n1 = static_cast<int64_t>(cnt[1]);
    s.within0 = within[0];
    s.within1 = within[1];
    s.cross   = all_pairs - within[0] - within[1];
    s.sq      = 2.0 * static_cast<double>(n) * dev2;   // Σ_{i≠j} (x_i − x_j)²
    return s;
}

DistanceSums distanceSumsHistogram(const uint32_t* c0, const uint32_t* c1, int nbins, double lo) {
    DistanceSums s;
    double n = 0.0, total = 0.0;
    for (int k = 0; k < nbins; k++) {
        const double m = static_cast<double>(c0[k]) + c1[k];
        n += m;
        total += m * (lo + k);
        s.n0 += c0[k];
        s.n1 += c1[k];
    }
    const double mean = n > 0 ? total / n : 0.0;
    // As distanceSums1D, one bin at a time (equal values are 0 apart).
    double cnt_below = 0.0, below = 0.0, all_pairs = 0.0, dev2 = 0.0;
    double cnt[2] = {0, 0}, sum[2] = {0, 0}, within[2] = {0, 0};
    for (int k = 0; k < nbins; k++) {
        const double m0 = c0[k], m1 = c1[k], m = m0 + m1;
        if (m == 0.0) continue;
        const double v = lo + k;
        const double row = v * cnt_below - below + (total - below - m * v) - v * (n - cnt_below - m);
        s.row_sq  += m * row * row;
        all_pairs += m * (v * cnt_below - below);
        within[0] += m0 * (v * cnt[0] - sum[0]);
        within[1] += m1 * (v * cnt[1] - sum[1]);
        cnt_below += m;
        below     += m * v;
        cnt[0] += m0; sum[0] += m0 * v;
        cnt[1] += m1; sum[1] += m1 * v;
        dev2 += m * (v - mean) * (v - mean);
    }
    s.within0 = within[0];
    s.within1 = within[1];
    s.cross   = all_pairs - within[0] - within[1];
    s.sq      = 2.0 * n * dev2;
    return s;
}

DistanceSums distanceSumsRows(float* x, int64_t n, int64_t p, const uint8_t* g, int threads) {
    DistanceSums s;
    for (int64_t i = 0; i < n; i++) (g[i] ? s.n1 : s.n0)++;
    if (n < 2 || p < 1) return s;
    threads = std::max(1, threads);

    // Centre the columns, then |x_i − x_j|² = |x_i|² + |x_j|² − 2 x_i·x_j
    // with the dot products from one matrix product per block pair.
    std::vector<double> mean(static_cast<size_t>(p), 0.0);
    for (int64_t i = 0; i < n; i++) {
        const float* r = x + i * p;
        for (int64_t c = 0; c < p; c++) mean[static_cast<size_t>(c)] += r[c];
    }
    for (double& m : mean) m /= static_cast<double>(n);
    std::vector<double> norm(static_cast<size_t>(n), 0.0);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < n; i++) {
        float* r = x + i * p;
        double acc = 0.0;
        for (int64_t c = 0; c < p; c++) {
            r[c] = static_cast<float>(r[c] - mean[static_cast<size_t>(c)]);
            acc += static_cast<double>(r[c]) * r[c];
        }
        norm[static_cast<size_t>(i)] = acc;
    }
    double norm_sum = 0.0;
    for (double v : norm) norm_sum += v;
    s.sq = 2.0 * static_cast<double>(n) * norm_sum;   // Σ_{i≠j} |x_i − x_j|²

    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    constexpr int64_t kBlock = 512;
    const int64_t nb = (n + kBlock - 1) / kBlock;
    std::vector<std::pair<int64_t, int64_t>> tiles;
    for (int64_t a = 0; a < nb; a++)
        for (int64_t b = a; b < nb; b++) tiles.emplace_back(a, b);

    std::vector<std::vector<double>> rows(static_cast<size_t>(threads));
    std::vector<double> part(static_cast<size_t>(threads) * 3, 0.0);   // within0, within1, cross
    #pragma omp parallel num_threads(threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<double>& row = rows[static_cast<size_t>(tid)];
        row.assign(static_cast<size_t>(n), 0.0);
        double w[3] = {0, 0, 0};
        Eigen::MatrixXf dot;
        // Inside a parallel region Eigen runs each product on this thread.
        #pragma omp for schedule(dynamic)
        for (int64_t t = 0; t < static_cast<int64_t>(tiles.size()); t++) {
            const int64_t i0 = tiles[static_cast<size_t>(t)].first * kBlock;
            const int64_t j0 = tiles[static_cast<size_t>(t)].second * kBlock;
            const int64_t ri = std::min(kBlock, n - i0), rj = std::min(kBlock, n - j0);
            const Eigen::Map<const RowMat> A(x + i0 * p, ri, p);
            const Eigen::Map<const RowMat> B(x + j0 * p, rj, p);
            dot.noalias() = A * B.transpose();
            const bool diag = i0 == j0;
            for (int64_t jj = 0; jj < rj; jj++) {
                const int64_t j = j0 + jj;
                const double nj = norm[static_cast<size_t>(j)];
                double row_j = 0.0;
                for (int64_t ii = 0; ii < (diag ? jj : ri); ii++) {
                    const int64_t i = i0 + ii;
                    const double d = std::sqrt(std::max(
                        0.0, norm[static_cast<size_t>(i)] + nj - 2.0 * static_cast<double>(dot(ii, jj))));
                    row[static_cast<size_t>(i)] += d;
                    row_j += d;
                    w[g[i] != g[j] ? 2 : (g[i] ? 1 : 0)] += d;
                }
                row[static_cast<size_t>(j)] += row_j;
            }
        }
        for (int k = 0; k < 3; k++) part[static_cast<size_t>(tid) * 3 + static_cast<size_t>(k)] = w[k];
    }
    std::vector<double> row_all(static_cast<size_t>(n), 0.0);
    for (const auto& r : rows)
        if (!r.empty())
            for (int64_t i = 0; i < n; i++) row_all[static_cast<size_t>(i)] += r[static_cast<size_t>(i)];
    for (double r : row_all) s.row_sq += r * r;
    for (int t = 0; t < threads; t++) {
        s.within0 += part[static_cast<size_t>(t) * 3 + 0];
        s.within1 += part[static_cast<size_t>(t) * 3 + 1];
        s.cross   += part[static_cast<size_t>(t) * 3 + 2];
    }
    return s;
}
