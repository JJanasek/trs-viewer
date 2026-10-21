#include "ttest.h"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
// Welch t for one sample from its group sums. Shared by compute() and
// computeMerged() so both give exactly the same result.
inline float welchT(double s0, double q0, double s1, double q1, double n0, double n1) {
    double mean0 = s0 / n0;
    double mean1 = s1 / n1;
    // Variance using sum-of-squares formula
    double var0 = (q0 - s0 * mean0) / (n0 - 1.0);
    double var1 = (q1 - s1 * mean1) / (n1 - 1.0);
    if (var0 < 0.0) var0 = 0.0;
    if (var1 < 0.0) var1 = 0.0;
    double denom = std::sqrt(var0 / n0 + var1 / n1);
    return (denom > 0.0) ? static_cast<float>((mean0 - mean1) / denom) : 0.0f;
}

std::string notEnoughTraces(int64_t n0, int64_t n1) {
    return "Need >= 2 traces per group (Group 0: " + std::to_string(n0) +
           ", Group 1: " + std::to_string(n1) + ").";
}
}  // namespace

TTestAccumulator::TTestAccumulator(int32_t num_samples)
    : num_samples_(num_samples)
{
    N_[0] = N_[1] = 0;
    for (int g = 0; g < 2; g++) {
        sum_[g].assign(num_samples, 0.0);
        sum2_[g].assign(num_samples, 0.0);
    }
}

void TTestAccumulator::reset() {
    N_[0] = N_[1] = 0;
    for (int g = 0; g < 2; g++) {
        std::fill(sum_[g].begin(), sum_[g].end(), 0.0);
        std::fill(sum2_[g].begin(), sum2_[g].end(), 0.0);
    }
}

void TTestAccumulator::addTrace(int group, const float* samples, int32_t count) {
    if (group < 0 || group > 1) return;
    int32_t n = std::min(count, num_samples_);
    double* s  = sum_[group].data();
    double* s2 = sum2_[group].data();
    for (int32_t i = 0; i < n; i++) {
        double v = static_cast<double>(samples[i]);
        s[i]  += v;
        s2[i] += v * v;
    }
    N_[group]++;
}

void TTestAccumulator::addTraces(const int* groups, const float* const* bufs,
                                 const int32_t* counts, int n, int threads) {
    int32_t width = 0;
    for (int k = 0; k < n; k++) {
        if (groups[k] < 0 || groups[k] > 1) continue;
        width = std::max(width, std::min(counts[k], num_samples_));
    }
    // Slices of at least 16K samples; tiny traces stay on one thread.
    constexpr int32_t kMinSlice = 1 << 14;
    const int32_t n_slices = static_cast<int32_t>(std::clamp<int64_t>(
        width / kMinSlice, 1, std::max(1, threads)));
    #pragma omp parallel for schedule(static) num_threads(n_slices) if(n_slices > 1)
    for (int32_t sl = 0; sl < n_slices; sl++) {
        const int32_t lo = static_cast<int32_t>(static_cast<int64_t>(width) * sl / n_slices);
        const int32_t hi = static_cast<int32_t>(static_cast<int64_t>(width) * (sl + 1) / n_slices);
        for (int k = 0; k < n; k++) {
            const int g = groups[k];
            if (g < 0 || g > 1) continue;
            const int32_t end = std::min({hi, counts[k], num_samples_});
            const float* x = bufs[k];
            double* s  = sum_[g].data();
            double* s2 = sum2_[g].data();
            for (int32_t i = lo; i < end; i++) {
                const double v = static_cast<double>(x[i]);
                s[i]  += v;
                s2[i] += v * v;
            }
        }
    }
    for (int k = 0; k < n; k++)
        if (groups[k] == 0 || groups[k] == 1) N_[groups[k]]++;
}

void TTestAccumulator::mergeFrom(const TTestAccumulator& other) {
    for (int g = 0; g < 2; g++) {
        for (int32_t i = 0; i < num_samples_; i++) {
            sum_[g][i]  += other.sum_[g][i];
            sum2_[g][i] += other.sum2_[g][i];
        }
        N_[g] += other.N_[g];
    }
}

void TTestAccumulator::computeWelchDf(std::vector<double>& df_out) const {
    df_out.resize(num_samples_);
    double n0 = static_cast<double>(N_[0]);
    double n1 = static_cast<double>(N_[1]);
    for (int32_t s = 0; s < num_samples_; s++) {
        double mean0 = sum_[0][s] / n0;
        double mean1 = sum_[1][s] / n1;
        double var0 = (sum2_[0][s] - sum_[0][s] * mean0) / (n0 - 1.0);
        double var1 = (sum2_[1][s] - sum_[1][s] * mean1) / (n1 - 1.0);
        if (var0 < 0.0) var0 = 0.0;
        if (var1 < 0.0) var1 = 0.0;
        double u0  = var0 / n0;
        double u1  = var1 / n1;
        double num = (u0 + u1) * (u0 + u1);
        double den = u0 * u0 / (n0 - 1.0) + u1 * u1 / (n1 - 1.0);
        df_out[s]  = (den > 1e-300) ? num / den : n0 + n1 - 2.0;
    }
}

void TTestAccumulator::groupMeans(int g, std::vector<double>& out) const {
    out.assign(static_cast<size_t>(num_samples_), 0.0);
    if ((g != 0 && g != 1) || N_[g] == 0) return;
    const double n = static_cast<double>(N_[g]);
    for (int32_t s = 0; s < num_samples_; s++) out[static_cast<size_t>(s)] = sum_[g][s] / n;
}

int64_t TTestAccumulator::countGroup(int g) const {
    return (g == 0 || g == 1) ? N_[g] : 0;
}

bool TTestAccumulator::compute(std::vector<float>& out, std::string& error, int threads) const {
    if (N_[0] < 2 || N_[1] < 2) {
        error = notEnoughTraces(N_[0], N_[1]);
        return false;
    }
    out.resize(num_samples_);
    const double n0 = static_cast<double>(N_[0]);
    const double n1 = static_cast<double>(N_[1]);
    (void)threads;
    #pragma omp parallel for schedule(static) num_threads(std::max(1, threads)) if(threads > 1 && num_samples_ > (1 << 16))
    for (int32_t s = 0; s < num_samples_; s++)
        out[s] = welchT(sum_[0][s], sum2_[0][s], sum_[1][s], sum2_[1][s], n0, n1);
    return true;
}

bool TTestAccumulator::computeMerged(const std::vector<TTestAccumulator>& parts,
                                     std::vector<float>& out, std::string& error,
                                     int threads) {
    int64_t N0 = 0, N1 = 0;
    for (const auto& p : parts) { N0 += p.N_[0]; N1 += p.N_[1]; }
    if (N0 < 2 || N1 < 2) {
        error = notEnoughTraces(N0, N1);
        return false;
    }
    const int32_t ns = parts.front().num_samples_;
    out.resize(static_cast<size_t>(ns));
    const double n0 = static_cast<double>(N0);
    const double n1 = static_cast<double>(N1);
    const int np = static_cast<int>(parts.size());
    float* o = out.data();
    (void)threads;
    #pragma omp parallel for schedule(static) num_threads(std::max(1, threads))
    for (int32_t s = 0; s < ns; s++) {
        // Starting from 0.0 and adding the parts in order matches
        // reset() + mergeFrom() exactly.
        double s0 = 0.0, q0 = 0.0, s1 = 0.0, q1 = 0.0;
        for (int k = 0; k < np; k++) {
            const TTestAccumulator& p = parts[static_cast<size_t>(k)];
            s0 += p.sum_[0][s];  q0 += p.sum2_[0][s];
            s1 += p.sum_[1][s];  q1 += p.sum2_[1][s];
        }
        o[s] = welchT(s0, q0, s1, q1, n0, n1);
    }
    return true;
}
