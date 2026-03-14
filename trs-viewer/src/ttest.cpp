#include "ttest.h"
#include <algorithm>
#include <cmath>

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

int64_t TTestAccumulator::countGroup(int g) const {
    return (g == 0 || g == 1) ? N_[g] : 0;
}

bool TTestAccumulator::compute(std::vector<float>& out, std::string& error) const {
    if (N_[0] < 2 || N_[1] < 2) {
        error = "Need >= 2 traces per group (Group 0: " +
                std::to_string(N_[0]) + ", Group 1: " + std::to_string(N_[1]) + ").";
        return false;
    }
    out.resize(num_samples_);
    double n0 = static_cast<double>(N_[0]);
    double n1 = static_cast<double>(N_[1]);
    for (int32_t s = 0; s < num_samples_; s++) {
        double mean0 = sum_[0][s] / n0;
        double mean1 = sum_[1][s] / n1;
        // Variance using sum-of-squares formula
        double var0 = (sum2_[0][s] - sum_[0][s] * mean0) / (n0 - 1.0);
        double var1 = (sum2_[1][s] - sum_[1][s] * mean1) / (n1 - 1.0);
        if (var0 < 0.0) var0 = 0.0;
        if (var1 < 0.0) var1 = 0.0;
        double denom = std::sqrt(var0 / n0 + var1 / n1);
        out[s] = (denom > 0.0) ? static_cast<float>((mean0 - mean1) / denom) : 0.0f;
    }
    return true;
}
