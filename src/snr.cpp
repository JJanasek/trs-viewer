#include "snr.h"
#include <algorithm>
#include <cmath>
#include <numeric>

SNRAccumulator::SNRAccumulator(int32_t num_samples, int32_t num_classes)
    : num_samples_(num_samples), num_classes_(num_classes)
{
    sum_ .assign(num_classes, std::vector<double>(num_samples, 0.0));
    sum2_.assign(num_classes, std::vector<double>(num_samples, 0.0));
    N_   .assign(num_classes, 0);
}

void SNRAccumulator::addTrace(int32_t k, const float* samples, int32_t count) {
    if (k < 0 || k >= num_classes_) return;
    int32_t n = std::min(count, num_samples_);
    double* s  = sum_[k].data();
    double* s2 = sum2_[k].data();
    for (int32_t i = 0; i < n; i++) {
        double v = static_cast<double>(samples[i]);
        s[i]  += v;
        s2[i] += v * v;
    }
    N_[k]++;
}

int64_t SNRAccumulator::countClass(int32_t k) const {
    return (k >= 0 && k < num_classes_) ? N_[k] : 0;
}

int64_t SNRAccumulator::totalTraces() const {
    return std::accumulate(N_.begin(), N_.end(), int64_t{0});
}

bool SNRAccumulator::compute(std::vector<float>& out, std::string& error) const {
    // Need at least 2 classes with >= 2 traces.
    int active = 0;
    for (int32_t k = 0; k < num_classes_; k++)
        if (N_[k] >= 2) active++;
    if (active < 2) {
        error = "Need at least 2 classes with >= 2 traces each (got " +
                std::to_string(active) + ").";
        return false;
    }

    int64_t N_total = totalTraces();
    double  N_d     = static_cast<double>(N_total);
    out.resize(num_samples_);

    for (int32_t s = 0; s < num_samples_; s++) {
        // Global mean
        double global_sum = 0.0;
        for (int32_t k = 0; k < num_classes_; k++)
            if (N_[k] > 0) global_sum += sum_[k][s];
        double global_mean = global_sum / N_d;

        // Signal: frequency-weighted variance of class means
        double signal = 0.0;
        // Noise: frequency-weighted mean of within-class sample variances
        double noise  = 0.0;

        for (int32_t k = 0; k < num_classes_; k++) {
            if (N_[k] < 1) continue;
            double nk   = static_cast<double>(N_[k]);
            double mean_k = sum_[k][s] / nk;
            double diff   = mean_k - global_mean;
            signal += nk * diff * diff;

            if (N_[k] >= 2) {
                // Unbiased sample variance for class k
                double var_k = (sum2_[k][s] - sum_[k][s] * mean_k) / (nk - 1.0);
                if (var_k < 0.0) var_k = 0.0;
                noise += nk * var_k;
            }
        }

        signal /= N_d;
        noise  /= N_d;

        out[s] = (noise > 0.0) ? static_cast<float>(signal / noise) : 0.0f;
    }
    return true;
}
