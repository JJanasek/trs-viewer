#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Online SNR accumulator: SNR[s] = Var(E[T[s]|class]) / E[Var(T[s]|class)]
//
// Supports up to num_classes distinct integer class labels [0, num_classes).
// Memory: 2 * num_classes * num_samples * sizeof(double).
class SNRAccumulator {
public:
    SNRAccumulator(int32_t num_samples, int32_t num_classes);

    // class_label must be in [0, num_classes). Silently ignored otherwise.
    void addTrace(int32_t class_label, const float* samples, int32_t count);

    // Compute per-sample SNR into out (resized to num_samples).
    // Returns false if fewer than 2 classes have >= 2 traces each.
    bool compute(std::vector<float>& out, std::string& error) const;

    int32_t numClasses()  const { return num_classes_; }
    int32_t numSamples()  const { return num_samples_; }
    int64_t countClass(int32_t k) const;
    int64_t totalTraces() const;

private:
    int32_t num_samples_;
    int32_t num_classes_;
    std::vector<std::vector<double>> sum_;   // [class][sample]
    std::vector<std::vector<double>> sum2_;  // [class][sample]
    std::vector<int64_t>             N_;     // [class]
};
