#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Online Welch t-test accumulator (per-sample).
// Memory: 4 × num_samples × sizeof(double) bytes.
// Call addTrace() once per trace, then compute().
class TTestAccumulator {
public:
    explicit TTestAccumulator(int32_t num_samples);
    void reset();
    // group: 0 or 1.  Any other value is silently ignored.
    void addTrace(int group, const float* samples, int32_t count);
    // Compute Welch t-statistics into out (resized to num_samples).
    bool compute(std::vector<float>& out, std::string& error) const;
    // Compute per-sample Welch-Satterthwaite degrees of freedom.
    // Requires N_[0] >= 2 and N_[1] >= 2.
    void computeWelchDf(std::vector<double>& df_out) const;
    int64_t countGroup(int g) const;
    int32_t numSamples() const { return num_samples_; }
    int64_t estimatedBytes() const {
        return static_cast<int64_t>(num_samples_) * 4LL * static_cast<int64_t>(sizeof(double));
    }
private:
    int32_t             num_samples_;
    std::vector<double> sum_[2];
    std::vector<double> sum2_[2];
    int64_t             N_[2];
};
