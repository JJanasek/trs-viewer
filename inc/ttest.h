#pragma once
#include <cstdint>
#include <functional>
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
    // Adds n traces at once, split by sample range over `threads` OpenMP
    // threads (each thread owns a slice of every sum). groups[k] < 0 skips
    // trace k. Same result as addTrace() on each in order — bit-identical,
    // since every sample still sees the traces in the same order. Lets
    // parallel runs share one accumulator instead of one per thread.
    void addTraces(const int* groups, const float* const* bufs, const int32_t* counts,
                   int n, int threads);
    // Adds n traces whose values are generated on the fly: gen(k, lo, hi, y)
    // writes samples [lo, hi) of trace k into y. The samples are split into
    // small blocks spread over `threads` OpenMP threads, and each block takes
    // all n traces while its sums stay in cache — for derived values (e.g.
    // products of sample pairs) far too many to buffer per trace. gen is
    // called concurrently. Same result as addTrace() on each in order.
    void addGenerated(const int* groups, int n, int threads,
                      const std::function<void(int k, int32_t lo, int32_t hi, float* y)>& gen);
    // Adds another accumulator's totals into this one (same num_samples).
    // Used to combine per-thread partial accumulators after parallel
    // accumulation, since addTrace() itself isn't safe to call concurrently
    // on a shared instance.
    void mergeFrom(const TTestAccumulator& other);
    // Compute Welch t-statistics into out (resized to num_samples).
    bool compute(std::vector<float>& out, std::string& error, int threads = 1) const;
    // Welch t over the sum of several partial accumulators (same num_samples)
    // without building that sum first: equivalent to reset() + mergeFrom()
    // of each part in order + compute() — bit-identical, since the per-sample
    // additions happen in the same order — but one pass over the parts
    // instead of three, no extra 4 × num_samples doubles, and split over
    // `threads` OpenMP threads. Used for live t-test snapshots.
    static bool computeMerged(const std::vector<TTestAccumulator>& parts,
                              std::vector<float>& out, std::string& error,
                              int threads = 1);
    // Compute per-sample Welch-Satterthwaite degrees of freedom.
    // Requires N_[0] >= 2 and N_[1] >= 2.
    void computeWelchDf(std::vector<double>& df_out) const;
    int64_t countGroup(int g) const;
    // Per-sample mean of group g (resized to num_samples; zeros if empty).
    void groupMeans(int g, std::vector<double>& out) const;
    // A new accumulator over the given samples of this one, in that order
    // (e.g. one chosen cell per position of a larger map).
    TTestAccumulator gather(const std::vector<int64_t>& samples) const;
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
