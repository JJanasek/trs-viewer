#pragma once
// Distance correlation between observations x_1..x_n (numbers, or vectors
// such as a window of a trace) and a 0/1 class: the dependency measure of
// Székely, Rizzo & Bakirov (2007). It is zero only under independence, so it
// catches a difference in any property of the two groups' distributions —
// mean, variance, higher moments, or, for vectors, how points move together —
// not only the mean. Used for non-specific leakage detection with the class
// "fixed vs random" as in Chowdhury & Oswald, "Multivariate Leakage
// Detection", TCHES 2026(2).
//
// With a 0/1 class the unbiased estimator (Székely & Rizzo 2014, the paper's
// eq. 4/5) only needs a few sums of the pairwise distances a_ij = |x_i − x_j|,
// collected in DistanceSums: per-sample values sort in O(n log n) (or bin in
// O(n) for 8-bit samples), vectors take one blocked matrix product.

#include <cstdint>
#include <utility>
#include <vector>

struct DistanceSums {
    int64_t n0 = 0, n1 = 0;                       // observations per class
    double  within0 = 0, within1 = 0, cross = 0;  // Σ a_ij over unordered pairs: both class 0,
                                                  // both class 1, one of each
    double  row_sq = 0;                           // Σ_i (Σ_j a_ij)²
    double  sq     = 0;                           // Σ_{i≠j} a_ij², ordered pairs
};

// Unbiased squared distance correlation R*²(X, class); 0 if undefined (fewer
// than 4 observations, a constant X, or an empty class). Can be slightly
// negative: the estimator is unbiased, so it scatters around 0 when there is
// no dependence.
double dcorSquared(const DistanceSums& s);

// The test statistic. Under independence n·R*² + 1 is approximately — and
// for any dimension at most — χ²₁-distributed (Shen, Panda & Vogelstein 2022,
// the paper's eq. 6), so
//     z = √(max(0, n·R*² + 1))
// behaves like |N(0,1)|: z > 4.5 has the false-positive rate of |t| > 4.5 for
// many traces, and Šidák/Bonferroni thresholds apply unchanged.
float dcorZ(const DistanceSums& s);

// Two-sided p-value of z: P(χ²₁ ≥ z²).
double dcorPValue(double z);

// Scalars x_i with classes g_i ∈ {0, 1}. O(n log n): sorts (value, class)
// pairs in `scratch`.
DistanceSums distanceSums1D(const float* x, const uint8_t* g, int64_t n,
                            std::vector<std::pair<float, uint8_t>>& scratch);

// Integer values lo, lo+1, …, lo+nbins−1 given as per-class counts (e.g. raw
// 8-bit samples binned into 256 bins). O(nbins).
DistanceSums distanceSumsHistogram(const uint32_t* count0, const uint32_t* count1,
                                   int nbins, double lo);

// Vectors: the n rows of x (n × p, row-major) with classes g, Euclidean
// distances between rows, by blocked matrix products on `threads` OpenMP
// threads. x is centred (each column's mean removed) in place, which leaves
// the distances unchanged and keeps them accurate.
DistanceSums distanceSumsRows(float* x, int64_t n, int64_t p, const uint8_t* g, int threads);
