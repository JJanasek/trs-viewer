#pragma once
// A first-order t-test for every bit of a range of data bytes, for example all
// 256 bits of a 32-byte key. The traces are read once, not once per bit: each
// block of samples is held in memory for all traces, and every bit's group 1 is
// summed from it (group 0 is the total minus group 1).

#include "processing.h"
#include "trs_file.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

constexpr int64_t kBitScanBlockBytes = int64_t(512) << 20;

struct BitScanResult {
    int32_t n_bits    = 0;    // rows: row = 8 * (byte - first_byte) + bit
    int32_t n_samples = 0;
    int32_t n_traces  = 0;    // traces used
    int32_t first_byte = 0;
    std::vector<float>   t;   // n_bits × n_samples, row-major; empty if curves weren't kept
    // Per bit. A bit with fewer than 2 traces on either side has t = 0 throughout.
    std::vector<int64_t> n0, n1;
    std::vector<float>   peak_t;       // the t of largest magnitude (signed)
    std::vector<int32_t> peak_sample;
    std::vector<int64_t> n_over;       // samples with |t| > 4.5

    bool hasCurves() const { return !t.empty(); }
};

// Working memory of the block, the data bytes and the per-bit statistics.
int64_t estimateBitScanBytes(int64_t n_traces, int64_t n_bits, int64_t n_samples);

// Tests bits of data bytes [first_byte, first_byte + n_bytes), over samples
// [first_sample, first_sample + n_samples) (0 = to the end), after the pipeline.
// `traces` and `shifts` as in correlateCpa(). keep_curves: also the full
// t-curves, n_bits × samples floats. progress(done, total) → false to cancel.
// Safe on a worker thread.
bool computeBitScan(TrsFile* file, int32_t first_trace, const std::vector<int32_t>& traces,
                    int64_t first_sample, int64_t n_samples, const std::vector<int32_t>& shifts,
                    const std::vector<std::shared_ptr<ITransform>>& pipeline,
                    int32_t first_byte, int32_t n_bytes, bool keep_curves, BitScanResult& out,
                    const std::function<bool(int64_t, int64_t)>& progress, std::string& error,
                    int64_t block_bytes = kBitScanBlockBytes);
