#pragma once

#include "trs_file.h"

#include <functional>
#include <string>
#include <vector>

// Progress callback: return false to cancel.
using AlignProgress = std::function<bool(int done, int total)>;

struct AlignResult {
    // shifts[i]: the feature in trace i is shifts[i] samples later than in the
    // reference trace.  Positive → advance read pointer (skip first shifts[i]
    // raw samples); negative → pad with zeros at the start.
    std::vector<int32_t> shifts;
};

// ---------------------------------------------------------------------------
// Peak alignment
// ---------------------------------------------------------------------------
// Finds the peak (argmax|v| or argmax v) in the reference region of the
// reference trace.  Each other trace is searched over ±search_half samples
// around that position; the shift that maps its peak onto the reference peak
// is recorded.  Works on raw (pre-pipeline) samples.
bool alignByPeak(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,   // index within [0, num_traces)
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int32_t        search_half,        // ±samples to search in each trace
    bool           use_abs,            // true → peak = argmax|v|; false → argmax v
    AlignResult&   out,
    AlignProgress  progress,
    std::string&   error);

// ---------------------------------------------------------------------------
// Cross-correlation alignment
// ---------------------------------------------------------------------------
// Uses the reference region of the reference trace as a normalised template.
// Each trace is searched over ±search_half lags for the lag that maximises the
// normalised cross-correlation with that template.
bool alignByXCorr(
    TrsFile*       file,
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int32_t        search_half,
    AlignResult&   out,
    AlignProgress  progress,
    std::string&   error);
