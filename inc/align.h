#pragma once

#include "processing.h"
#include "trs_file.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Progress callback: return false to cancel.
using AlignProgress = std::function<bool(int done, int total)>;

// Sentinel shift value meaning "this trace did not meet the alignment
// quality threshold and must be excluded" rather than "shift by this amount."
// Every consumer of a shifts vector that may carry discarded entries (the
// align dialog's own preview/apply, t-test, CPA, cross-correlation) checks
// for this value and skips the trace entirely instead of applying it as an
// offset.
inline constexpr int32_t kAlignDiscardShift = INT32_MIN;

struct AlignResult {
    // shifts[i]: the feature in trace i is shifts[i] samples later than in the
    // reference trace.  Positive → advance read pointer (skip first shifts[i]
    // raw samples); negative → pad with zeros at the start.  A value of
    // kAlignDiscardShift means the trace fell below the correlation
    // threshold passed to alignByXCorr and should be excluded, not shifted.
    std::vector<int32_t> shifts;

    // scores[i]: normalised cross-correlation ([-1, 1]) between trace i and
    // the reference at the chosen shift. Only populated by alignByXCorr; the
    // reference trace itself gets a score of 1.0. Empty when produced by
    // alignByPeak, which has no equivalent match-quality metric.
    std::vector<float> scores;
};

// ---------------------------------------------------------------------------
// Peak alignment
// ---------------------------------------------------------------------------
// Finds the peak (argmax|v| or argmax v) in the reference region of the
// reference trace.  Each other trace is searched over ±search_half samples
// around that position; the shift that maps its peak onto the reference peak
// is recorded.  If `pipeline` is non-empty it is applied to every loaded
// window first (reset + apply, same as the t-test accumulation path) so
// alignment sees the same processed samples as the rest of the app.  If the
// pipeline changes the sample count (decimation, stride, FFT/STFT, …),
// positions found in the processed buffer are rescaled back to raw sample
// offsets, so the returned shifts always apply to raw sample positions.
bool alignByPeak(
    TrsFile*       file,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
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
// normalised cross-correlation with that template.  See alignByPeak for the
// `pipeline` contract.
//
// min_correlation: traces whose best NCC score is below this value get
// out.shifts[i] = kAlignDiscardShift instead of the computed shift (their
// score is still recorded in out.scores[i]). Pass -2.0 (or anything < -1) to
// disable filtering — NCC never goes below -1, so nothing is ever discarded.
bool alignByXCorr(
    TrsFile*       file,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int32_t        first_trace,
    int32_t        num_traces,
    int32_t        ref_trace_offset,
    int64_t        ref_first_sample,
    int64_t        ref_num_samples,
    int32_t        search_half,
    float          min_correlation,
    AlignResult&   out,
    AlignProgress  progress,
    std::string&   error);
