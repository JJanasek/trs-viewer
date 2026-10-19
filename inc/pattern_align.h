#pragma once
// Pattern alignment: the user marks several regions ("patterns") on the
// traces — typically while a Window Resample stage makes the structure
// visible — and each pattern is aligned on its own. Only the per-trace
// shifts are kept, so analyses (t-test, …) can later read the *raw* traces
// pattern by pattern, each with its own shifts.
//
// Alignment of one pattern runs in two stages:
//   1. coarse: alignByXCorr / alignByPeak on the current pipeline (fast on a
//      resampled signal, but only accurate to about one resampling step);
//   2. refine: normalised cross-correlation at full raw resolution, within
//      ±(one resampling step + 1) of the coarse shift, using the pipeline
//      with every sample-count-changing stage removed (so filters, |x|,
//      moving averages still shape the signal, but nothing is decimated).

#include "align.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct PatternDef {
    std::string name;
    int64_t raw_start   = 0;     // region [raw_start, raw_end) in raw samples
    int64_t raw_end     = 0;
    int32_t search_half = 100;   // ± raw samples each trace may move

    enum class Method { XCorr = 0, Peak = 1 };
    Method  method   = Method::XCorr;
    bool    peak_abs = true;     // Peak: argmax |v| instead of argmax v
    float   min_corr = -2.0f;    // < -1 keeps every trace
    int32_t ref_offset = 0;      // template trace(s), relative to first_trace
    int32_t ref_count  = 1;
    bool    refine_raw = true;   // stage 2 above
};

struct PatternResult {
    bool    aligned     = false;
    int32_t first_trace = 0;     // shifts[i] belongs to trace first_trace + i
    std::vector<int32_t> shifts; // raw samples; kAlignDiscardShift = excluded
    std::vector<float>   scores; // NCC per trace (may be empty for Peak), see below
    // Scores are the normalised cross-correlation on the signal the pattern
    // was aligned on (e.g. the resampled one), not after the raw
    // refinement: raw samples are noisier, and a threshold meant for the
    // aligned signal would otherwise throw most traces away.
    float   median_score = NAN;   // over every scored trace, excluded ones too
    int32_t discarded    = 0;
    int32_t at_limit     = 0;     // best match on the edge of the search window
};

struct Pattern {
    PatternDef    def;
    PatternResult res;
};

// Exact raw start position of processed sample p — e.g. p·hop for a window
// resampler — for any chain of stages (see rawCountForProcessedCount).
int64_t rawStartForProcessedIndex(const std::vector<std::shared_ptr<ITransform>>& pipeline,
                                  int64_t p, int64_t max_raw);

// The stages of `pipeline` that keep the sample count, cloned. Used for the
// raw-resolution refinement.
std::vector<std::shared_ptr<ITransform>> countPreservingStages(
    const std::vector<std::shared_ptr<ITransform>>& pipeline, int64_t probe_len);

bool alignPattern(TrsFile* file,
                  const std::vector<std::shared_ptr<ITransform>>& pipeline,
                  int32_t first_trace, int32_t num_traces,
                  const PatternDef& def, PatternResult& out,
                  AlignProgress progress, std::string& error);

// Shift files with a pattern column. Format:
//   # trs-viewer pattern shifts v1
//   # pattern <index> <raw_start> <raw_end> <name…>
//   pattern,trace,offset
// Discarded traces are omitted. Old single-alignment files (trace,offset)
// load as one pattern covering [0, num_samples).
bool savePatternShifts(const std::string& path, const std::vector<Pattern>& patterns,
                       std::string& error);
bool loadPatternShifts(const std::string& path, int64_t num_samples,
                       std::vector<Pattern>& patterns, std::string& error);
