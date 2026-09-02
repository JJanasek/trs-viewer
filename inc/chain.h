#pragma once

// ---------------------------------------------------------------------------
// Chain: a saveable/loadable, ordered list of high-level operations (add a
// transform, clear the pipeline, align traces, reload live, export) that can
// be authored once via the Chain Editor and replayed with one click instead
// of re-doing the same sequence of dialogs by hand every time.
//
// ChainStep is intentionally a flat struct rather than a real tagged union —
// every field is a plain int/double/bool/QString, so JSON (de)serialization
// in saveChain()/loadChain() is a mechanical field-by-field mapping (only
// the fields relevant to a step's `kind` are read/written, via a switch).
// Execution (turning a step into an actual pipeline/align/export action)
// lives in MainWindow, since it needs the live Dataset/plot state; this
// header only owns the step data model, its JSON persistence, and the
// AddTransform step's two-way mapping to/from a real ITransform instance.
// ---------------------------------------------------------------------------

#include "processing.h"

#include <QString>

#include <cstdint>
#include <memory>
#include <vector>

struct ChainStep {
    enum class Kind {
        AddTransform, ClearPipeline, Align, Reload, Export,
        ExportShifts, LoadShifts, RunTTest,
    };
    Kind kind = Kind::AddTransform;

    // --- AddTransform --------------------------------------------------
    // transform_type is the combo_transform_ index (0-10, see
    // MainWindow::createTransform()) — reused directly as the type tag so
    // building/describing a step never needs runtime type detection: the
    // caller already knows which concrete ITransform subclass idx maps to.
    int    transform_type = 0;
    int    i_window  = 0;   // MovingAverage / WindowResample / STFT window size
    int    i_stride  = 0;   // StrideResample stride
    int    i_hop     = 0;   // STFT hop size (derived from overlap at author time)
    double d_overlap = 0.0; // WindowResample / STFT overlap, 0-1
    double d_value   = 0.0; // Offset value / Scale factor / GaussianNoise std
    double d_q       = 0.0; // Biquad Q
    double d_cutoff  = 0.0; // Biquad cutoff (fraction of Nyquist)
    int    i_choice  = 0;   // FFT/STFT window-function index, or Biquad filter-type index

    // --- Align (mirrors onAlignTraces()'s parameter set; region/search are
    //     in pipeline-processed sample units, exactly like that dialog) ---
    int     align_method    = 0;     // 0 = Peak, 1 = Cross-correlation
    int32_t first_trace     = 0;
    int32_t trace_count     = 0;
    int32_t ref_offset      = 0;     // reference trace index within [0, trace_count)
    int32_t ref_count       = 1;     // >1 averages this many consecutive traces
                                      // starting at ref_offset into the template
                                      // instead of using ref_offset alone — see
                                      // alignByPeak/alignByXCorr's ref_trace_count.
    int64_t ref_first       = 0;     // processed-sample units
    int64_t ref_len         = 0;     // processed-sample units
    int32_t search_half     = 0;     // processed-sample units
    bool    peak_use_abs    = true;  // Peak only
    bool    discard_enabled = false; // XCorr only
    double  min_corr        = 0.5;   // XCorr only
    int     output_mode     = 0;     // 0 = avg-pad, 1 = zero-pad, 2 = crop
    int32_t align_tile_size    = 0;  // 0 = untiled; >0 = per-(trace,tile) alignment,
                                      // see Dataset::align_tile_size.
    int32_t align_preview_tile = 0;  // which tile's preview to bake on Apply — there's
                                      // no dialog at execution time to ask, unlike the
                                      // interactive "Apply to Main View" button.

    // --- Export -----------------------------------------------------------
    int     export_format      = 0;  // 0 = TRS, 1 = NPY, 2 = NPZ
    int32_t exp_first          = 0;
    int32_t exp_count          = 0;
    bool    use_last_alignment = true;
    QString path;                    // empty => prompt via QFileDialog when this step runs
    // ExportShifts/LoadShifts also use `path` (same empty-means-prompt rule);
    // ExportShifts writes activeDs().align_shifts, LoadShifts sets it.
    int32_t tile_idx           = -1; // Export/RunTTest only: which tile's shifts+window to
                                      // use when use_last_alignment and the dataset's
                                      // alignment is tiled. -1 = unset (JSON backward
                                      // compat) — treated as tile 0 at run time.

    // --- RunTTest (mirrors onRunTTest()'s parameter set; reuses
    //     first_trace/trace_count above for the trace range,
    //     use_last_alignment above for its own alignment checkbox, and
    //     tile_idx above for which tile to run when tiled) ----------------
    int64_t ttest_first_sample = 0;
    int64_t ttest_n_samples    = 0;  // 0 = all
    int32_t ttest_byte_idx     = 0;  // ignored when the file has a "ttest" param
    bool    ttest_abs          = false; // report |t| instead of the signed t-statistic

    // One-line label for the Chain Editor's step list.
    QString summary() const;
};

bool saveChain(const QString& path, const std::vector<ChainStep>& steps, QString& err);
bool loadChain(const QString& path, std::vector<ChainStep>& steps, QString& err);

// Non-interactive counterpart of MainWindow::createTransform(): builds a
// fresh ITransform directly from a step's stored fields, no dialogs.
// Returns nullptr for an unrecognised transform_type.
std::shared_ptr<ITransform> buildTransformFromStep(const ChainStep& step);

// The reverse direction — reads a just-constructed transform's own
// parameters back out via its public getters into a ChainStep, given the
// combo_transform_ index that was used to create it (so, like
// buildTransformFromStep, no runtime type detection is needed: the caller
// already knows tx's concrete type from idx). Used when authoring an
// AddTransform step: call MainWindow::createTransform(idx) as normal, then
// describeTransformStep(idx, *tx) to capture it into the chain.
ChainStep describeTransformStep(int transform_type, const ITransform& tx);
