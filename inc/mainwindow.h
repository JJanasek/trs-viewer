     #pragma once

#include "trs_file.h"
#include "processing.h"
#include "plot_widget.h"
#include "ttest.h"
#include "align.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QPushButton>
#include <QSpinBox>
#include <QTabBar>

#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <vector>

class QVBoxLayout;
class QDockWidget;
class QProgressBar;
class JobManager;
struct ChainStep;

// ---------------------------------------------------------------------------
// Snapshot of mutable per-dataset state, used for undo.
// ---------------------------------------------------------------------------
struct DatasetSnapshot {
    std::vector<std::shared_ptr<ITransform>> pipeline; // each is clone()'d
    std::vector<int32_t> align_shifts;
    int32_t align_first_trace  = 0;
    int64_t align_first_sample = 0;
    int64_t align_n_samples    = 0;
    int32_t align_tile_size    = 0;
    int32_t align_baked_tile   = 0;
    PlotViewState view;
};

// ---------------------------------------------------------------------------
// Per-file state bundle.  MainWindow keeps a vector of these.
// ---------------------------------------------------------------------------
struct Dataset {
    // shared_ptr, not unique_ptr: a background analysis job holds its own
    // reference to the file it is reading, so closing the tab (or the whole
    // dataset going away) while that job is still running unmaps nothing out
    // from under it — the file simply outlives the Dataset until the last
    // job using it finishes. See AnalysisJob in mainwindow.cpp.
    std::shared_ptr<TrsFile>                  file;
    std::vector<std::shared_ptr<ITransform>>  pipeline;
    QString                                   display_name;

    // True for a tab holding a derived result (t-test/SNR/... curve) rather
    // than a real trace file — file is null, and most file-dependent menu
    // actions refuse to run against it.
    bool                 is_result          = false;

    // This tab's own, persistent plot — every tab gets one so several can
    // be shown at once when tiled (see MainWindow::tiled_).
    PlotWidget*          plot_widget        = nullptr;

    // Optional analysis-specific toolbar (t-test/SNR/... controls) shown
    // above plot_widget when this tab holds a derived result.
    QWidget*             extra_toolbar      = nullptr;

    // Alignment state (populated by "Apply to Main View" or drag-align)
    std::vector<int32_t> align_shifts;
    int32_t              align_first_trace  = 0;
    int64_t              align_first_sample = 0;
    int64_t              align_n_samples    = 0;
    // 0 = untiled (align_shifts[local_trace], today's exact behavior).
    // >0 = each trace is num_samples/align_tile_size independent tiles,
    // align_shifts[local_trace*n_tiles+tile_idx] — see resolveAlignmentTile().
    int32_t              align_tile_size    = 0;
    // Which tile the current baked preview (align_shifts + "Apply to Main
    // View") represents; used by rebakeAlignedView() on later pipeline
    // edits. Ignored untiled.
    int32_t              align_baked_tile   = 0;

    // Plot state
    int32_t              plot_first_trace   = 0;
    bool                 plot_file_backed   = false;

    // Undo stack (most-recent first); capped at kMaxUndo
    std::deque<DatasetSnapshot> undo_stack;
    static constexpr int kMaxUndo = 50;
};

// ---------------------------------------------------------------------------
// Tile-aware alignment resolution — see Dataset::align_tile_size above.
// ---------------------------------------------------------------------------

// Result of slicing one tile's plain per-trace shifts + sample window out of
// a (possibly tile-flattened) shifts array — see sliceAlignmentTile()/
// resolveAlignmentTile() in mainwindow.cpp.
struct ResolvedAlignment {
    std::vector<int32_t> shifts;      // plain shifts[local_trace], untiled shape
    int64_t first_sample = 0;         // this tile's (or the whole array's) own window start
    int64_t n_samples    = 0;         // ...and length
};

// Extracts tile_idx's plain per-trace shifts + sample window out of `shifts`.
// Untiled (tile_size <= 0): `shifts` is returned unchanged, with
// first_sample/n_samples taken from fallback_first_sample/fallback_n_samples
// (a dataset's own align_first_sample/align_n_samples, typically). Tiled:
// `shifts` is the flattened shifts[local_trace*n_tiles+tile_idx] array (size
// == num_traces*n_tiles, n_tiles == n_samples_total/tile_size); returns
// tile_idx's own per-trace slice, with first_sample/n_samples set to that
// tile's own [tile_idx*tile_size, +tile_size) window. Malformed/stale tiled
// data (n_tiles<=0, or shifts.size() not divisible by n_tiles) degrades to
// an empty shifts vector — the same "no alignment" fallback every consumer
// already handles — rather than misindexing.
ResolvedAlignment sliceAlignmentTile(const std::vector<int32_t>& shifts,
                                      int32_t tile_size, int64_t n_samples_total,
                                      int32_t tile_idx,
                                      int64_t fallback_first_sample,
                                      int64_t fallback_n_samples);

// Thin Dataset-level wrapper around sliceAlignmentTile() — the form every
// consumer (t-test, export, the baked preview) actually calls.
ResolvedAlignment resolveAlignmentTile(const Dataset& ds, int32_t tile_idx);

// Loops tile_idx over [0, n_samples_total/tile_size), calling alignByPeak/
// alignByXCorr once per tile — unmodified, serially (alignByXCorr already
// parallelises internally via OpenMP; nesting a parallel loop around this
// would oversubscribe) — with ref_first_sample offset by tile_idx*tile_size,
// assembling the per-tile results into one flattened
// out.shifts[local_trace*n_tiles+tile_idx] array (out.scores likewise, for
// XCorr). Requires file->header().num_samples % tile_size == 0 (clear error
// otherwise). progress is called with combined
// (tile_idx*num_traces+done, n_tiles*num_traces) so it spans the whole run;
// returns false (with error set) and leaves `out` untouched — not partially
// filled — on cancellation or a per-tile failure.
bool runTiledAlignment(
    TrsFile* file, const std::vector<std::shared_ptr<ITransform>>& pipeline,
    int32_t first_trace, int32_t num_traces, int32_t ref_trace_offset,
    int32_t ref_trace_count,
    int64_t ref_first_sample, int64_t ref_num_samples, int32_t search_half,
    int32_t tile_size, bool is_peak, bool peak_use_abs, float min_correlation,
    AlignResult& out, AlignProgress progress, std::string& error);

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    void openFile(const QString& path);

private slots:
    void onOpenFile();
    void onImportTraceSubset();
    void onWarmPageCache();
    void onCloseDataset();
    void onSwitchDataset(int idx);
    void onApplyTraces();
    void onAddTransform();
    void onRemoveTransform();
    void onMoveTransformUp();
    void onMoveTransformDown();
    void onResetView();
    void onViewChanged(int64_t start, int64_t end, int64_t total);
    void onMeasurementUpdated(int64_t s1, double v1,
                               int64_t s2, double v2, bool has_p2);
    void onThemeChanged(int index);
    void onExportTrs();
    void onExportPng();
    void onExportPdf();
    void onRunTTest();
    void onCropEdit();
    void onRunXCorr();
    void onAlignTraces();
    void onLoadNpyTTest();
    void onLoadNpyHeatmap();
    void onLoadSpectrumNpz();
    void onOpenNpyTraces();
    void onExportNpy();
    void onExportNpz();
    void onRunCpa();
    void onRunSNR();
    void onRunStaticSNR();
    void onRunFFT();
    void onExportDataset();
    void onDragAlignChanged();
    void onUndoAction();
    void onToggleTile(bool on);
    void onTabMoved(int from, int to);
    void onChainEditor();

private:
    void setupMenuBar();
    void updateFileInfo();
    void updateTraceDataDisplay();
    void rebuildTransformList();
    std::shared_ptr<ITransform> createTransform(int combo_index);
    static QString recentDir(const QString& key);
    static void    updateRecentDir(const QString& key, const QString& file_path);
    void saveSnapshot();
    void restoreSnapshot(DatasetSnapshot snap);
    void updateUndoButton();

    // Switches the active plot's interaction mode back to Pan and re-checks
    // the "Pan" toolbar button (unchecking "↔ Align"/Measure/Box Zoom via
    // their shared exclusive QButtonGroup) — called by both "⟳ Reset" and
    // "Load / Refresh"/"Un-apply Shifts" so a leftover Align/Measure/Box
    // Zoom mode from before doesn't silently carry over into the fresh view.
    void resetInteractionModeToPan();

    // Warms a freshly-opened (lazily memory-mapped) file's page cache with a
    // cancellable, byte-progress QProgressDialog. Only shows the dialog if
    // warming actually takes a while (small files finish before the dialog's
    // minimum duration elapses) — cancelling just stops early, the file is
    // already fully usable via on-demand page faults either way.
    void prefetchWithProgress(TrsFile* file, const QString& label);

    // Opens a new tab holding a derived, pre-computed 1-D result (t-test/
    // SNR/... curve) — same mechanism as opening another trace file, just
    // with is_result=true and no backing TrsFile. Generic across analyses.
    void addResultTab(const std::vector<float>& result, const QString& title,
                       const QColor& color, const QString& trace_label);

    // Creates a new, fully-wired PlotWidget for a tab (dataset or result).
    PlotWidget* createPlotWidgetForTab();
    // Arranges tab widgets in the view container: only the active tab's
    // widget when single-view, every tab's widget stacked when tiled_.
    void updateViewLayout();
    // Finds which dataset owns `pw` (its plot_widget), or -1.
    int  datasetIndexForWidget(PlotWidget* pw) const;
    // If `pw` belongs to a non-active tab, makes it active (side panel only
    // — does not touch any plot's content) so state-changing signals from a
    // tiled, non-selected panel land on the right dataset.
    void activateDatasetForWidget(PlotWidget* pw);

    // Re-applies the *current* pipeline to a tab's stored alignment (shifts +
    // window) and redraws it. Called whenever the pipeline changes on a tab
    // that is showing a static baked-in alignment result (plot_file_backed
    // == false, e.g. after Align Traces -> Apply to Main View) so pipeline
    // edits keep taking visible effect there instead of freezing at whatever
    // the pipeline was when alignment was applied. No-op for file-backed or
    // unaligned/result tabs.
    void rebakeAlignedView();

    // Builds and shows the Align Traces dialog — the actual implementation
    // behind the onAlignTraces() slot. Split out so the Chain Editor's
    // "Align Traces…" add-step action can open the very same interactive
    // dialog (drag-on-plot region, Run, results table) instead of a blind
    // parameter form: pass a non-null onAddToChain and "Apply to Main View"
    // additionally captures its just-used parameters into a ChainStep and
    // hands it to the callback, on top of its normal behavior. Passing
    // nullptr (the plain onAlignTraces() case) leaves that button unchanged.
    void showAlignDialog(std::function<void(const ChainStep&)> onAddToChain = nullptr);

    // Reads `shifts` into `count` (up to `max_display`) baked, aligned
    // traces on `pw` — raw sample window per shifts[i]/output_mode
    // (0=avg-pad,1=zero-pad,2=crop), pipeline applied same as the rest of
    // the app. base_sample/window_len (default 0/0, meaning "the whole
    // trace") bound both the output window and the raw read: 0 for
    // window_len means h.num_samples, otherwise the read is clamped to
    // [base_sample, base_sample+window_len) instead of the whole file — so
    // a tile's own bounds, passed by a tiled caller, stop a large shift from
    // pulling in a neighboring tile's samples. "Crop to common range"
    // (output_mode==2) is only honored when base_sample==0 and
    // window_len==h.num_samples (untiled); a tiled caller requesting it
    // silently falls back to the tile's own window instead. Shared by Align
    // Traces' "Show in New Window"/"Apply to Main View" and the Chain
    // "Align" step. Returns false (and shows a warning on msg_parent) if
    // crop mode leaves no common range, or the user declines a
    // large-allocation warning.
    bool buildAlignedTraces(PlotWidget* pw, const std::vector<int32_t>& shifts,
                             int32_t first_tr, int output_mode, int max_display,
                             QWidget* msg_parent,
                             int64_t base_sample = 0, int64_t window_len = 0);

    // Commits an already-computed, possibly tile-flattened shifts vector as
    // the active dataset's alignment (saveSnapshot + align_first_trace/
    // align_shifts/align_tile_size/align_baked_tile) and bakes a
    // NUM_COLORS-trace preview of tile_idx into the main plot — exactly what
    // Align Traces' "Apply to Main View" button does, factored out so the
    // Chain "Align" step can reuse it after running alignByPeak/alignByXCorr
    // (or runTiledAlignment) itself. tile_size==0/tile_idx==0 is the untiled
    // case (today's exact behavior). Returns false (with err set) if
    // buildAlignedTraces fails.
    bool computeAndStoreAlignment(const std::vector<int32_t>& shifts, int32_t first_tr,
                                   int output_mode, int32_t tile_size, int32_t tile_idx,
                                   QWidget* msg_parent, QString& err);

    // Executes one Chain step against the active dataset. Returns false
    // (with err set) on failure; the Chain Editor's Run loop stops there.
    bool runChainStep(const ChainStep& step, QWidget* msg_parent, QString& err);

    // Builds and shows the Welch t-test configuration dialog — the actual
    // implementation behind the onRunTTest() slot, split out the same way
    // showAlignDialog() is: a non-null onAddToChain adds an "Add to Chain"
    // checkbox that, alongside running the t-test as normal, captures the
    // just-used parameters into a ChainStep for the Chain Editor.
    void showTTestDialog(std::function<void(const ChainStep&)> onAddToChain = nullptr);

    // The parallelised accumulation + Welch computation core of the t-test,
    // shared by showTTestDialog() and the Chain "Run T-test" step. shifts
    // empty => no alignment; when non-empty, shifts[i] applies to absolute
    // trace index (shifts_first_trace + i) — eff_first need not equal
    // shifts_first_trace, so the shift can be applied to any subset of the
    // aligned traces, not just the whole aligned range starting at trace 0
    // of the window. Traces whose absolute index falls outside
    // [shifts_first_trace, shifts_first_trace + shifts.size()) are excluded
    // (counted alongside kAlignDiscardShift-marked ones), not run unshifted.
    // read_lo/read_hi bound the shift-adjusted raw-sample read — [0,
    // h.num_samples) untiled, one tile's own [tile_idx*tile_size,
    // +tile_size) when tiled, so a shift large enough to push the read past
    // a tile's own edge zero-pads there instead of silently pulling in a
    // neighboring tile's samples. abs_value reports |t| instead of the
    // signed t-statistic (each sample rectified after acc.compute()
    // succeeds). Returns false (with err set — "Cancelled." on user cancel,
    // empty on a declined memory warning) on failure.

    // Opens a new result tab with the full interactive t-test view (threshold
    // line, Calc TH, Style, Export PDF/PNG/NPY/TRS, trim controls) from an
    // already-computed result — the second half of showTTestDialog(), shared
    // with the Chain "Run T-test" step so a chain-driven run gets the same
    // rich result tab as running it from the menu. abs_value should match
    // what was passed to computeTTest(): it pre-checks "One-sided (+)" and
    // labels the tab/axis "|t|-value", since the negative threshold half is
    // meaningless once every value has been rectified to non-negative.
    void buildTTestResultTab(const std::shared_ptr<TTestAccumulator>& acc_ptr,
                              std::vector<float> tstat, int64_t n0, int64_t n1,
                              int32_t eff_count, bool abs_value);

    // Chain "Run T-test" step: resolves the step's trace range (the last
    // alignment's range if use_last_alignment and one exists, else the
    // step's own first_trace/trace_count), then calls computeTTest() +
    // buildTTestResultTab(). Returns false (with err set) on failure.
    bool runTTestChainStep(const ChainStep& step, QWidget* msg_parent, QString& err);

    // --- Background jobs -------------------------------------------------
    // Long analyses run on worker threads via jobs_ so one dataset's t-test
    // (or CPA/SNR/...) doesn't freeze every other tab. refreshJobsDock()
    // mirrors jobs_'s current state into the dock, which shows itself while
    // anything is running and hides again when the last job finishes.
    void refreshJobsDock();

    JobManager*  jobs_      = nullptr;
    QDockWidget* jobs_dock_ = nullptr;
    QVBoxLayout* jobs_rows_ = nullptr;
    struct JobRow { QWidget* row; QProgressBar* bar; QLabel* label; };
    std::map<int, JobRow> job_rows_;

    // Multi-dataset state
    std::vector<Dataset> datasets_;
    int                  active_idx_ = -1;
    bool                 tiled_      = false;

    bool     hasActiveDs() const { return active_idx_ >= 0; }
    Dataset& activeDs()          { return datasets_[static_cast<size_t>(active_idx_)]; }
    const Dataset& activeDs() const { return datasets_[static_cast<size_t>(active_idx_)]; }
    // The widget that should currently receive toolbar actions (zoom, mode,
    // reset, ...): the active tab's own widget, or a persistent empty
    // placeholder when no tab is open yet.
    PlotWidget* plotWidget() const {
        return hasActiveDs() ? datasets_[static_cast<size_t>(active_idx_)].plot_widget
                              : placeholder_widget_;
    }

    // Widgets
    PlotWidget*  placeholder_widget_ = nullptr;  // shown only while datasets_ is empty
    QWidget*     view_container_  = nullptr;
    QVBoxLayout* view_layout_     = nullptr;
    QPushButton* btn_tile_        = nullptr;
    QTabBar*     tab_bar_         = nullptr;

    // Side panel
    QLabel*      lbl_file_        = nullptr;
    QLabel*      lbl_info_        = nullptr;
    QLabel*      lbl_trace_data_  = nullptr;
    QSpinBox*    spin_data_idx_   = nullptr;
    QSpinBox*    spin_first_      = nullptr;
    QSpinBox*    spin_count_      = nullptr;
    QPushButton* btn_apply_       = nullptr;
    QLabel*      lbl_view_        = nullptr;
    QLabel*      lbl_measure_     = nullptr;
    QComboBox*   combo_transform_ = nullptr;
    QListWidget* list_transforms_ = nullptr;
    QPushButton* btn_add_tx_      = nullptr;
    QPushButton* btn_rm_tx_       = nullptr;
    QPushButton* btn_up_tx_       = nullptr;
    QPushButton* btn_dn_tx_       = nullptr;

    // Toolbar (above plot)
    QPushButton*  btn_mode_pan_      = nullptr;
    QPushButton*  btn_mode_measure_  = nullptr;
    QPushButton*  btn_mode_box_zoom_ = nullptr;
    QPushButton*  btn_mode_align_    = nullptr;
    QPushButton*  btn_stack_        = nullptr;
    QPushButton*  btn_zoom_in_       = nullptr;
    QPushButton*  btn_zoom_out_     = nullptr;
    QPushButton*  btn_reset_        = nullptr;
    QPushButton*  btn_undo_         = nullptr;
    QPushButton*  btn_unapply_      = nullptr;
    QComboBox*    combo_theme_      = nullptr;
    QButtonGroup* mode_group_       = nullptr;

    static const QColor TRACE_COLORS[];
    static constexpr int NUM_COLORS = 8;
};
