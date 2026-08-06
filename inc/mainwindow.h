     #pragma once

#include "trs_file.h"
#include "processing.h"
#include "plot_widget.h"
#include "ttest.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QPushButton>
#include <QSpinBox>
#include <QTabBar>

#include <deque>
#include <memory>
#include <vector>

class QVBoxLayout;

// ---------------------------------------------------------------------------
// Snapshot of mutable per-dataset state, used for undo.
// ---------------------------------------------------------------------------
struct DatasetSnapshot {
    std::vector<std::shared_ptr<ITransform>> pipeline; // each is clone()'d
    std::vector<int32_t> align_shifts;
    int32_t align_first_trace  = 0;
    int64_t align_first_sample = 0;
    int64_t align_n_samples    = 0;
    PlotViewState view;
};

// ---------------------------------------------------------------------------
// Per-file state bundle.  MainWindow keeps a vector of these.
// ---------------------------------------------------------------------------
struct Dataset {
    std::unique_ptr<TrsFile>                  file;
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

    // Plot state
    int32_t              plot_first_trace   = 0;
    bool                 plot_file_backed   = false;

    // Undo stack (most-recent first); capped at kMaxUndo
    std::deque<DatasetSnapshot> undo_stack;
    static constexpr int kMaxUndo = 50;
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    void openFile(const QString& path);

private slots:
    void onOpenFile();
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
    QComboBox*    combo_theme_      = nullptr;
    QButtonGroup* mode_group_       = nullptr;

    static const QColor TRACE_COLORS[];
    static constexpr int NUM_COLORS = 8;
};
