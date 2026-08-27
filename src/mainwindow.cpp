#include "mainwindow.h"
#include <zlib.h>
#include "heatmap_widget.h"
#include "ttest.h"
#include "align.h"
#include "xcorr.h"
#include "cpa.h"
#include "snr.h"
#include "leakage_model.h"
#include "leakage_model_dialog.h"
#include "chain.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QColorDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QSettings>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLineEdit>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPainter>
#include <QPdfWriter>
#include <QPageLayout>
#include <QPageSize>
#include <QProgressDialog>
#include <QPushButton>
#include <QShortcut>
#include <QRadioButton>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QHeaderView>
#include <QTableWidget>
#include <QVBoxLayout>

#include <unsupported/Eigen/FFT>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Statistical helpers for t-test threshold calculation
// ---------------------------------------------------------------------------
// Threshold formula follows:
//   Zhang, Ding, Durvaux, Standaert, Fei — "Towards Sound and Optimal Leakage
//   Detection Procedure", IACR ePrint 2017/287 (EuroS&P 2018).
//
// The correct per-sample significance level for an overall type I error rate α
// over n_L independent tests is the Šidák correction (Section 3.1):
//   α_TH = 1 − (1 − α)^(1/n_L)
// The t-test threshold is then TH = CDF_t^{−1}(1 − α_TH/2, ν_s), where ν_s
// are the Welch degrees of freedom.  For large ν_s this converges to the
// standard normal quantile.
// ---------------------------------------------------------------------------

// Rational approximation for Φ^{-1}(p) — Abramowitz & Stegun 26.2.17
// Max |error| ≤ 4.5×10^{-4}.
static double invNormCdf(double p) {
    if (p <= 0.0) return -1e300;
    if (p >= 1.0) return  1e300;
    bool upper = (p > 0.5);
    double q = upper ? 1.0 - p : p;
    double t = std::sqrt(-2.0 * std::log(q));
    double z = t - (2.515517 + t * (0.802853 + t * 0.010328))
                   / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308)));
    return upper ? z : -z;
}

// Inverse t-distribution CDF using Cornish-Fisher expansion (accurate for df > 5).
// Falls back to normal approximation for df >= 200.
static double invTCdf(double p, double df) {
    double z = invNormCdf(p);
    if (df >= 200.0) return z;
    double g = (z*z*z + z) / (4.0 * df);
    double h = (5.0*z*z*z*z*z + 16.0*z*z*z + 3.0*z) / (96.0 * df * df);
    return z + g + h;
}

const QColor MainWindow::TRACE_COLORS[] = {
    QColor("#4fc3f7"),  // light blue
    QColor("#ef5350"),  // red
    QColor("#66bb6a"),  // green
    QColor("#ffa726"),  // orange
    QColor("#ab47bc"),  // purple
    QColor("#26c6da"),  // cyan
    QColor("#d4e157"),  // lime
    QColor("#ff7043"),  // deep orange
};

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("TRS Viewer");
    resize(1440, 860);

    setupMenuBar();

    // ----------------------------------------------------------------
    // Central: side panel  |  (toolbar + plot)
    // ----------------------------------------------------------------
    QSplitter* splitter = new QSplitter(Qt::Horizontal, this);
    setCentralWidget(splitter);

    // ---- Side panel ----
    QWidget*     side   = new QWidget;
    QVBoxLayout* side_l = new QVBoxLayout(side);
    side_l->setSpacing(8);
    side->setMinimumWidth(230);
    side->setMaximumWidth(320);

    // File info
    QGroupBox*   grp_file = new QGroupBox("File");
    QVBoxLayout* fl       = new QVBoxLayout(grp_file);
    lbl_file_ = new QLabel("No file loaded");
    lbl_file_->setWordWrap(true);
    lbl_info_ = new QLabel;
    lbl_info_->setWordWrap(true);
    fl->addWidget(lbl_file_);
    fl->addWidget(lbl_info_);

    // Trace selector
    QGroupBox*   grp_trace = new QGroupBox("Traces");
    QFormLayout* tfl       = new QFormLayout(grp_trace);
    spin_first_ = new QSpinBox;
    spin_first_->setMinimum(0);
    spin_first_->setValue(0);
    spin_count_ = new QSpinBox;
    spin_count_->setMinimum(1);
    spin_count_->setMaximum(1000);
    spin_count_->setValue(1);
    btn_apply_ = new QPushButton("Load / Refresh");
    btn_apply_->setToolTip("(Re)load the live, file-backed view for the trace range above.\n"
                           "Any previously computed alignment is kept and re-shown visually, "
                           "but a baked \"Apply to Main View\" result reverts to live — same "
                           "effect as \"Un-apply Shifts\" on the toolbar. Also switches the "
                           "plot back to Pan mode if it was left in Align/Measure/Box Zoom.");
    connect(btn_apply_, &QPushButton::clicked, this, &MainWindow::onApplyTraces);
    connect(spin_first_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [this](int v){ spin_data_idx_->setValue(v); });
    tfl->addRow("First trace:", spin_first_);
    tfl->addRow("Count:", spin_count_);
    tfl->addRow(btn_apply_);

    // View info
    lbl_view_ = new QLabel;
    lbl_view_->setWordWrap(true);

    // Measurement readout
    QGroupBox*   grp_meas = new QGroupBox("Measurement");
    QVBoxLayout* ml       = new QVBoxLayout(grp_meas);
    lbl_measure_ = new QLabel("–");
    lbl_measure_->setWordWrap(true);
    lbl_measure_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    ml->addWidget(lbl_measure_);

    // Processing pipeline
    QGroupBox*   grp_tx = new QGroupBox("Processing Pipeline");
    QVBoxLayout* txl    = new QVBoxLayout(grp_tx);

    combo_transform_ = new QComboBox;
    combo_transform_->addItems({
        "Absolute Value",
        "Negate",
        "Moving Average",
        "Window Resample (avg)",
        "Stride Resample (pick every Nth)",
        "Offset  (add constant)",
        "Scale   (multiply by constant)",
        "FFT Magnitude",
        "STFT Magnitude",
        "Gaussian Noise",
        "Filter (Lowpass/Highpass/Bandpass/Notch)",
    });

    list_transforms_ = new QListWidget;
    list_transforms_->setSelectionMode(QAbstractItemView::SingleSelection);

    QHBoxLayout* tx_btns = new QHBoxLayout;
    btn_add_tx_ = new QPushButton("Add");
    btn_rm_tx_  = new QPushButton("Remove");
    btn_up_tx_  = new QPushButton("↑");
    btn_dn_tx_  = new QPushButton("↓");
    tx_btns->addWidget(btn_add_tx_);
    tx_btns->addWidget(btn_rm_tx_);
    tx_btns->addWidget(btn_up_tx_);
    tx_btns->addWidget(btn_dn_tx_);

    connect(btn_add_tx_, &QPushButton::clicked, this, &MainWindow::onAddTransform);
    connect(btn_rm_tx_,  &QPushButton::clicked, this, &MainWindow::onRemoveTransform);
    connect(btn_up_tx_,  &QPushButton::clicked, this, &MainWindow::onMoveTransformUp);
    connect(btn_dn_tx_,  &QPushButton::clicked, this, &MainWindow::onMoveTransformDown);

    txl->addWidget(combo_transform_);
    txl->addWidget(list_transforms_);
    txl->addLayout(tx_btns);

    // Trace data inspector
    QGroupBox*   grp_data = new QGroupBox("Trace Data");
    QVBoxLayout* dl       = new QVBoxLayout(grp_data);

    // Navigation row: ◀  [index spinbox]  ▶
    auto* data_nav   = new QWidget;
    auto* data_nav_l = new QHBoxLayout(data_nav);
    data_nav_l->setContentsMargins(0, 0, 0, 0);
    auto* btn_data_prev  = new QPushButton("◀");
    auto* btn_data_next  = new QPushButton("▶");
    spin_data_idx_ = new QSpinBox;
    spin_data_idx_->setMinimum(0);
    spin_data_idx_->setValue(0);
    spin_data_idx_->setKeyboardTracking(false);
    btn_data_prev->setFixedWidth(28);
    btn_data_next->setFixedWidth(28);
    data_nav_l->addWidget(btn_data_prev);
    data_nav_l->addWidget(spin_data_idx_, 1);
    data_nav_l->addWidget(btn_data_next);

    lbl_trace_data_ = new QLabel("–");
    lbl_trace_data_->setWordWrap(true);
    lbl_trace_data_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    lbl_trace_data_->setFont(QFont("Monospace", 8));

    dl->addWidget(data_nav);
    dl->addWidget(lbl_trace_data_);

    connect(spin_data_idx_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::updateTraceDataDisplay);
    connect(btn_data_prev, &QPushButton::clicked, this, [this](){
        spin_data_idx_->setValue(spin_data_idx_->value() - 1);
    });
    connect(btn_data_next, &QPushButton::clicked, this, [this](){
        spin_data_idx_->setValue(spin_data_idx_->value() + 1);
    });

    side_l->addWidget(grp_file);
    side_l->addWidget(grp_trace);
    side_l->addWidget(grp_data);
    side_l->addWidget(lbl_view_);
    side_l->addWidget(grp_meas);
    side_l->addWidget(grp_tx);
    side_l->addStretch();

    // ---- Right pane: toolbar + plot ----
    QWidget*     right_pane = new QWidget;
    QVBoxLayout* right_l    = new QVBoxLayout(right_pane);
    right_l->setContentsMargins(0, 0, 0, 0);
    right_l->setSpacing(2);

    // Toolbar row
    QWidget*     toolbar   = new QWidget;
    QHBoxLayout* toolbar_l = new QHBoxLayout(toolbar);
    toolbar_l->setContentsMargins(4, 2, 4, 2);
    toolbar_l->setSpacing(4);

    // Mode buttons (checkable, exclusive)
    btn_mode_pan_      = new QPushButton("Pan");
    btn_mode_measure_  = new QPushButton("Measure");
    btn_mode_box_zoom_ = new QPushButton("⬚ Box Zoom");
    btn_mode_align_    = new QPushButton("↔ Align");
    btn_mode_pan_->setCheckable(true);
    btn_mode_measure_->setCheckable(true);
    btn_mode_box_zoom_->setCheckable(true);
    btn_mode_align_->setCheckable(true);
    btn_mode_pan_->setChecked(true);
    btn_mode_pan_->setToolTip("Drag to pan, scroll wheel to zoom");
    btn_mode_measure_->setToolTip("Click two points to measure distance (P)");
    btn_mode_box_zoom_->setToolTip("Drag to select a region and zoom into it (Z)");
    btn_mode_align_->setToolTip("Click and drag a trace left/right to shift it");

    mode_group_ = new QButtonGroup(this);
    mode_group_->addButton(btn_mode_pan_,      0);
    mode_group_->addButton(btn_mode_measure_,  1);
    mode_group_->addButton(btn_mode_box_zoom_, 2);
    mode_group_->addButton(btn_mode_align_,    3);
    mode_group_->setExclusive(true);

    connect(mode_group_, &QButtonGroup::idClicked, this, [this](int id) {
        InteractionMode m = id == 0 ? InteractionMode::Pan
                          : id == 1 ? InteractionMode::Measure
                          : id == 2 ? InteractionMode::BoxZoom
                                    : InteractionMode::AlignDrag;
        plotWidget()->setMode(m);
        if (id == 0 || id == 2 || id == 3) lbl_measure_->setText("–");
    });

    // Stacked view: draw each trace in its own non-overlapping lane instead
    // of overlaying them all. Independent of the interaction mode above.
    btn_stack_ = new QPushButton("☰ Stack");
    btn_stack_->setCheckable(true);
    btn_stack_->setToolTip("Draw each trace in its own lane instead of overlapping it with the others");
    connect(btn_stack_, &QPushButton::toggled, this, [this](bool on) {
        plotWidget()->setStacked(on);
    });

    // Separator
    auto* sep1 = new QFrame; sep1->setFrameShape(QFrame::VLine);
    auto* sep2 = new QFrame; sep2->setFrameShape(QFrame::VLine);
    auto* sep3 = new QFrame; sep3->setFrameShape(QFrame::VLine);

    // Zoom buttons
    btn_zoom_in_  = new QPushButton("＋ Zoom In");
    btn_zoom_out_ = new QPushButton("－ Zoom Out");
    btn_reset_    = new QPushButton("⟳ Reset  [R]");
    btn_zoom_in_->setToolTip("Zoom in X (also: scroll wheel up)");
    btn_zoom_out_->setToolTip("Zoom out X (also: scroll wheel down)");
    btn_reset_->setToolTip("Pan/zoom back to the full range, clear crop ranges, and switch "
                           "back to Pan mode (out of Align/Measure/Box Zoom).\n"
                           "Does NOT touch traces, the pipeline, or alignment — a baked "
                           "\"Apply to Main View\" result and any computed shifts are left alone. "
                           "Use \"Load / Refresh\" or \"Un-apply Shifts\" for that.");

    auto* btn_yzoom_in  = new QPushButton("↑ Amp");
    auto* btn_yzoom_out = new QPushButton("↓ Amp");
    btn_yzoom_in ->setToolTip("Zoom in Y / taller traces (also: Ctrl/Shift+scroll up)");
    btn_yzoom_out->setToolTip("Zoom out Y / shorter traces (also: Ctrl/Shift+scroll down)");

    // plotWidget() resolves to whichever tab is active at click time (there's
    // no single fixed plot widget any more — each tab owns its own).
    connect(btn_zoom_in_,  &QPushButton::clicked, this, [this](){ plotWidget()->zoomIn(); });
    connect(btn_zoom_out_, &QPushButton::clicked, this, [this](){ plotWidget()->zoomOut(); });
    connect(btn_reset_,    &QPushButton::clicked, this, &MainWindow::onResetView);
    connect(btn_yzoom_in,  &QPushButton::clicked, this, [this](){ plotWidget()->zoomInY(); });
    connect(btn_yzoom_out, &QPushButton::clicked, this, [this](){ plotWidget()->zoomOutY(); });

    btn_undo_ = new QPushButton("↩ Undo");
    btn_undo_->setToolTip("Undo last action [Ctrl+Z]");
    btn_undo_->setEnabled(false);
    connect(btn_undo_, &QPushButton::clicked, this, &MainWindow::onUndoAction);

    // Reverts "Align Traces -> Apply to Main View"'s baked-in static traces
    // back to normal, live, file-backed ones (same rebuild Load/Refresh
    // already does) — the computed alignment itself (align_shifts etc.) is
    // kept, just no longer frozen into the plot, so pipeline edits and
    // t-test/CPA/xcorr reuse still work as if nothing changed.
    btn_unapply_ = new QPushButton("Un-apply Shifts");
    btn_unapply_->setToolTip("Restore the live, file-backed view after \"Apply to Main "
                             "View\" — keeps the computed alignment for later reuse, "
                             "just stops it being baked into the plot. Only enabled while "
                             "baked; identical to clicking \"Load / Refresh\" in the "
                             "Traces panel, just labeled for this specific situation.");
    btn_unapply_->setEnabled(false);
    connect(btn_unapply_, &QPushButton::clicked, this, &MainWindow::onApplyTraces);

    // Theme selector
    combo_theme_ = new QComboBox;
    combo_theme_->addItems({"Dark", "Light"});
    connect(combo_theme_, &QComboBox::currentIndexChanged,
            this, &MainWindow::onThemeChanged);

    toolbar_l->addWidget(btn_mode_pan_);
    toolbar_l->addWidget(btn_mode_measure_);
    toolbar_l->addWidget(btn_mode_box_zoom_);
    toolbar_l->addWidget(btn_mode_align_);
    toolbar_l->addWidget(btn_stack_);
    toolbar_l->addWidget(sep1);
    toolbar_l->addWidget(btn_zoom_in_);
    toolbar_l->addWidget(btn_zoom_out_);
    toolbar_l->addWidget(btn_reset_);
    toolbar_l->addWidget(btn_yzoom_in);
    toolbar_l->addWidget(btn_yzoom_out);
    toolbar_l->addWidget(sep2);
    toolbar_l->addWidget(btn_undo_);
    toolbar_l->addWidget(btn_unapply_);
    toolbar_l->addWidget(sep3);
    toolbar_l->addWidget(new QLabel("Theme:"));
    toolbar_l->addWidget(combo_theme_);
    toolbar_l->addStretch();

    // "Tile Vertically" — show every open tab's plot stacked at once instead
    // of one at a time. Independent of everything else on the toolbar.
    btn_tile_ = new QPushButton("⬒ Tile Tabs");
    btn_tile_->setCheckable(true);
    btn_tile_->setToolTip("Show every open tab stacked vertically, each with its own "
                         "axes, instead of one at a time");
    connect(btn_tile_, &QPushButton::toggled, this, &MainWindow::onToggleTile);
    toolbar_l->addWidget(btn_tile_);

    // Dataset tab bar
    tab_bar_ = new QTabBar;
    tab_bar_->setTabsClosable(true);
    tab_bar_->setMovable(true);
    tab_bar_->setExpanding(false);
    connect(tab_bar_, &QTabBar::currentChanged, this, &MainWindow::onSwitchDataset);
    connect(tab_bar_, &QTabBar::tabMoved, this, &MainWindow::onTabMoved);
    connect(tab_bar_, &QTabBar::tabCloseRequested, this, [this](int idx) {
        // Select a neighbour before removing so onSwitchDataset fires cleanly.
        if (tab_bar_->count() > 1) {
            int next = (idx > 0) ? idx - 1 : 1;
            tab_bar_->setCurrentIndex(next);
        }
        datasets_[static_cast<size_t>(idx)].plot_widget->deleteLater();
        if (datasets_[static_cast<size_t>(idx)].extra_toolbar)
            datasets_[static_cast<size_t>(idx)].extra_toolbar->deleteLater();
        datasets_.erase(datasets_.begin() + idx);
        tab_bar_->removeTab(idx);
        // Recompute active_idx_ after removal.
        active_idx_ = tab_bar_->currentIndex();
        if (active_idx_ < 0) {
            // No datasets left
            active_idx_ = -1;
            updateFileInfo();
            rebuildTransformList();
        }
        updateViewLayout();
    });

    // View container: holds every tab's own PlotWidget. Single-view mode
    // shows only the active one; tiled mode stacks all of them.
    view_container_ = new QWidget;
    view_layout_    = new QVBoxLayout(view_container_);
    view_layout_->setContentsMargins(0, 0, 0, 0);
    view_layout_->setSpacing(4);

    placeholder_widget_ = new PlotWidget(view_container_);
    view_layout_->addWidget(placeholder_widget_, 1);

    right_l->addWidget(tab_bar_);
    right_l->addWidget(toolbar);
    right_l->addWidget(view_container_, 1);

    splitter->addWidget(side);
    splitter->addWidget(right_pane);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);

    // Keyboard shortcuts
    auto* sc_undo = new QShortcut(QKeySequence::Undo, this);  // Ctrl+Z
    connect(sc_undo, &QShortcut::activated, this, &MainWindow::onUndoAction);
    auto* sc_reset = new QShortcut(QKeySequence("R"), this);
    connect(sc_reset, &QShortcut::activated, this, &MainWindow::onResetView);
    auto* sc_plus  = new QShortcut(QKeySequence("+"), this);
    connect(sc_plus,  &QShortcut::activated, this, [this]() { plotWidget()->zoomIn(); });
    auto* sc_minus = new QShortcut(QKeySequence("-"), this);
    connect(sc_minus, &QShortcut::activated, this, [this]() { plotWidget()->zoomOut(); });
    auto* sc_p = new QShortcut(QKeySequence("P"), this);
    connect(sc_p, &QShortcut::activated, this, [this]() {
        bool measure = (plotWidget()->mode() == InteractionMode::Pan);
        plotWidget()->setMode(measure ? InteractionMode::Measure
                                       : InteractionMode::Pan);
        btn_mode_pan_->setChecked(!measure);
        btn_mode_measure_->setChecked(measure);
        btn_mode_box_zoom_->setChecked(false);
        if (!measure) lbl_measure_->setText("–");
    });
    auto* sc_z = new QShortcut(QKeySequence("Z"), this);
    connect(sc_z, &QShortcut::activated, this, [this]() {
        bool box_zoom = (plotWidget()->mode() == InteractionMode::BoxZoom);
        plotWidget()->setMode(box_zoom ? InteractionMode::Pan
                                       : InteractionMode::BoxZoom);
        btn_mode_pan_->setChecked(box_zoom);
        btn_mode_measure_->setChecked(false);
        btn_mode_box_zoom_->setChecked(!box_zoom);
        if (box_zoom) lbl_measure_->setText("–");
    });
}

MainWindow::~MainWindow() = default;

// ---------------------------------------------------------------------------
// Dataset switching
// ---------------------------------------------------------------------------

void MainWindow::onSwitchDataset(int idx) {
    if (idx < 0 || idx >= static_cast<int>(datasets_.size())) return;
    if (idx == active_idx_) return;
    active_idx_ = idx;

    // Each tab owns its own persistent PlotWidget with its own traces/view
    // already loaded — switching just changes which one is active and
    // refreshes the side panel; it never reloads or resets the plot.
    rebuildTransformList();

    int n = activeDs().file ? activeDs().file->header().num_traces : 0;
    spin_first_->setMaximum(std::max(0, n - 1));
    spin_count_->setMaximum(std::max(1, n));

    updateFileInfo();
    updateUndoButton();
    updateViewLayout();
}

void MainWindow::onCloseDataset() {
    if (!hasActiveDs()) return;
    int idx = active_idx_;
    if (tab_bar_->count() > 1) {
        int next = (idx > 0) ? idx - 1 : 1;
        tab_bar_->setCurrentIndex(next);
    }
    datasets_[static_cast<size_t>(idx)].plot_widget->deleteLater();
    if (datasets_[static_cast<size_t>(idx)].extra_toolbar)
        datasets_[static_cast<size_t>(idx)].extra_toolbar->deleteLater();
    datasets_.erase(datasets_.begin() + idx);
    tab_bar_->removeTab(idx);
    active_idx_ = tab_bar_->currentIndex();
    if (active_idx_ < 0 || datasets_.empty()) {
        active_idx_ = -1;
        updateFileInfo();
        rebuildTransformList();
    }
    updateViewLayout();
}

// ---------------------------------------------------------------------------
// Tab / tile view management
// ---------------------------------------------------------------------------

PlotWidget* MainWindow::createPlotWidgetForTab() {
    auto* pw = new PlotWidget(view_container_);

    // Interaction signals from any tab's widget can fire while that tab is
    // not the active one (tiled mode shows every widget at once) — route
    // through activateDatasetForWidget() first so activeDs()/plotWidget()
    // resolve to the dataset that actually owns the widget that fired.
    connect(pw, &PlotWidget::viewChanged, this,
            [this, pw](int64_t s, int64_t e, int64_t t) {
                activateDatasetForWidget(pw);
                onViewChanged(s, e, t);
            });
    connect(pw, &PlotWidget::measurementUpdated, this,
            [this, pw](int64_t s1, double v1, int64_t s2, double v2, bool has_p2) {
                activateDatasetForWidget(pw);
                onMeasurementUpdated(s1, v1, s2, v2, has_p2);
            });
    connect(pw, &PlotWidget::traceShiftsChanged, this,
            [this, pw]() {
                activateDatasetForWidget(pw);
                onDragAlignChanged();
            });
    connect(pw, &PlotWidget::alignDragStarted, this,
            [this, pw]() {
                activateDatasetForWidget(pw);
                saveSnapshot();
            });
    connect(pw, &PlotWidget::beforeViewChange, this,
            [this, pw]() {
                activateDatasetForWidget(pw);
                saveSnapshot();
            });
    return pw;
}

void MainWindow::updateViewLayout() {
    // Pull everything currently in the layout back out (the widgets
    // themselves stay alive — they're owned by view_container_/datasets_ —
    // this just clears the arrangement so it can be rebuilt from scratch).
    while (view_layout_->count() > 0) {
        QLayoutItem* item = view_layout_->takeAt(0);
        if (item->widget()) item->widget()->setVisible(false);
        delete item;
    }

    if (datasets_.empty()) {
        placeholder_widget_->setVisible(true);
        view_layout_->addWidget(placeholder_widget_, 1);
        return;
    }
    placeholder_widget_->setVisible(false);

    auto addDatasetWidgets = [this](Dataset& ds) {
        if (ds.extra_toolbar) {
            view_layout_->addWidget(ds.extra_toolbar, 0);
            ds.extra_toolbar->setVisible(true);
        }
        view_layout_->addWidget(ds.plot_widget, 1);
        ds.plot_widget->setVisible(true);
    };

    if (tiled_) {
        for (auto& ds : datasets_) addDatasetWidgets(ds);
    } else if (hasActiveDs()) {
        addDatasetWidgets(activeDs());
    }
}

int MainWindow::datasetIndexForWidget(PlotWidget* pw) const {
    for (size_t i = 0; i < datasets_.size(); i++)
        if (datasets_[i].plot_widget == pw) return static_cast<int>(i);
    return -1;
}

void MainWindow::activateDatasetForWidget(PlotWidget* pw) {
    int idx = datasetIndexForWidget(pw);
    if (idx < 0 || idx == active_idx_) return;
    active_idx_ = idx;

    // Side panel only — never touch the plot contents or the view layout
    // here, since this can be called for a tiled, non-selected panel that
    // must keep rendering exactly as it already is.
    tab_bar_->blockSignals(true);
    tab_bar_->setCurrentIndex(idx);
    tab_bar_->blockSignals(false);

    rebuildTransformList();
    int n = activeDs().file ? activeDs().file->header().num_traces : 0;
    spin_first_->setMaximum(std::max(0, n - 1));
    spin_count_->setMaximum(std::max(1, n));
    updateFileInfo();
    updateUndoButton();
}

void MainWindow::onToggleTile(bool on) {
    tiled_ = on;
    updateViewLayout();
}

void MainWindow::addResultTab(const std::vector<float>& result, const QString& title,
                               const QColor& color, const QString& trace_label)
{
    Dataset ds;
    ds.is_result    = true;
    ds.display_name = title;
    ds.plot_widget  = createPlotWidgetForTab();
    datasets_.push_back(std::move(ds));
    active_idx_ = static_cast<int>(datasets_.size()) - 1;

    tab_bar_->addTab(datasets_[static_cast<size_t>(active_idx_)].display_name);
    tab_bar_->setCurrentIndex(active_idx_);

    auto data = std::make_shared<std::vector<float>>(result);
    plotWidget()->addTrace(data, color, trace_label);
    plotWidget()->resetView();

    rebuildTransformList();
    updateFileInfo();
    updateViewLayout();
}

void MainWindow::onTabMoved(int from, int to) {
    // Keep datasets_ index-aligned with the tab bar's now-reordered tabs —
    // everything else (activeDs(), updateViewLayout()'s tiled stacking
    // order) indexes into datasets_ assuming that alignment holds.
    if (from < 0 || from >= static_cast<int>(datasets_.size()) ||
        to   < 0 || to   >= static_cast<int>(datasets_.size()))
        return;
    Dataset moved = std::move(datasets_[static_cast<size_t>(from)]);
    datasets_.erase(datasets_.begin() + from);
    datasets_.insert(datasets_.begin() + to, std::move(moved));
    active_idx_ = tab_bar_->currentIndex();
    updateViewLayout();
}

// ---------------------------------------------------------------------------

void MainWindow::setupMenuBar() {
    QMenu* file_menu = menuBar()->addMenu("&File");

    auto* act_open = new QAction("&Open TRS file…", this);
    act_open->setShortcut(QKeySequence::Open);
    connect(act_open, &QAction::triggered, this, &MainWindow::onOpenFile);
    file_menu->addAction(act_open);

    auto* act_open_npy = new QAction("Open NPY/NPZ as &traces…", this);
    connect(act_open_npy, &QAction::triggered, this, &MainWindow::onOpenNpyTraces);
    file_menu->addAction(act_open_npy);

    auto* act_close = new QAction("&Close Dataset", this);
    act_close->setShortcut(QKeySequence("Ctrl+W"));
    connect(act_close, &QAction::triggered, this, &MainWindow::onCloseDataset);
    file_menu->addAction(act_close);

    file_menu->addSeparator();

    auto* act_quit = new QAction("&Quit", this);
    act_quit->setShortcut(QKeySequence::Quit);
    connect(act_quit, &QAction::triggered, this, &QWidget::close);
    file_menu->addAction(act_quit);

    QMenu* export_menu = menuBar()->addMenu("&Export");

    auto* act_exp_trs = new QAction("Export &TRS (processed traces)…", this);
    connect(act_exp_trs, &QAction::triggered, this, &MainWindow::onExportTrs);
    export_menu->addAction(act_exp_trs);

    auto* act_exp_npy_traces = new QAction("Export traces as &NPY (2-D matrix)…", this);
    connect(act_exp_npy_traces, &QAction::triggered, this, &MainWindow::onExportNpy);
    export_menu->addAction(act_exp_npy_traces);

    auto* act_exp_npz = new QAction("Export traces as NP&Z (traces + data)…", this);
    connect(act_exp_npz, &QAction::triggered, this, &MainWindow::onExportNpz);
    export_menu->addAction(act_exp_npz);

    auto* act_exp_dataset = new QAction("Export &Dataset (NPZ)…", this);
    connect(act_exp_dataset, &QAction::triggered, this, &MainWindow::onExportDataset);
    export_menu->addAction(act_exp_dataset);

    export_menu->addSeparator();

    auto* act_exp_png = new QAction("Export plot as &PNG…", this);
    act_exp_png->setShortcut(QKeySequence("Ctrl+Shift+S"));
    connect(act_exp_png, &QAction::triggered, this, &MainWindow::onExportPng);
    export_menu->addAction(act_exp_png);

    auto* act_exp_pdf = new QAction("Export plot as P&DF…", this);
    connect(act_exp_pdf, &QAction::triggered, this, &MainWindow::onExportPdf);
    export_menu->addAction(act_exp_pdf);

    QMenu* sca_menu = menuBar()->addMenu("&SCA");
    auto* act_ttest = new QAction("Run &Welch t-test…", this);
    connect(act_ttest, &QAction::triggered, this, &MainWindow::onRunTTest);
    sca_menu->addAction(act_ttest);

    auto* act_xcorr = new QAction("&Cross-Correlation…", this);
    connect(act_xcorr, &QAction::triggered, this, &MainWindow::onRunXCorr);
    sca_menu->addAction(act_xcorr);

    auto* act_dpa = new QAction("&CPA…", this);
    connect(act_dpa, &QAction::triggered, this, &MainWindow::onRunCpa);
    sca_menu->addAction(act_dpa);

    auto* act_snr = new QAction("&SNR…", this);
    connect(act_snr, &QAction::triggered, this, &MainWindow::onRunSNR);
    sca_menu->addAction(act_snr);

    auto* act_static_snr = new QAction("Static SNR |μ/σ|…", this);
    connect(act_static_snr, &QAction::triggered, this, &MainWindow::onRunStaticSNR);
    sca_menu->addAction(act_static_snr);

    auto* act_fft = new QAction("&FFT Spectrum…", this);
    connect(act_fft, &QAction::triggered, this, &MainWindow::onRunFFT);
    sca_menu->addAction(act_fft);

    auto* act_align = new QAction("&Align Traces…", this);
    connect(act_align, &QAction::triggered, this, &MainWindow::onAlignTraces);
    sca_menu->addAction(act_align);

    sca_menu->addSeparator();

    auto* act_load_npy_ttest = new QAction("Load t-test &NPY…", this);
    connect(act_load_npy_ttest, &QAction::triggered, this, &MainWindow::onLoadNpyTTest);
    sca_menu->addAction(act_load_npy_ttest);

    auto* act_load_npy_heatmap = new QAction("Load heatmap &NPY…", this);
    connect(act_load_npy_heatmap, &QAction::triggered, this, &MainWindow::onLoadNpyHeatmap);
    sca_menu->addAction(act_load_npy_heatmap);

    QMenu* crop_menu = menuBar()->addMenu("C&rop");
    auto* act_crop = new QAction("&Range Editor…", this);
    connect(act_crop, &QAction::triggered, this, &MainWindow::onCropEdit);
    crop_menu->addAction(act_crop);

    QMenu* chain_menu = menuBar()->addMenu("&Chain");
    auto* act_chain = new QAction("Chain &Editor…", this);
    act_chain->setToolTip("Build, save/load, and run a saved sequence of steps (add "
                          "transform, align, clear pipeline, reload, export) instead "
                          "of redoing the same clicks by hand every time.");
    connect(act_chain, &QAction::triggered, this, &MainWindow::onChainEditor);
    chain_menu->addAction(act_chain);
}

// ---------------------------------------------------------------------------
// Undo helpers
// ---------------------------------------------------------------------------
void MainWindow::saveSnapshot() {
    if (!hasActiveDs()) return;
    auto& ds = activeDs();
    DatasetSnapshot snap;
    for (const auto& t : ds.pipeline)
        snap.pipeline.push_back(t->clone());
    snap.align_shifts       = ds.align_shifts;
    snap.align_first_trace  = ds.align_first_trace;
    snap.align_first_sample = ds.align_first_sample;
    snap.align_n_samples    = ds.align_n_samples;
    snap.view               = plotWidget()->captureViewState();
    ds.undo_stack.push_front(std::move(snap));
    if (static_cast<int>(ds.undo_stack.size()) > Dataset::kMaxUndo)
        ds.undo_stack.pop_back();
    updateUndoButton();
}

void MainWindow::restoreSnapshot(DatasetSnapshot snap) {
    auto& ds = activeDs();
    ds.pipeline         = std::move(snap.pipeline);
    ds.align_shifts     = std::move(snap.align_shifts);
    ds.align_first_trace  = snap.align_first_trace;
    ds.align_first_sample = snap.align_first_sample;
    ds.align_n_samples    = snap.align_n_samples;

    rebuildTransformList();
    plotWidget()->setTransforms(ds.pipeline);
    plotWidget()->restoreViewState(snap.view);

    // Re-apply shifts to currently visible traces. Signals blocked for the
    // same reason as the identical loop in onApplyTraces(): each
    // setTraceShift() call below would otherwise emit traceShiftsChanged(),
    // which feeds straight back into ds.align_shifts via
    // onDragAlignChanged() with only a partial (in-progress) shift list —
    // corrupting the very array this loop is reading from, mid-loop. This
    // is a pure re-display of already-known shifts, not a new user drag.
    {
        QSignalBlocker blocker(plotWidget());
        plotWidget()->clearTraceShifts();
        if (!ds.align_shifts.empty() && ds.plot_file_backed) {
            int n = static_cast<int>(plotWidget()->traceShifts().size());
            for (int i = 0; i < n; i++) {
                int shift_idx = ds.plot_first_trace + i - ds.align_first_trace;
                if (shift_idx >= 0 && shift_idx < static_cast<int>(ds.align_shifts.size())) {
                    int32_t shift = ds.align_shifts[static_cast<size_t>(shift_idx)];
                    if (shift != kAlignDiscardShift)
                        plotWidget()->setTraceShift(i, shift);
                }
            }
        }
    }

    updateFileInfo();
    updateUndoButton();
}

void MainWindow::onUndoAction() {
    if (!hasActiveDs() || activeDs().undo_stack.empty()) return;
    auto snap = std::move(activeDs().undo_stack.front());
    activeDs().undo_stack.pop_front();
    restoreSnapshot(std::move(snap));
}

void MainWindow::updateUndoButton() {
    if (btn_undo_)
        btn_undo_->setEnabled(hasActiveDs() && !activeDs().undo_stack.empty());
    // Piggybacks on the same call sites as the undo button — every one of
    // them is a point where plot_file_backed could plausibly have changed
    // (tab switch, undo/redo, Apply to Main View, Load/Refresh itself).
    if (btn_unapply_)
        btn_unapply_->setEnabled(hasActiveDs() && !activeDs().is_result
                                  && !activeDs().plot_file_backed);
}

void MainWindow::resetInteractionModeToPan() {
    // QButtonGroup::idClicked only fires for an actual user click, not for a
    // programmatic setChecked(), so the two need to be driven separately:
    // setMode() is what the plot actually obeys, setChecked() is just
    // keeping the toolbar's exclusive Pan/Measure/Box Zoom/Align buttons in
    // sync with it (setChecked(true) here also unchecks whichever of the
    // other three was active, since they share one exclusive QButtonGroup).
    plotWidget()->setMode(InteractionMode::Pan);
    if (btn_mode_pan_) btn_mode_pan_->setChecked(true);
}

// ---------------------------------------------------------------------------
// Recent-directory helpers (persisted via QSettings)
// ---------------------------------------------------------------------------
QString MainWindow::recentDir(const QString& key) {
    return QSettings("trs-viewer", "trs-viewer").value("recentDir/" + key).toString();
}

void MainWindow::updateRecentDir(const QString& key, const QString& file_path) {
    QSettings("trs-viewer", "trs-viewer")
        .setValue("recentDir/" + key, QFileInfo(file_path).absolutePath());
}

void MainWindow::onOpenFile() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open TRS file", recentDir("trs"), "TRS files (*.trs);;All files (*)");
    if (!path.isEmpty()) { updateRecentDir("trs", path); openFile(path); }
}

void MainWindow::prefetchWithProgress(TrsFile* file, const QString& label) {
    QProgressDialog prog(label, "Skip", 0, 100, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);   // small/already-cached files never show it
    prog.setValue(0);

    file->prefetch([&](int64_t done, int64_t total) {
        prog.setValue(total > 0 ? static_cast<int>(done * 100 / total) : 100);
        QApplication::processEvents();
        return !prog.wasCanceled();
    });
    prog.setValue(100);
}

void MainWindow::openFile(const QString& path) {
    auto f = std::make_unique<TrsFile>();
    std::string err;
    if (!f->open(path.toStdString(), err)) {
        QMessageBox::critical(this, "Error opening TRS file",
                              QString::fromStdString(err));
        return;
    }
    prefetchWithProgress(f.get(), "Reading " + QFileInfo(path).fileName() + "…");

    Dataset ds;
    ds.file         = std::move(f);
    ds.display_name = QFileInfo(path).fileName();
    ds.plot_widget  = createPlotWidgetForTab();
    datasets_.push_back(std::move(ds));
    active_idx_ = static_cast<int>(datasets_.size()) - 1;

    tab_bar_->addTab(datasets_[static_cast<size_t>(active_idx_)].display_name);
    tab_bar_->setCurrentIndex(active_idx_);

    rebuildTransformList();
    plotWidget()->setTransforms(activeDs().pipeline);

    int n = activeDs().file->header().num_traces;
    spin_first_->setMaximum(std::max(0, n - 1));
    spin_first_->setValue(0);
    spin_count_->setMaximum(n);
    spin_count_->setValue(1);

    updateFileInfo();
    onApplyTraces();
    updateViewLayout();
}

static QString hexBytes(const uint8_t* p, size_t n, int group = 0) {
    QString s;
    for (size_t i = 0; i < n; i++) {
        if (group > 0 && i > 0 && i % static_cast<size_t>(group) == 0) s += ' ';
        s += QString("%1").arg(p[i], 2, 16, QChar('0'));
    }
    return s;
}

void MainWindow::updateFileInfo() {
    if (!hasActiveDs()) { lbl_file_->setText("No file"); lbl_info_->clear();
                      lbl_trace_data_->setText("–"); return; }
    if (activeDs().is_result) {
        lbl_file_->setText(activeDs().display_name);
        lbl_info_->setText("Derived result (no source file)");
        lbl_trace_data_->setText("–");
        return;
    }

    const auto& h = activeDs().file->header();
    lbl_file_->setText(QString::fromStdString(activeDs().file->path()).section('/', -1));

    const char* type_str = "?";
    switch (h.sample_type) {
    case SampleType::INT8:    type_str = "int8";    break;
    case SampleType::INT16:   type_str = "int16";   break;
    case SampleType::INT32:   type_str = "int32";   break;
    case SampleType::FLOAT32: type_str = "float32"; break;
    }

    int64_t effective_samples = h.num_samples;
    for (const auto& t : activeDs().pipeline)
        effective_samples = t->transformedCount(effective_samples);

    QString info = QString("Traces:  %1\nSamples: %2\nType:    %3\nData:    %4 B/trace")
        .arg(h.num_traces).arg(h.num_samples)
        .arg(type_str).arg(h.data_length);
    if (effective_samples != h.num_samples)
        info += QString("\nAfter pipeline: %1").arg(effective_samples);
    lbl_info_->setText(info);

    // Sync data navigator range
    spin_data_idx_->setMaximum(std::max(0, h.num_traces - 1));

    updateTraceDataDisplay();
}

void MainWindow::updateTraceDataDisplay() {
    if (!hasActiveDs() || activeDs().is_result) { lbl_trace_data_->setText("–"); return; }
    const auto& h = activeDs().file->header();
    if (h.data_length <= 0) { lbl_trace_data_->setText("(no data)"); return; }

    int ti = spin_data_idx_->value();
    if (ti >= h.num_traces) { lbl_trace_data_->setText("(out of range)"); return; }

    auto raw = activeDs().file->readData(ti);
    if (raw.empty()) { lbl_trace_data_->setText("(empty)"); return; }

    QString text = QString("Trace %1 / %2\n").arg(ti).arg(h.num_traces - 1);

    // If we have named params, show them by name with decoded integer values
    if (!h.param_map.empty() && h.param_map.find("LEGACY_DATA") == h.param_map.end()) {
        for (const auto& kv : h.param_map) {
            const TrsTraceParam& p = kv.second;
            if (p.offset + p.length > static_cast<int>(raw.size())) continue;
            const uint8_t* b = raw.data() + p.offset;
            int64_t val = 0;
            // Always interpret as little-endian signed integer for display
            for (int i = p.length - 1; i >= 0; i--)
                val = (val << 8) | b[i];
            // sign-extend if needed (type 2=i16, 4=i32, 8=i64)
            if (p.type != 1 && p.length <= 8) {
                int bits = p.length * 8;
                int64_t sign_bit = int64_t(1) << (bits - 1);
                if (val & sign_bit) val |= ~((sign_bit << 1) - 1);
            }
            text += QString("%1: %2\n").arg(QString::fromStdString(kv.first), -16).arg(val);
        }
        text = text.trimmed();
    } else if (h.param_map.find("LEGACY_DATA") != h.param_map.end()) {
        auto it = h.param_map.find("LEGACY_DATA");
        if (it->second.length == 32 && it->second.offset == 0 && raw.size() >= 32) {
            text += "PT: " + hexBytes(raw.data(),      16, 4) + "\n";
            text += "CT: " + hexBytes(raw.data() + 16, 16, 4);
        } else {
            for (size_t off = 0; off < raw.size(); off += 16) {
                size_t n = std::min<size_t>(16, raw.size() - off);
                text += hexBytes(raw.data() + off, n, 4) + "\n";
            }
            text = text.trimmed();
        }
    } else {
        for (size_t off = 0; off < raw.size(); off += 16) {
            size_t n = std::min<size_t>(16, raw.size() - off);
            text += hexBytes(raw.data() + off, n, 4) + "\n";
        }
        text = text.trimmed();
    }
    lbl_trace_data_->setText(text);
}

void MainWindow::onApplyTraces() {
    if (!hasActiveDs() || activeDs().is_result) return;

    // This button (re)populates the trace list — it's not the Reset button,
    // so the current pan/zoom should survive it. clearTraces() unconditionally
    // wipes the view, so capture it first and restore afterward (clamped to
    // whatever the reloaded traces actually span). A dataset with no prior
    // view (first load) falls back to a full reset instead.
    const bool    had_view    = plotWidget()->totalSamples() > 0;
    const int64_t saved_start = plotWidget()->viewStart();
    const int64_t saved_end   = plotWidget()->viewEnd();
    const float   saved_yscale = plotWidget()->yScale();

    plotWidget()->clearTraces();

    int first = spin_first_->value();
    int count = spin_count_->value();
    int max   = activeDs().file->header().num_traces;

    for (int i = 0; i < count && (first + i) < max; i++) {
        QColor  col   = TRACE_COLORS[(first + i) % NUM_COLORS];
        QString label = QString("Trace %1").arg(first + i);
        plotWidget()->addTrace(activeDs().file.get(), first + i, col, label);
    }
    plotWidget()->setTransforms(activeDs().pipeline);

    if (had_view) {
        plotWidget()->setViewRange(saved_start, saved_end);
        plotWidget()->setYScale(saved_yscale);
    } else {
        plotWidget()->resetView();
    }

    // Alignment state (align_shifts + its own first_trace/first_sample/
    // n_samples) is kept on purpose: it's addressed by absolute trace index,
    // not by whatever range happens to be loaded into the preview right now,
    // so re-populating the list with a different first/count (e.g. widening
    // the count to cover all traces before running a t-test) does not make
    // it stale. It only becomes stale when the underlying file itself
    // changes, and every load path (openFile/onOpenNpyTraces) already starts
    // from a freshly-constructed Dataset with empty alignment fields — so
    // there is nothing to clear here.

    // Mark plot as file-backed so drag-align updates alignment state.
    activeDs().plot_first_trace  = first;
    activeDs().plot_file_backed  = true;

    // Re-apply previously computed shifts to the freshly (re)loaded traces
    // so the preview still shows them aligned, same as after undo/redo.
    // setTraceShift()/clearTraceShifts() emit traceShiftsChanged() for every
    // trace whose shift actually changes, which is wired (via
    // onDragAlignChanged()) straight back into activeDs().align_shifts —
    // that's correct for a real user drag, but here we're re-displaying
    // shifts already recorded in align_shifts, one setTraceShift() call at a
    // time. Left unguarded, each call in the loop below fed the *partial*
    // traceShifts() snapshot (only traces processed so far) straight back
    // into align_shifts, so every iteration after the first read a
    // freshly-truncated array back out — collapsing align_shifts down to
    // just the first trace's shift, with every trace after it silently
    // un-shifted. Block signals for the whole re-apply so it's pure display.
    {
        QSignalBlocker blocker(plotWidget());
        plotWidget()->clearTraceShifts();
        if (!activeDs().align_shifts.empty()) {
            int n = static_cast<int>(plotWidget()->traceShifts().size());
            for (int i = 0; i < n; i++) {
                int shift_idx = first + i - activeDs().align_first_trace;
                if (shift_idx >= 0 && shift_idx < static_cast<int>(activeDs().align_shifts.size())) {
                    int32_t shift = activeDs().align_shifts[static_cast<size_t>(shift_idx)];
                    if (shift != kAlignDiscardShift)
                        plotWidget()->setTraceShift(i, shift);
                }
            }
        }
    }
    resetInteractionModeToPan();
    updateUndoButton();   // also refreshes btn_unapply_'s enabled state
}

void MainWindow::onDragAlignChanged() {
    // Called whenever a trace is drag-shifted in the main plot.
    // Only update alignment state when the plot holds file-backed traces.
    if (!activeDs().file || !activeDs().plot_file_backed) return;

    auto shifts = plotWidget()->traceShifts();
    if (shifts.empty()) return;

    activeDs().align_first_trace  = activeDs().plot_first_trace;
    activeDs().align_shifts       = std::move(shifts);
    activeDs().align_first_sample = 0;
    activeDs().align_n_samples    = activeDs().file->header().num_samples;
}

void MainWindow::rebakeAlignedView() {
    Dataset& ds = activeDs();
    if (ds.plot_file_backed || ds.align_n_samples <= 0 || !ds.file) return;

    const auto& shifts = ds.align_shifts;
    int32_t first_tr    = ds.align_first_trace;
    int64_t out_start    = ds.align_first_sample;
    int64_t out_len       = ds.align_n_samples;
    const TrsHeader& h  = ds.file->header();

    int64_t effective_len = out_len;
    for (const auto& t : ds.pipeline) effective_len = t->transformedCount(effective_len);

    // Preserve the current pan/zoom across the rebuild — clearTraces()
    // zeroes the view and the first addTrace() below would otherwise reset
    // it to full range, which would be a jarring surprise on every pipeline
    // tweak (as opposed to the live, non-baked path's setTransforms(), which
    // never touches the view at all).
    PlotViewState saved_view = plotWidget()->captureViewState();
    plotWidget()->clearTraces();
    int shown = 0;
    for (int i = 0; i < static_cast<int>(shifts.size()) && shown < NUM_COLORS; i++) {
        int64_t shift = static_cast<int64_t>(shifts[static_cast<size_t>(i)]);
        if (shift == kAlignDiscardShift) continue;

        auto data = std::make_shared<std::vector<float>>(
            static_cast<size_t>(std::max(out_len, effective_len)), 0.0f);

        int64_t raw_start = out_start + shift;
        int64_t raw_end   = raw_start + out_len;
        int64_t src_start = std::max<int64_t>(0, raw_start);
        int64_t src_end   = std::min<int64_t>(h.num_samples, raw_end);
        int64_t dst_off   = src_start - raw_start;

        if (src_start < src_end)
            ds.file->readSamples(first_tr + i, src_start, src_end - src_start,
                                  data->data() + static_cast<size_t>(dst_off));

        for (const auto& t : ds.pipeline) t->reset();
        int64_t n_out = out_len;
        for (const auto& t : ds.pipeline) n_out = t->apply(data->data(), n_out, 0);
        data->resize(static_cast<size_t>(std::max<int64_t>(0, n_out)));

        plotWidget()->addTrace(std::move(data), TRACE_COLORS[shown % NUM_COLORS],
            QString("T%1 (%2%3)").arg(first_tr + i).arg(shift >= 0 ? "+" : "").arg(shift));
        shown++;
    }
    plotWidget()->restoreViewState(saved_view);
}

void MainWindow::onAddTransform() {
    auto tx = createTransform(combo_transform_->currentIndex());
    if (!tx) return;
    saveSnapshot();
    activeDs().pipeline.push_back(tx);
    rebuildTransformList();
    updateFileInfo();
    // Only push the live pipeline to the plot when it's showing raw,
    // file-backed traces. When the main view holds a static baked-in
    // result (e.g. after Align Traces → Apply to Main View, which already
    // applied the pipeline once to produce it), re-syncing here would
    // silently re-apply the pipeline on top of that already-processed
    // result the next time it repaints.
    if (activeDs().plot_file_backed)
        plotWidget()->setTransforms(activeDs().pipeline);
    else
        rebakeAlignedView();
    plotWidget()->update();
}

void MainWindow::onRemoveTransform() {
    int row = list_transforms_->currentRow();
    if (row < 0 || row >= static_cast<int>(activeDs().pipeline.size())) return;
    saveSnapshot();
    activeDs().pipeline.erase(activeDs().pipeline.begin() + row);
    rebuildTransformList();
    updateFileInfo();
    if (activeDs().plot_file_backed)
        plotWidget()->setTransforms(activeDs().pipeline);
    else
        rebakeAlignedView();
    plotWidget()->update();
}

void MainWindow::onMoveTransformUp() {
    int row = list_transforms_->currentRow();
    if (row <= 0 || row >= static_cast<int>(activeDs().pipeline.size())) return;
    saveSnapshot();
    std::swap(activeDs().pipeline[row], activeDs().pipeline[row - 1]);
    rebuildTransformList();
    list_transforms_->setCurrentRow(row - 1);
    updateFileInfo();
    if (activeDs().plot_file_backed)
        plotWidget()->setTransforms(activeDs().pipeline);
    else
        rebakeAlignedView();
    plotWidget()->update();
}

void MainWindow::onMoveTransformDown() {
    int row = list_transforms_->currentRow();
    if (row < 0 || row + 1 >= static_cast<int>(activeDs().pipeline.size())) return;
    saveSnapshot();
    std::swap(activeDs().pipeline[row], activeDs().pipeline[row + 1]);
    rebuildTransformList();
    list_transforms_->setCurrentRow(row + 1);
    updateFileInfo();
    if (activeDs().plot_file_backed)
        plotWidget()->setTransforms(activeDs().pipeline);
    else
        rebakeAlignedView();
    plotWidget()->update();
}

void MainWindow::onResetView() {
    // View-only reset: pan/zoom back to the full range and clear crop
    // ranges. Alignment (align_shifts and any per-trace drag shifts) is
    // deliberately left untouched — Reset is for the view, not for
    // discarding an alignment you just computed. Nothing else clears
    // alignment either: onApplyTraces() ("Load / Refresh" / "Un-apply
    // Shifts") explicitly keeps it too, since it's addressed by absolute
    // trace index rather than by whatever range happens to be loaded — see
    // the comment in onApplyTraces(). It only goes away when the underlying
    // file changes, i.e. a fresh Dataset from openFile()/onOpenNpyTraces().
    if (hasActiveDs()) saveSnapshot();
    plotWidget()->resetView();
    plotWidget()->clearCropRanges();
    resetInteractionModeToPan();
}

void MainWindow::onViewChanged(int64_t start, int64_t end, int64_t /*total*/) {
    int64_t raw_span = end - start;
    int64_t eff_span = raw_span;
    if (!hasActiveDs()) { lbl_view_->setText(QString("View: [%1 – %2]").arg(start).arg(end)); return; }
    for (const auto& t : activeDs().pipeline)
        eff_span = t->transformedCount(eff_span);
    if (eff_span != raw_span)
        lbl_view_->setText(
            QString("View: [%1 – %2]\nSpan: %3  (%4 after pipeline)")
                .arg(start).arg(end).arg(raw_span).arg(eff_span));
    else
        lbl_view_->setText(
            QString("View: [%1 – %2]\nSpan: %3 samples")
                .arg(start).arg(end).arg(raw_span));
}

void MainWindow::onMeasurementUpdated(int64_t s1, double v1,
                                       int64_t s2, double v2, bool has_p2)
{
    if (s1 == 0 && s2 == 0 && !has_p2) {   // emitted by clearMeasurement()
        lbl_measure_->setText("–");
        return;
    }
    if (!has_p2) {
        lbl_measure_->setText(
            QString("P1:  sample = %1\n      value  = %2\n\nClick a second point…")
                .arg(s1).arg(v1, 0, 'g', 6));
    } else {
        int64_t ds = s2 - s1;
        double  dv = v2 - v1;
        lbl_measure_->setText(
            QString("P1:  s=%1  v=%2\nP2:  s=%3  v=%4\n"
                    "Δs = %5\nΔv = %6")
                .arg(s1).arg(v1, 0, 'g', 5)
                .arg(s2).arg(v2, 0, 'g', 5)
                .arg(ds)
                .arg(dv, 0, 'g', 5));
    }
}

void MainWindow::onThemeChanged(int index) {
    plotWidget()->setTheme(index == 0 ? PlotTheme::dark() : PlotTheme::light());
}

void MainWindow::rebuildTransformList() {
    list_transforms_->clear();
    for (int i = 0; i < static_cast<int>(activeDs().pipeline.size()); i++) {
        list_transforms_->addItem(
            QString("%1. %2").arg(i + 1)
                             .arg(QString::fromStdString(activeDs().pipeline[i]->name())));
    }
}

// ---------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------

// Fill out[0..raw_ns) with samples of trace `trace_idx` starting at
// `dst_base + shift`, zero-padding wherever that falls outside
// [0, n_samples_total). This is the same windowed/shifted read used by the
// t-test and cross-correlation accumulators to make "aligned" traces
// actually line up on export instead of the raw, un-shifted samples being
// read every time. shift == 0 degenerates to a plain windowed read.
static void readAlignedWindow(TrsFile* src, int32_t trace_idx, int64_t dst_base,
                               int64_t raw_ns, int32_t shift, int32_t n_samples_total,
                               float* out)
{
    std::fill(out, out + raw_ns, 0.0f);
    int64_t adj_start = dst_base + shift;
    if (adj_start >= n_samples_total || adj_start + raw_ns <= 0) return;
    int64_t src_start = std::max<int64_t>(0, adj_start);
    int64_t src_end   = std::min<int64_t>(n_samples_total, adj_start + raw_ns);
    int64_t dst_off    = src_start - adj_start;
    if (src_start < src_end)
        src->readSamples(trace_idx, src_start, src_end - src_start, out + dst_off);
}

// Convert `n` float samples to the file's native sample coding and write
// them, rounding to nearest and clamping to the representable range for
// integer codings. *clipped is incremented for every value that had to be
// clamped, so callers can warn the user. FLOAT32 writes are exact.
static void writeSamplesAs(FILE* fp, SampleType type, const float* data, int64_t n,
                            int64_t* clipped)
{
    switch (type) {
    case SampleType::FLOAT32:
        std::fwrite(data, sizeof(float), static_cast<size_t>(n), fp);
        return;
    case SampleType::INT8: {
        std::vector<int8_t> tmp(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            long v = std::lround(data[i]);
            long c = std::clamp<long>(v, -128, 127);
            if (c != v) (*clipped)++;
            tmp[static_cast<size_t>(i)] = static_cast<int8_t>(c);
        }
        std::fwrite(tmp.data(), 1, tmp.size(), fp);
        return;
    }
    case SampleType::INT16: {
        std::vector<int16_t> tmp(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            long v = std::lround(data[i]);
            long c = std::clamp<long>(v, -32768, 32767);
            if (c != v) (*clipped)++;
            tmp[static_cast<size_t>(i)] = static_cast<int16_t>(c);
        }
        std::fwrite(tmp.data(), 2, tmp.size(), fp);
        return;
    }
    case SampleType::INT32: {
        std::vector<int32_t> tmp(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            long long v = std::llround(data[i]);
            long long c = std::clamp<long long>(v, INT32_MIN, INT32_MAX);
            if (c != v) (*clipped)++;
            tmp[static_cast<size_t>(i)] = static_cast<int32_t>(c);
        }
        std::fwrite(tmp.data(), 4, tmp.size(), fp);
        return;
    }
    }
}

// Write a TRS file, preserving the source file's sample coding (int8/16/32/
// float32) instead of forcing float32, and applying `shifts[i]` (if
// non-empty) to trace `first_trace + i` the same way the t-test/xcorr
// accumulators do — traces marked kAlignDiscardShift are dropped from the
// output entirely. Streams in 256 K-sample chunks to keep RAM usage low.
// Returns false and sets err_out on failure. On success, *n_written_out (if
// non-null) receives the actual number of traces written, which can be less
// than `count` when shifts marks some traces discarded.
static bool exportTracesToTrs(
    const QString& out_path,
    TrsFile* src,
    int32_t first_trace, int32_t count,
    const std::vector<int32_t>& shifts,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    QProgressDialog* progress,
    QString& err_out,
    int32_t* n_written_out = nullptr)
{
    const TrsHeader& h = src->header();
    int32_t n_req  = std::min(count, h.num_traces - first_trace);
    if (n_req <= 0) { err_out = "No traces to export."; return false; }
    int32_t n_samples = h.num_samples;

    // Drop discarded traces up front so NUMBER_TRACES is correct in the
    // header, which is written before the trace loop below.
    std::vector<int32_t> keep;
    keep.reserve(static_cast<size_t>(n_req));
    for (int32_t i = 0; i < n_req; i++) {
        int32_t shift = (i < static_cast<int32_t>(shifts.size())) ? shifts[i] : 0;
        if (shift == kAlignDiscardShift) continue;
        keep.push_back(i);
    }
    int32_t n_traces = static_cast<int32_t>(keep.size());
    if (n_traces <= 0) { err_out = "No traces to export (all discarded by alignment)."; return false; }
    if (n_written_out) *n_written_out = n_traces;

    // Compute effective output sample count after pipeline transforms.
    int64_t out_samples = n_samples;
    for (const auto& t : pipeline)
        out_samples = t->transformedCount(out_samples);

    FILE* fp = std::fopen(out_path.toLocal8Bit().constData(), "wb");
    if (!fp) { err_out = "Cannot create file:\n" + out_path; return false; }

    // Helper lambdas -------------------------------------------------------
    auto write_le16 = [&](int16_t v) {
        uint8_t b[2] = { uint8_t(v & 0xFF), uint8_t((v >> 8) & 0xFF) };
        std::fwrite(b, 1, 2, fp);
    };
    auto write_le32 = [&](int32_t v) {
        uint8_t b[4] = { uint8_t(v), uint8_t(v>>8), uint8_t(v>>16), uint8_t(v>>24) };
        std::fwrite(b, 1, 4, fp);
    };
    auto write_tlv_hdr = [&](uint8_t tag, uint8_t len) {
        std::fputc(tag, fp);
        std::fputc(len, fp);
    };

    // SAMPLE_CODING byte: lower nibble = bytes/sample, bit 4 = float flag.
    // Preserve the source coding rather than always forcing float32 so a
    // plain int16 file re-exported after light processing round-trips as
    // int16 instead of silently doubling in size and changing type.
    uint8_t coding_byte = 0x14; // default FLOAT32
    switch (h.sample_type) {
    case SampleType::INT8:    coding_byte = 0x01; break;
    case SampleType::INT16:   coding_byte = 0x02; break;
    case SampleType::INT32:   coding_byte = 0x04; break;
    case SampleType::FLOAT32: coding_byte = 0x14; break;
    }

    // TRS header TLVs -------------------------------------------------------
    write_tlv_hdr(0x41, 4); write_le32(n_traces);                         // NUMBER_TRACES
    write_tlv_hdr(0x42, 4); write_le32(static_cast<int32_t>(out_samples)); // NUMBER_SAMPLES
    write_tlv_hdr(0x43, 1); std::fputc(coding_byte, fp);                  // SAMPLE_CODING
    if (h.data_length > 0) {
        write_tlv_hdr(0x44, 2); write_le16(h.data_length); // DATA_LENGTH
    }
    std::fputc(0x5F, fp); std::fputc(0x00, fp);      // TRACE_BLOCK

    // Trace data -------------------------------------------------------------
    constexpr int64_t CHUNK = 256 * 1024;
    std::vector<float> buf(CHUNK);
    int64_t clipped = 0;

    for (int32_t ki = 0; ki < n_traces; ki++) {
        if (progress) {
            if (progress->wasCanceled()) {
                std::fclose(fp);
                QFile::remove(out_path);
                err_out = "Export cancelled.";
                return false;
            }
            progress->setValue(ki);
            QApplication::processEvents();
        }

        int32_t i       = keep[static_cast<size_t>(ki)];
        int32_t src_idx = first_trace + i;
        int32_t shift   = (i < static_cast<int32_t>(shifts.size())) ? shifts[i] : 0;

        // Auxiliary data bytes (plaintext / key)
        if (h.data_length > 0) {
            auto data = src->readData(src_idx);
            std::fwrite(data.data(), 1, data.size(), fp);
        }

        // Reset transforms for each new trace
        for (auto& t : pipeline) t->reset();

        int64_t written = 0;
        while (written < n_samples) {
            int64_t chunk = std::min(CHUNK, n_samples - written);
            readAlignedWindow(src, src_idx, written, chunk, shift, n_samples, buf.data());
            int64_t out_count = chunk;
            for (auto& t : pipeline) out_count = t->apply(buf.data(), out_count, written);
            writeSamplesAs(fp, h.sample_type, buf.data(), out_count, &clipped);
            written += chunk;
        }
    }

    std::fclose(fp);
    if (clipped > 0) {
        err_out = QString("%1 sample value(s) were clamped to fit the original %2-bit format.")
                      .arg(clipped).arg(h.sample_size * 8);
        // Not a failure — err_out is repurposed as a non-fatal warning the
        // caller may surface alongside the success message.
    }
    return true;
}

// Adds an "Alignment" group box with an "Apply last alignment shifts"
// checkbox to `fl`, wired so that checking it fills sp_first/sp_count with
// the last computed alignment's range and disables manual entry — mirrors
// the pattern already used by the t-test/xcorr configuration dialogs, so
// export can apply the same alignment instead of silently ignoring it.
// Returns the checkbox so the caller can read isChecked() after accept.
static QCheckBox* addAlignmentGroup(QFormLayout* fl, bool has_alignment,
                                     int32_t align_first_trace, int align_shift_count,
                                     int discarded_count,
                                     QSpinBox* sp_first, QSpinBox* sp_count)
{
    auto* grp  = new QGroupBox("Alignment");
    auto* fl_a = new QFormLayout(grp);
    auto* chk  = new QCheckBox("Apply last alignment shifts");
    chk->setChecked(has_alignment);
    chk->setEnabled(has_alignment);
    chk->setToolTip(has_alignment
        ? QString("Use shifts from the last alignment run (%1 traces).%2")
              .arg(align_shift_count)
              .arg(discarded_count > 0
                       ? QString(" %1 of those are marked discarded and will be skipped.").arg(discarded_count)
                       : QString())
        : "No alignment has been applied to the main view yet.");
    fl_a->addRow(chk);
    QObject::connect(chk, &QCheckBox::toggled, chk,
        [=](bool on) {
            if (on) {
                sp_first->setValue(align_first_trace);
                sp_count->setValue(align_shift_count);
            }
            sp_first->setEnabled(!on);
            sp_count->setEnabled(!on);
        });
    if (has_alignment) {
        sp_first->setValue(align_first_trace);
        sp_count->setValue(align_shift_count);
        sp_first->setEnabled(false);
        sp_count->setEnabled(false);
    }
    fl->addRow(grp);
    return chk;
}

void MainWindow::onExportTrs() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Export TRS", "No file loaded.");
        return;
    }

    // Ask which traces to export (default: all).
    int n = activeDs().file->header().num_traces;

    QDialog range_dlg(this);
    range_dlg.setWindowTitle("Export TRS — select range");
    auto* fl       = new QFormLayout(&range_dlg);
    auto* sp_first = new QSpinBox;
    auto* sp_count = new QSpinBox;
    sp_first->setRange(0, std::max(0, n - 1));
    sp_first->setValue(0);
    sp_count->setRange(1, n);
    sp_count->setValue(n);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);

    const bool has_alignment = (activeDs().align_n_samples > 0);
    const int  n_discarded = static_cast<int>(std::count(
        activeDs().align_shifts.begin(), activeDs().align_shifts.end(), kAlignDiscardShift));
    QCheckBox* chk_align = addAlignmentGroup(fl, has_alignment,
        activeDs().align_first_trace, static_cast<int>(activeDs().align_shifts.size()),
        n_discarded, sp_first, sp_count);

    auto* bb = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &range_dlg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &range_dlg, &QDialog::reject);
    if (range_dlg.exec() != QDialog::Accepted) return;

    int32_t first = static_cast<int32_t>(sp_first->value());
    int32_t count = static_cast<int32_t>(sp_count->value());
    const bool use_alignment = chk_align->isChecked();
    const std::vector<int32_t> shifts = use_alignment ? activeDs().align_shifts
                                                        : std::vector<int32_t>{};

    QString path = QFileDialog::getSaveFileName(
        this, "Export processed TRS", recentDir("trs"), "TRS files (*.trs)");
    if (path.isEmpty()) return;
    updateRecentDir("trs", path);

    QProgressDialog progress("Exporting traces…", "Cancel",
                             0, count, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(500);

    QString err;
    int32_t n_written = 0;
    bool ok = exportTracesToTrs(path, activeDs().file.get(), first, count, shifts,
                                activeDs().pipeline, &progress, err, &n_written);
    progress.setValue(count);

    if (!ok)
        QMessageBox::critical(this, "Export failed", err);
    else
        QMessageBox::information(this, "Export complete",
            QString("Saved %1 trace(s) to:\n%2%3").arg(n_written).arg(path)
                .arg(err.isEmpty() ? QString() : "\n\n" + err));
}

void MainWindow::onExportPng() {
    if (!hasActiveDs()) {
        QMessageBox::information(this, "Export PNG", "No file loaded.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Export PNG", recentDir("export"), "PNG images (*.png)");
    if (path.isEmpty()) return;
    updateRecentDir("export", path);

    QPixmap px = plotWidget()->grab();
    QImage img = px.toImage();
    constexpr int kMaxPx = 2400;
    if (img.width() > kMaxPx || img.height() > kMaxPx)
        img = img.scaled(kMaxPx, kMaxPx, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    if (!img.save(path, "PNG"))
        QMessageBox::critical(this, "Export failed",
                              "Could not write PNG to:\n" + path);
}

void MainWindow::onExportPdf() {
    if (!hasActiveDs()) {
        QMessageBox::information(this, "Export PDF", "No file loaded.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Export PDF", recentDir("export"), "PDF files (*.pdf)");
    if (path.isEmpty()) return;
    updateRecentDir("export", path);

    // Grab the plot at current widget resolution, then paint into PDF.
    QPixmap px = plotWidget()->grab();

    QPdfWriter writer(path);
    writer.setResolution(150);
    // Use A4 landscape; the pixmap is scaled to fill the printable area.
    writer.setPageSize(QPageSize(QPageSize::A4));
    writer.setPageOrientation(QPageLayout::Landscape);
    writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout::Millimeter);

    QPainter painter(&writer);
    if (!painter.isActive()) {
        QMessageBox::critical(this, "Export failed",
                              "Could not initialise PDF writer for:\n" + path);
        return;
    }
    painter.drawPixmap(painter.viewport(), px);
    painter.end();
}

// ---------------------------------------------------------------------------
// T-test
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Generic NPY reader — supports float32 arrays of any shape.
// ---------------------------------------------------------------------------
static bool loadNpy(const QString& path,
                    std::vector<float>& data,
                    std::vector<int64_t>& shape,
                    QString& err)
{
    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "rb");
    if (!fp) { err = "Cannot open: " + path; return false; }

    // Magic: \x93NUMPY
    uint8_t magic[6] = {};
    std::fread(magic, 1, 6, fp);
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M'  || magic[4] != 'P' || magic[5] != 'Y') {
        std::fclose(fp); err = "Not a NumPy (.npy) file."; return false;
    }

    uint8_t ver[2] = {};
    std::fread(ver, 1, 2, fp);

    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint8_t hl[2] = {}; std::fread(hl, 1, 2, fp);
        header_len = static_cast<uint32_t>(hl[0]) | (static_cast<uint32_t>(hl[1]) << 8);
    } else {
        uint8_t hl[4] = {}; std::fread(hl, 1, 4, fp);
        header_len = static_cast<uint32_t>(hl[0])
                   | (static_cast<uint32_t>(hl[1]) <<  8)
                   | (static_cast<uint32_t>(hl[2]) << 16)
                   | (static_cast<uint32_t>(hl[3]) << 24);
    }

    std::string hdr(header_len, '\0');
    if (std::fread(hdr.data(), 1, header_len, fp) != header_len) {
        std::fclose(fp); err = "Truncated NPY header."; return false;
    }

    // Check dtype
    if (hdr.find("'<f4'") == std::string::npos &&
        hdr.find("\"<f4\"") == std::string::npos) {
        std::fclose(fp);
        err = "Only little-endian float32 ('<f4') arrays are supported.";
        return false;
    }

    // Parse shape: find '(' after 'shape'
    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos) sp = hdr.find("\"shape\"");
    if (sp == std::string::npos) { std::fclose(fp); err = "Cannot find shape."; return false; }
    auto lp = hdr.find('(', sp);
    auto rp = hdr.find(')', lp != std::string::npos ? lp : 0);
    if (lp == std::string::npos || rp == std::string::npos) {
        std::fclose(fp); err = "Cannot parse shape."; return false;
    }
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() &&
               (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        if (!std::isdigit(static_cast<unsigned char>(shape_str[pos]))) break;
        size_t end = pos;
        while (end < shape_str.size() &&
               std::isdigit(static_cast<unsigned char>(shape_str[end]))) end++;
        shape.push_back(static_cast<int64_t>(std::stoll(shape_str.substr(pos, end - pos))));
        pos = end;
    }
    if (shape.empty()) { std::fclose(fp); err = "Empty shape."; return false; }

    int64_t n_elements = 1;
    for (int64_t d : shape) n_elements *= d;

    data.resize(static_cast<size_t>(n_elements));
    size_t nread = std::fread(data.data(), sizeof(float),
                              static_cast<size_t>(n_elements), fp);
    std::fclose(fp);
    if (static_cast<int64_t>(nread) != n_elements) {
        err = "File too short — expected " + QString::number(n_elements) + " float32 values.";
        return false;
    }
    return true;
}

static std::vector<uint8_t> buildNpy1DBytes(const std::string& dtype,
                                              const void* data, int64_t n, size_t elem_size)
{
    std::string dict = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': ("
                     + std::to_string(n) + ",), }";
    size_t content_len = dict.size() + 1;
    size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
    dict.resize(header_len - 1, ' ');
    dict += '\n';
    uint16_t hl = static_cast<uint16_t>(header_len);
    size_t data_bytes = static_cast<size_t>(n) * elem_size;
    std::vector<uint8_t> out;
    out.reserve(10 + header_len + data_bytes);
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
                              uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
    out.insert(out.end(), magic, magic + 10);
    out.insert(out.end(), dict.begin(), dict.end());
    const auto* d = reinterpret_cast<const uint8_t*>(data);
    out.insert(out.end(), d, d + data_bytes);
    return out;
}

static bool parseNpyBytesFloat32(const std::vector<uint8_t>& bytes,
                                  std::vector<float>& out, QString& err)
{
    if (bytes.size() < 10 || bytes[0] != 0x93 || bytes[1] != 'N') {
        err = "Invalid NPY block."; return false;
    }
    uint8_t ver = bytes[6];
    size_t data_start = (ver == 1) ? 10 + (bytes[8] | (bytes[9] << 8))
                                   : 12 + (bytes.size() < 12 ? 0 :
                                           (bytes[8] | (bytes[9]<<8) | (bytes[10]<<16) | (bytes[11]<<24)));
    if (data_start > bytes.size()) { err = "NPY header truncated."; return false; }
    std::string hdr(bytes.begin() + (ver == 1 ? 10 : 12), bytes.begin() + data_start);
    if (hdr.find("'<f4'") == std::string::npos && hdr.find("\"<f4\"") == std::string::npos) {
        err = "Expected float32 ('<f4') array."; return false;
    }
    size_t n = (bytes.size() - data_start) / sizeof(float);
    out.resize(n);
    std::memcpy(out.data(), bytes.data() + data_start, n * sizeof(float));
    return true;
}

static bool parseNpyBytesFloat64(const std::vector<uint8_t>& bytes,
                                  std::vector<double>& out, QString& err)
{
    if (bytes.size() < 10 || bytes[0] != 0x93 || bytes[1] != 'N') {
        err = "Invalid NPY block."; return false;
    }
    uint8_t ver = bytes[6];
    size_t data_start = (ver == 1) ? 10 + (bytes[8] | (bytes[9] << 8))
                                   : 12 + (bytes.size() < 12 ? 0 :
                                           (bytes[8] | (bytes[9]<<8) | (bytes[10]<<16) | (bytes[11]<<24)));
    if (data_start > bytes.size()) { err = "NPY header truncated."; return false; }
    std::string hdr(bytes.begin() + (ver == 1 ? 10 : 12), bytes.begin() + data_start);
    if (hdr.find("'<f8'") == std::string::npos && hdr.find("\"<f8\"") == std::string::npos) {
        err = "Expected float64 ('<f8') array."; return false;
    }
    size_t n = (bytes.size() - data_start) / sizeof(double);
    out.resize(n);
    std::memcpy(out.data(), bytes.data() + data_start, n * sizeof(double));
    return true;
}

static bool saveNpy(const QString& path, const float* data, int64_t n, QString& err) {
    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
    if (!fp) { err = "Cannot create: " + path; return false; }

    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    std::fwrite(magic, 1, 8, fp);

    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(static_cast<uint64_t>(n)) + ",), }";
    // Pad header to multiple of 64 bytes total (10-byte prefix + header)
    size_t content_len = dict.size() + 1;  // +1 for '\n'
    size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
    dict.resize(header_len - 1, ' ');
    dict += '\n';

    uint16_t hl = static_cast<uint16_t>(header_len);
    uint8_t hl_bytes[2] = {uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
    std::fwrite(hl_bytes, 1, 2, fp);
    std::fwrite(dict.c_str(), 1, dict.size(), fp);
    std::fwrite(data, sizeof(float), static_cast<size_t>(n), fp);
    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// 2-D NPY helpers (n_rows × n_cols float32 or uint8, C-order / row-major)
// ---------------------------------------------------------------------------

static std::vector<uint8_t> buildNpyBytes(const std::string& dtype,
                                           int64_t n_rows, int64_t n_cols,
                                           const void* data, size_t data_bytes)
{
    std::string dict = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': ("
                     + std::to_string(n_rows) + ", " + std::to_string(n_cols) + "), }";
    size_t content_len = dict.size() + 1;
    size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
    dict.resize(header_len - 1, ' ');
    dict += '\n';

    uint16_t hl = static_cast<uint16_t>(header_len);
    std::vector<uint8_t> out;
    out.reserve(10 + header_len + data_bytes);
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
                              uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
    out.insert(out.end(), magic, magic + 10);
    out.insert(out.end(), dict.begin(), dict.end());
    const auto* d = reinterpret_cast<const uint8_t*>(data);
    out.insert(out.end(), d, d + data_bytes);
    return out;
}

static bool saveNpy2D(const QString& path, const float* data,
                       int64_t n_rows, int64_t n_cols, QString& err)
{
    auto bytes = buildNpyBytes("<f4", n_rows, n_cols, data,
                                static_cast<size_t>(n_rows * n_cols) * sizeof(float));
    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
    if (!fp) { err = "Cannot create: " + path; return false; }
    std::fwrite(bytes.data(), 1, bytes.size(), fp);
    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// Minimal uncompressed NPZ writer (ZIP STORE method, no compression).
// Entries is a list of (filename, raw_bytes) pairs.
// ---------------------------------------------------------------------------

static uint32_t zip_crc32(const uint8_t* data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320u & ~((crc & 1u) - 1u));
    }
    return ~crc;
}

static void zip_write_u16(std::vector<uint8_t>& v, uint16_t x) {
    v.push_back(uint8_t(x));      v.push_back(uint8_t(x >> 8));
}
static void zip_write_u32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(uint8_t(x));      v.push_back(uint8_t(x >> 8));
    v.push_back(uint8_t(x >> 16)); v.push_back(uint8_t(x >> 24));
}

static bool saveNpz(const QString& path,
                     const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries,
                     QString& err)
{
    std::vector<uint8_t> buf;
    buf.reserve(1 << 20);

    struct CDEntry {
        std::string name;
        uint32_t crc, size, offset;
    };
    std::vector<CDEntry> cd;

    for (const auto& [name, data] : entries) {
        uint32_t crc  = zip_crc32(data.data(), data.size());
        uint32_t sz   = static_cast<uint32_t>(data.size());
        uint32_t off  = static_cast<uint32_t>(buf.size());
        cd.push_back({name, crc, sz, off});

        // Local file header
        zip_write_u32(buf, 0x04034b50u); // signature
        zip_write_u16(buf, 20);          // version needed: 2.0
        zip_write_u16(buf, 0);           // flags
        zip_write_u16(buf, 0);           // compression: store
        zip_write_u16(buf, 0);           // mod time
        zip_write_u16(buf, 0);           // mod date
        zip_write_u32(buf, crc);
        zip_write_u32(buf, sz);          // compressed size
        zip_write_u32(buf, sz);          // uncompressed size
        zip_write_u16(buf, static_cast<uint16_t>(name.size()));
        zip_write_u16(buf, 0);           // extra field length
        buf.insert(buf.end(), name.begin(), name.end());
        buf.insert(buf.end(), data.begin(), data.end());
    }

    uint32_t cd_offset = static_cast<uint32_t>(buf.size());

    for (const auto& e : cd) {
        zip_write_u32(buf, 0x02014b50u); // central dir signature
        zip_write_u16(buf, 20);          // version made by
        zip_write_u16(buf, 20);          // version needed
        zip_write_u16(buf, 0);           // flags
        zip_write_u16(buf, 0);           // compression: store
        zip_write_u16(buf, 0);           // mod time
        zip_write_u16(buf, 0);           // mod date
        zip_write_u32(buf, e.crc);
        zip_write_u32(buf, e.size);
        zip_write_u32(buf, e.size);
        zip_write_u16(buf, static_cast<uint16_t>(e.name.size()));
        zip_write_u16(buf, 0);  zip_write_u16(buf, 0);
        zip_write_u16(buf, 0);  zip_write_u16(buf, 0);
        zip_write_u32(buf, 0);  zip_write_u32(buf, e.offset);
        buf.insert(buf.end(), e.name.begin(), e.name.end());
    }

    uint32_t cd_size = static_cast<uint32_t>(buf.size()) - cd_offset;
    uint16_t n_entries = static_cast<uint16_t>(cd.size());

    // End of central directory
    zip_write_u32(buf, 0x06054b50u);
    zip_write_u16(buf, 0);  zip_write_u16(buf, 0);
    zip_write_u16(buf, n_entries); zip_write_u16(buf, n_entries);
    zip_write_u32(buf, cd_size);
    zip_write_u32(buf, cd_offset);
    zip_write_u16(buf, 0);  // comment length

    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
    if (!fp) { err = "Cannot create: " + path; return false; }
    std::fwrite(buf.data(), 1, buf.size(), fp);
    std::fclose(fp);
    return true;
}

// ---------------------------------------------------------------------------
// Minimal NPZ reader: returns named entries as raw byte vectors.
// Supports STORE (method 0) and DEFLATE (method 8) compression.
// ---------------------------------------------------------------------------

static bool loadNpz(const QString& path,
                     std::map<std::string, std::vector<uint8_t>>& entries,
                     QString& err)
{
    FILE* fp = std::fopen(path.toLocal8Bit().constData(), "rb");
    if (!fp) { err = "Cannot open: " + path; return false; }

    // Find end-of-central-directory by scanning backwards
    std::fseek(fp, 0, SEEK_END);
    long file_size = std::ftell(fp);
    if (file_size < 22) { std::fclose(fp); err = "File too small to be NPZ."; return false; }

    const int max_search = std::min((long)65558, file_size);
    std::vector<uint8_t> tail(static_cast<size_t>(max_search));
    std::fseek(fp, file_size - max_search, SEEK_SET);
    std::fread(tail.data(), 1, tail.size(), fp);

    int eocd_off = -1;
    for (int i = static_cast<int>(tail.size()) - 22; i >= 0; i--) {
        if (tail[i] == 0x50 && tail[i+1] == 0x4B && tail[i+2] == 0x05 && tail[i+3] == 0x06) {
            eocd_off = i;
            break;
        }
    }
    if (eocd_off < 0) { std::fclose(fp); err = "No EOCD found; not a valid ZIP/NPZ."; return false; }

    const uint8_t* eocd = tail.data() + eocd_off;
    uint16_t n_entries  = static_cast<uint16_t>(eocd[8])  | (static_cast<uint16_t>(eocd[9])  << 8);
    uint32_t cd_size    = static_cast<uint32_t>(eocd[12]) | (static_cast<uint32_t>(eocd[13]) << 8)
                        | (static_cast<uint32_t>(eocd[14]) << 16) | (static_cast<uint32_t>(eocd[15]) << 24);
    uint32_t cd_offset  = static_cast<uint32_t>(eocd[16]) | (static_cast<uint32_t>(eocd[17]) << 8)
                        | (static_cast<uint32_t>(eocd[18]) << 16) | (static_cast<uint32_t>(eocd[19]) << 24);

    std::vector<uint8_t> cd(cd_size);
    std::fseek(fp, static_cast<long>(cd_offset), SEEK_SET);
    if (std::fread(cd.data(), 1, cd_size, fp) != cd_size) {
        std::fclose(fp); err = "Cannot read central directory."; return false;
    }

    size_t pos = 0;
    for (uint16_t ei = 0; ei < n_entries; ei++) {
        if (pos + 46 > cd.size()) break;
        if (cd[pos] != 0x50 || cd[pos+1] != 0x4B || cd[pos+2] != 0x01 || cd[pos+3] != 0x02) break;
        uint16_t method   = static_cast<uint16_t>(cd[pos+10]) | (static_cast<uint16_t>(cd[pos+11]) << 8);
        uint32_t comp_sz  = static_cast<uint32_t>(cd[pos+20]) | (static_cast<uint32_t>(cd[pos+21]) << 8)
                          | (static_cast<uint32_t>(cd[pos+22]) << 16) | (static_cast<uint32_t>(cd[pos+23]) << 24);
        uint32_t uncomp_sz= static_cast<uint32_t>(cd[pos+24]) | (static_cast<uint32_t>(cd[pos+25]) << 8)
                          | (static_cast<uint32_t>(cd[pos+26]) << 16) | (static_cast<uint32_t>(cd[pos+27]) << 24);
        uint16_t fname_len= static_cast<uint16_t>(cd[pos+28]) | (static_cast<uint16_t>(cd[pos+29]) << 8);
        uint16_t extra_len= static_cast<uint16_t>(cd[pos+30]) | (static_cast<uint16_t>(cd[pos+31]) << 8);
        uint16_t comm_len = static_cast<uint16_t>(cd[pos+32]) | (static_cast<uint16_t>(cd[pos+33]) << 8);
        uint32_t lh_offset= static_cast<uint32_t>(cd[pos+42]) | (static_cast<uint32_t>(cd[pos+43]) << 8)
                          | (static_cast<uint32_t>(cd[pos+44]) << 16) | (static_cast<uint32_t>(cd[pos+45]) << 24);
        std::string fname(reinterpret_cast<const char*>(cd.data() + pos + 46), fname_len);
        pos += 46 + fname_len + extra_len + comm_len;

        if (method != 0 && method != 8) {
            err = QString("Entry '%1' uses compression method %2; only STORE (0) and DEFLATE (8) are supported.")
                      .arg(QString::fromStdString(fname)).arg(method);
            std::fclose(fp); return false;
        }

        // Read local file header to find data offset
        std::fseek(fp, static_cast<long>(lh_offset) + 26, SEEK_SET);
        uint8_t lh_extra[4] = {};
        std::fread(lh_extra, 1, 4, fp);
        uint16_t lh_fname_len = static_cast<uint16_t>(lh_extra[0]) | (static_cast<uint16_t>(lh_extra[1]) << 8);
        uint16_t lh_extra_len = static_cast<uint16_t>(lh_extra[2]) | (static_cast<uint16_t>(lh_extra[3]) << 8);
        long data_off = static_cast<long>(lh_offset) + 30 + lh_fname_len + lh_extra_len;
        std::fseek(fp, data_off, SEEK_SET);

        std::vector<uint8_t> data(uncomp_sz);
        if (method == 0) {
            // STORE — raw bytes
            if (std::fread(data.data(), 1, uncomp_sz, fp) != uncomp_sz) {
                err = QString("Truncated data for entry '%1'.").arg(QString::fromStdString(fname));
                std::fclose(fp); return false;
            }
        } else {
            // DEFLATE (raw, no zlib wrapper) — inflate with -MAX_WBITS
            std::vector<uint8_t> comp(comp_sz);
            if (std::fread(comp.data(), 1, comp_sz, fp) != comp_sz) {
                err = QString("Truncated compressed data for entry '%1'.").arg(QString::fromStdString(fname));
                std::fclose(fp); return false;
            }
            z_stream zs{};
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
                err = "zlib inflateInit2 failed."; std::fclose(fp); return false;
            }
            zs.next_in  = comp.data();
            zs.avail_in = static_cast<uInt>(comp_sz);
            zs.next_out = data.data();
            zs.avail_out= static_cast<uInt>(uncomp_sz);
            int zret = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (zret != Z_STREAM_END) {
                err = QString("DEFLATE decompression failed for entry '%1' (zlib code %2).")
                          .arg(QString::fromStdString(fname)).arg(zret);
                std::fclose(fp); return false;
            }
        }
        entries[fname] = std::move(data);
    }

    std::fclose(fp);
    return true;
}

// Parse a NPY array from an in-memory byte buffer into a flat float32 vector + shape.
static bool parseNpyBytes(const std::vector<uint8_t>& buf,
                           std::vector<float>& data,
                           std::vector<int64_t>& shape,
                           QString& err)
{
    if (buf.size() < 10) { err = "NPY entry too small."; return false; }
    if (buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M'  || buf[4] != 'P' || buf[5] != 'Y') {
        err = "Entry does not have NPY magic."; return false;
    }
    uint8_t ver = buf[6];
    uint32_t header_len = 0;
    size_t hdr_start;
    if (ver == 1) {
        header_len = static_cast<uint32_t>(buf[8]) | (static_cast<uint32_t>(buf[9]) << 8);
        hdr_start = 10;
    } else {
        if (buf.size() < 12) { err = "NPY v2+ header too short."; return false; }
        header_len = static_cast<uint32_t>(buf[8])  | (static_cast<uint32_t>(buf[9])  << 8)
                   | (static_cast<uint32_t>(buf[10]) << 16) | (static_cast<uint32_t>(buf[11]) << 24);
        hdr_start = 12;
    }
    if (hdr_start + header_len > buf.size()) { err = "NPY header truncated."; return false; }
    std::string hdr(reinterpret_cast<const char*>(buf.data() + hdr_start), header_len);

    // dtype detection
    bool is_f32 = (hdr.find("'<f4'") != std::string::npos || hdr.find("\"<f4\"") != std::string::npos);
    bool is_f64 = (hdr.find("'<f8'") != std::string::npos || hdr.find("\"<f8\"") != std::string::npos ||
                   hdr.find("'float64'") != std::string::npos);
    bool is_u8  = (hdr.find("'|u1'") != std::string::npos || hdr.find("\"u1\""  ) != std::string::npos ||
                   hdr.find("'uint8'") != std::string::npos);
    bool is_i16 = (hdr.find("'<i2'") != std::string::npos || hdr.find("\"<i2\"") != std::string::npos);
    bool is_i32 = (hdr.find("'<i4'") != std::string::npos || hdr.find("\"<i4\"") != std::string::npos);
    bool is_i64 = (hdr.find("'<i8'") != std::string::npos || hdr.find("\"<i8\"") != std::string::npos);
    if (!is_f32 && !is_f64 && !is_u8 && !is_i16 && !is_i32 && !is_i64) {
        err = "Unsupported dtype. Supported: float32, float64, uint8, int16, int32, int64."; return false;
    }

    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos) sp = hdr.find("\"shape\"");
    if (sp == std::string::npos) { err = "Cannot find shape in NPY header."; return false; }
    auto lp = hdr.find('(', sp);
    auto rp = hdr.find(')', lp != std::string::npos ? lp : 0);
    if (lp == std::string::npos || rp == std::string::npos) { err = "Cannot parse shape."; return false; }
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() &&
               (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size() || !std::isdigit(static_cast<unsigned char>(shape_str[pos]))) break;
        size_t end = pos;
        while (end < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[end]))) end++;
        shape.push_back(std::stoll(shape_str.substr(pos, end - pos)));
        pos = end;
    }
    if (shape.empty()) { err = "Empty shape."; return false; }

    int64_t n_elements = 1;
    for (int64_t d : shape) n_elements *= d;

    size_t data_offset = hdr_start + header_len;
    size_t elem_bytes = is_f32 ? 4 : is_f64 ? 8 : is_u8 ? 1 : is_i16 ? 2 : is_i32 ? 4 : 8;
    if (data_offset + static_cast<size_t>(n_elements) * elem_bytes > buf.size()) {
        err = "NPY data truncated."; return false;
    }
    data.resize(static_cast<size_t>(n_elements));
    const uint8_t* src = buf.data() + data_offset;
    if (is_f32) {
        std::memcpy(data.data(), src, static_cast<size_t>(n_elements) * 4);
    } else if (is_f64) {
        for (int64_t i = 0; i < n_elements; i++) {
            double v; std::memcpy(&v, src + i * 8, 8);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    } else if (is_u8) {
        for (int64_t i = 0; i < n_elements; i++)
            data[static_cast<size_t>(i)] = static_cast<float>(src[i]);
    } else if (is_i16) {
        for (int64_t i = 0; i < n_elements; i++) {
            int16_t v; std::memcpy(&v, src + i * 2, 2);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    } else if (is_i32) {
        for (int64_t i = 0; i < n_elements; i++) {
            int32_t v; std::memcpy(&v, src + i * 4, 4);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    } else { // i64
        for (int64_t i = 0; i < n_elements; i++) {
            int64_t v; std::memcpy(&v, src + i * 8, 8);
            data[static_cast<size_t>(i)] = static_cast<float>(v);
        }
    }
    return true;
}

// Parse NPY bytes without converting to float — returns shape, elem_size, and raw data payload.
// Supports: |u1 |i1 <u2 <i2 <u4 <i4 <u8 <i8 <f4 <f8
static bool parseNpyRaw(const std::vector<uint8_t>& buf,
                        std::vector<int64_t>& shape,
                        int& elem_size,
                        std::vector<uint8_t>& payload,
                        QString& err)
{
    if (buf.size() < 10 || buf[0] != 0x93 || buf[1] != 'N' || buf[2] != 'U' ||
        buf[3] != 'M'  || buf[4] != 'P' || buf[5] != 'Y') {
        err = "Not a valid NPY entry."; return false;
    }
    uint8_t ver = buf[6];
    uint32_t hlen = 0;
    size_t hstart;
    if (ver == 1) {
        hlen = static_cast<uint32_t>(buf[8]) | (static_cast<uint32_t>(buf[9]) << 8);
        hstart = 10;
    } else {
        if (buf.size() < 12) { err = "NPY v2 header too short."; return false; }
        hlen = static_cast<uint32_t>(buf[8])  | (static_cast<uint32_t>(buf[9])  << 8)
             | (static_cast<uint32_t>(buf[10]) << 16) | (static_cast<uint32_t>(buf[11]) << 24);
        hstart = 12;
    }
    if (hstart + hlen > buf.size()) { err = "NPY header truncated."; return false; }
    std::string hdr(reinterpret_cast<const char*>(buf.data() + hstart), hlen);

    // elem_size from dtype
    struct { const char* token; int sz; } dtypes[] = {
        {"|u1",1},{"|i1",1},{"'u1'",1},{"'i1'",1},
        {"<u2",2},{"<i2",2},{"<u4",4},{"<i4",4},
        {"<u8",8},{"<i8",8},{"<f4",4},{"<f8",8},
        {nullptr,0}
    };
    elem_size = 0;
    for (auto& d : dtypes) {
        if (!d.token) break;
        if (hdr.find(d.token) != std::string::npos) { elem_size = d.sz; break; }
    }
    if (elem_size == 0) { err = "Unsupported dtype in NPY entry."; return false; }

    // shape
    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos) sp = hdr.find("\"shape\"");
    if (sp == std::string::npos) { err = "Cannot find shape."; return false; }
    auto lp = hdr.find('(', sp);
    auto rp = hdr.find(')', lp != std::string::npos ? lp : 0);
    if (lp == std::string::npos || rp == std::string::npos) { err = "Cannot parse shape."; return false; }
    std::string shape_str = hdr.substr(lp + 1, rp - lp - 1);
    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
        if (pos >= shape_str.size()) break;
        char* end = nullptr;
        long long v = std::strtoll(shape_str.c_str() + pos, &end, 10);
        if (end == shape_str.c_str() + pos) break;
        shape.push_back(static_cast<int64_t>(v));
        pos = static_cast<size_t>(end - shape_str.c_str());
    }

    size_t data_start = hstart + hlen;
    size_t data_bytes = buf.size() - data_start;
    payload.assign(buf.data() + data_start, buf.data() + data_start + data_bytes);
    return true;
}

// ---------------------------------------------------------------------------
// Load NPY helpers
// ---------------------------------------------------------------------------

void MainWindow::onLoadNpyTTest() {
    QString path = QFileDialog::getOpenFileName(
        this, "Load t-test NPY/NPZ", recentDir("npy"),
        "NumPy files (*.npy *.npz);;All files (*)");
    if (path.isEmpty()) return;
    updateRecentDir("npy", path);

    std::vector<float> data;
    std::vector<double> df_loaded;
    QString err;

    if (path.endsWith(".npz", Qt::CaseInsensitive)) {
        std::map<std::string, std::vector<uint8_t>> entries;
        if (!loadNpz(path, entries, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }
        auto it = entries.find("tstat.npy");
        if (it == entries.end()) {
            QMessageBox::critical(this, "Load failed",
                "NPZ does not contain 'tstat.npy'."); return;
        }
        if (!parseNpyBytesFloat32(it->second, data, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }
        auto it2 = entries.find("df.npy");
        if (it2 != entries.end()) {
            QString df_err;
            parseNpyBytesFloat64(it2->second, df_loaded, df_err);
        }
    } else {
        std::vector<int64_t> shape;
        if (!loadNpy(path, data, shape, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }
        if (shape.size() != 1) {
            QMessageBox::critical(this, "Load failed",
                QString("Expected a 1-D array, got %1-D.").arg(shape.size())); return;
        }
    }

    auto tstat_ptr   = std::make_shared<std::vector<float>>(std::move(data));
    auto df_ptr      = std::make_shared<std::vector<double>>(std::move(df_loaded));
    auto current_ptr = std::make_shared<std::vector<float>>(*tstat_ptr);

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(QString("T-test — %1 samples — %2")
                            .arg(tstat_ptr->size())
                            .arg(QFileInfo(path).fileName()));
    dlg->resize(1100, 520);

    auto* pw = new PlotWidget(dlg);
    pw->setTheme(PlotTheme::light());
    pw->addTrace(current_ptr, QColor("#1f77b4"), "t-value");
    pw->setTraceFilled(0, true);
    pw->setAxisLabels("Sample Index", "t-value");
    pw->setThresholds(true, 4.5, -4.5);
    pw->resetView();

    auto* lbl_thr  = new QLabel("Threshold ±:");
    auto* spin_thr = new QDoubleSpinBox;
    spin_thr->setRange(0.1, 1000.0); spin_thr->setValue(4.5);
    spin_thr->setDecimals(2); spin_thr->setSingleStep(0.1);
    connect(spin_thr, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [pw](double v) { pw->setThresholds(true, v, -v); });

    auto* chk_onesided_npy = new QCheckBox("One-sided (+)");
    chk_onesided_npy->setToolTip("Show only positive threshold (use after abs() preprocessing)");
    connect(chk_onesided_npy, &QCheckBox::toggled, dlg, [pw, lbl_thr](bool on) {
        pw->setThresholdOneSided(on);
        lbl_thr->setText(on ? "Threshold +:" : "Threshold ±:");
    });

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, dlg, [dlg, current_ptr]() {
        QString p = QFileDialog::getSaveFileName(dlg, "Export t-test as NumPy",
                                                 MainWindow::recentDir("npy"), "NumPy files (*.npy)");
        if (p.isEmpty()) return;
        MainWindow::updateRecentDir("npy", p);
        QString e;
        if (!saveNpy(p, current_ptr->data(), static_cast<int64_t>(current_ptr->size()), e))
            QMessageBox::critical(dlg, "Export failed", e);
        else
            QMessageBox::information(dlg, "Saved", "Saved: " + p);
    });

    auto* btn_calc_th_npy = new QPushButton("Calc TH…");
    connect(btn_calc_th_npy, &QPushButton::clicked, dlg, [=]() {
        auto* cd = new QDialog(dlg);
        cd->setWindowTitle("Threshold Calculator");
        cd->setWindowModality(Qt::WindowModal);
        auto* fl = new QFormLayout(cd);

        auto* sp_alpha = new QDoubleSpinBox;
        sp_alpha->setRange(1e-6, 0.5); sp_alpha->setDecimals(6);
        sp_alpha->setValue(0.01);      sp_alpha->setSingleStep(0.01);

        int64_t n_L = static_cast<int64_t>(tstat_ptr->size());
        auto* lbl_nL = new QLabel(QString::number(n_L));

        auto* lbl_ath = new QLabel;
        auto* lbl_nu  = new QLabel;
        auto* lbl_th  = new QLabel;
        lbl_th->setTextFormat(Qt::RichText);

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Apply | QDialogButtonBox::Close);
        connect(bb, &QDialogButtonBox::rejected, cd, &QDialog::close);

        fl->addRow("Significance level α:", sp_alpha);
        fl->addRow("Trace length n_L:",      lbl_nL);
        fl->addRow(new QLabel);
        fl->addRow("Šidák α_TH:",            lbl_ath);

        if (!df_ptr->empty()) {
            // Use median Welch df from the loaded df array
            std::vector<double> df_sorted = *df_ptr;
            std::sort(df_sorted.begin(), df_sorted.end());
            double median_nu = df_sorted[df_sorted.size() / 2];

            auto recalc = [=]() {
                double a    = sp_alpha->value();
                double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
                double th   = invTCdf(1.0 - a_th / 2.0, median_nu);
                lbl_ath->setText(QString::number(a_th, 'g', 4));
                lbl_nu ->setText(QString::number(median_nu, 'f', 1));
                lbl_th ->setText(QString("<b>%1</b>").arg(th, 0, 'f', 4));
            };
            connect(sp_alpha, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                    cd, [=](double) { recalc(); });
            connect(bb->button(QDialogButtonBox::Apply), &QPushButton::clicked, cd, [=]() {
                double a    = sp_alpha->value();
                double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
                spin_thr->setValue(invTCdf(1.0 - a_th / 2.0, median_nu));
            });
            fl->addRow("Median Welch ν̂:", lbl_nu);
            recalc();
        } else {
            // No df data — let user enter group sizes for approximation
            auto* sp_nA = new QSpinBox; sp_nA->setRange(2, 10000000); sp_nA->setValue(100);
            auto* sp_nB = new QSpinBox; sp_nB->setRange(2, 10000000); sp_nB->setValue(100);

            auto calc_nu = [](int64_t nA, int64_t nB) -> double {
                double a = static_cast<double>(nA), b = static_cast<double>(nB);
                double num = (1.0/a + 1.0/b) * (1.0/a + 1.0/b);
                double den = 1.0/(a*a*(a-1.0)) + 1.0/(b*b*(b-1.0));
                return (den > 0.0) ? num / den : a + b - 2.0;
            };

            auto recalc = [=]() {
                double a    = sp_alpha->value();
                double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
                double nu   = calc_nu(sp_nA->value(), sp_nB->value());
                double th   = invTCdf(1.0 - a_th / 2.0, nu);
                lbl_ath->setText(QString::number(a_th, 'g', 4));
                lbl_nu ->setText(QString::number(nu, 'f', 1));
                lbl_th ->setText(QString("<b>%1</b>").arg(th, 0, 'f', 4));
            };
            connect(sp_alpha, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                    cd, [=](double) { recalc(); });
            connect(sp_nA, QOverload<int>::of(&QSpinBox::valueChanged), cd, [=](int) { recalc(); });
            connect(sp_nB, QOverload<int>::of(&QSpinBox::valueChanged), cd, [=](int) { recalc(); });
            connect(bb->button(QDialogButtonBox::Apply), &QPushButton::clicked, cd, [=]() {
                double a    = sp_alpha->value();
                double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
                spin_thr->setValue(invTCdf(1.0 - a_th / 2.0, calc_nu(sp_nA->value(), sp_nB->value())));
            });
            fl->addRow("Group A  n_A:",       sp_nA);
            fl->addRow("Group B  n_B:",       sp_nB);
            fl->addRow("Approx. Welch ν̂:",   lbl_nu);
            recalc();
        }

        fl->addRow("Threshold TH:", lbl_th);
        fl->addRow(bb);
        cd->show();
    });

    // Style dialog for NPY dialog
    auto* btn_style_npy = new QPushButton("Style…");
    connect(btn_style_npy, &QPushButton::clicked, dlg, [=]() {
        auto* sd = new QDialog(dlg);
        sd->setWindowTitle("Plot Style");
        sd->setWindowModality(Qt::NonModal);
        auto* fl2 = new QFormLayout(sd);

        auto* le_title = new QLineEdit;
        le_title->setPlaceholderText("e.g. Welch t-test — AES-128 key byte 0");
        connect(le_title, &QLineEdit::textChanged, sd, [pw](const QString& t) { pw->setTitle(t); });

        auto* sp_width = new QDoubleSpinBox;
        sp_width->setRange(0.5, 6.0); sp_width->setValue(1.5); sp_width->setSingleStep(0.5);
        connect(sp_width, QOverload<double>::of(&QDoubleSpinBox::valueChanged), sd,
                [pw](double v) { pw->setTraceWidth(static_cast<float>(v)); });

        auto* btn_color = new QPushButton("Pick color…");
        btn_color->setStyleSheet(QString("background:%1").arg(QColor("#4fc3f7").name()));
        connect(btn_color, &QPushButton::clicked, sd, [=]() {
            QColor c = QColorDialog::getColor(QColor("#4fc3f7"), sd);
            if (!c.isValid()) return;
            pw->setTraceColor(0, c);
            btn_color->setStyleSheet(QString("background:%1").arg(c.name()));
        });

        auto* btn_dark  = new QPushButton("Dark theme");
        auto* btn_light = new QPushButton("Light theme");
        connect(btn_dark,  &QPushButton::clicked, sd, [pw]() { pw->setTheme(PlotTheme::dark()); });
        connect(btn_light, &QPushButton::clicked, sd, [pw]() { pw->setTheme(PlotTheme::light()); });

        auto* bb2 = new QDialogButtonBox(QDialogButtonBox::Close);
        connect(bb2, &QDialogButtonBox::rejected, sd, &QDialog::close);

        fl2->addRow("Title:",       le_title);
        fl2->addRow("Line width:",  sp_width);
        fl2->addRow("Trace color:", btn_color);
        auto* theme_row = new QWidget; auto* trl = new QHBoxLayout(theme_row);
        trl->setContentsMargins(0,0,0,0); trl->addWidget(btn_dark); trl->addWidget(btn_light);
        fl2->addRow("Theme:", theme_row);
        fl2->addRow(bb2);
        sd->show();
    });

    auto* btn_exp_pdf_npy = new QPushButton("Export PDF…");
    connect(btn_exp_pdf_npy, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export t-test as PDF", MainWindow::recentDir("export"), "PDF files (*.pdf)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPdfWriter writer(path);
        writer.setPageSize(QPageSize(QPageSize::A4));
        writer.setPageOrientation(QPageLayout::Landscape);
        writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout::Millimeter);
        QPainter painter(&writer);
        double sx = static_cast<double>(writer.width())  / pw->width();
        double sy = static_cast<double>(writer.height()) / pw->height();
        double sc = std::min(sx, sy);
        painter.scale(sc, sc);
        pw->render(&painter);
        painter.end();
        QMessageBox::information(dlg, "Exported", "Saved: " + path);
    });
    auto* btn_exp_png_npy = new QPushButton("Export PNG…");
    connect(btn_exp_png_npy, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export t-test as PNG", MainWindow::recentDir("export"), "PNG images (*.png)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPixmap pix = pw->grab();
        if (!pix.save(path, "PNG"))
            QMessageBox::warning(dlg, "Export PNG", "Could not save:\n" + path);
        else
            QMessageBox::information(dlg, "Exported", "Saved: " + path);
    });

    auto* ctrl   = new QWidget(dlg);
    auto* ctrl_l = new QHBoxLayout(ctrl);
    ctrl_l->setContentsMargins(4, 2, 4, 2);
    ctrl_l->setSpacing(6);

    // Mode buttons
    auto* btn_pan_npy  = new QPushButton("Pan");
    auto* btn_meas_npy = new QPushButton("Measure");
    auto* btn_bz_npy   = new QPushButton("⬚ Zoom");
    auto* btn_crop_npy = new QPushButton("✂ Cut");
    auto* btn_rst_npy  = new QPushButton("Reset");
    btn_pan_npy->setCheckable(true);  btn_pan_npy->setChecked(true);
    btn_meas_npy->setCheckable(true);
    btn_bz_npy->setCheckable(true);
    btn_crop_npy->setCheckable(true);
    btn_pan_npy->setToolTip("Drag to pan, scroll to zoom");
    btn_meas_npy->setToolTip("Click two points to measure distance");
    btn_bz_npy->setToolTip("Drag to rubber-band zoom");
    btn_crop_npy->setToolTip("Drag to select a region to cut (exclude from export)");
    btn_rst_npy->setToolTip("Reset view");
    auto* mg_npy = new QButtonGroup(dlg);
    mg_npy->addButton(btn_pan_npy,  0);
    mg_npy->addButton(btn_meas_npy, 1);
    mg_npy->addButton(btn_bz_npy,   2);
    mg_npy->addButton(btn_crop_npy, 3);
    connect(mg_npy, QOverload<int>::of(&QButtonGroup::idClicked), dlg, [pw](int id) {
        InteractionMode m = id == 0 ? InteractionMode::Pan
                          : id == 1 ? InteractionMode::Measure
                          : id == 2 ? InteractionMode::BoxZoom
                                    : InteractionMode::CropSelect;
        pw->setMode(m);
    });
    connect(btn_rst_npy, &QPushButton::clicked, dlg, [pw]() { pw->resetView(); });

    ctrl_l->addWidget(btn_pan_npy);
    ctrl_l->addWidget(btn_meas_npy);
    ctrl_l->addWidget(btn_bz_npy);
    ctrl_l->addWidget(btn_crop_npy);
    ctrl_l->addWidget(btn_rst_npy);
    ctrl_l->addSpacing(8);
    ctrl_l->addWidget(new QLabel(QString("Samples: <b>%1</b>").arg(tstat_ptr->size())));
    {
        auto* lbl_f = qobject_cast<QLabel*>(ctrl_l->itemAt(ctrl_l->count()-1)->widget());
        if (lbl_f) lbl_f->setTextFormat(Qt::RichText);
    }
    ctrl_l->addStretch();
    auto* btn_yzi_npy = new QPushButton("↑ Amp");
    auto* btn_yzo_npy = new QPushButton("↓ Amp");
    btn_yzi_npy->setToolTip("Zoom in Y (Ctrl/Shift+scroll up)");
    btn_yzo_npy->setToolTip("Zoom out Y / shorter traces (Ctrl/Shift+scroll down)");
    connect(btn_yzi_npy, &QPushButton::clicked, dlg, [pw](){ pw->zoomInY(); });
    connect(btn_yzo_npy, &QPushButton::clicked, dlg, [pw](){ pw->zoomOutY(); });

    ctrl_l->addWidget(lbl_thr);
    ctrl_l->addWidget(spin_thr);
    ctrl_l->addWidget(chk_onesided_npy);
    ctrl_l->addWidget(btn_calc_th_npy);
    ctrl_l->addSpacing(8);
    ctrl_l->addWidget(btn_yzi_npy);
    ctrl_l->addWidget(btn_yzo_npy);
    ctrl_l->addSpacing(8);
    ctrl_l->addWidget(btn_style_npy);
    ctrl_l->addStretch();
    ctrl_l->addWidget(btn_exp_npy);
    ctrl_l->addWidget(btn_exp_pdf_npy);
    ctrl_l->addWidget(btn_exp_png_npy);

    // Trim row
    int64_t n_full_npy = static_cast<int64_t>(tstat_ptr->size());
    auto* trim_row_npy = new QWidget(dlg);
    auto* trim_l_npy   = new QHBoxLayout(trim_row_npy);
    trim_l_npy->setContentsMargins(4, 0, 4, 0);
    auto* sp_excl_start_npy = new QSpinBox; sp_excl_start_npy->setRange(0, static_cast<int>(n_full_npy / 2));
    auto* sp_excl_end_npy   = new QSpinBox; sp_excl_end_npy->setRange(0, static_cast<int>(n_full_npy / 2));
    sp_excl_start_npy->setToolTip("Samples to exclude from the start");
    sp_excl_end_npy->setToolTip("Samples to exclude from the end");
    auto* lbl_vis_npy = new QLabel; lbl_vis_npy->setTextFormat(Qt::RichText);

    auto upd_lbl_npy = [=]() {
        lbl_vis_npy->setText(QString("Exporting <b>%1</b> / %2 samples")
            .arg(current_ptr->size()).arg(n_full_npy));
    };
    // Spinbox trim: rebuild current_ptr from original
    auto skip_npy = std::make_shared<bool>(false);
    auto do_trim_npy = [=]() {
        int ts = sp_excl_start_npy->value(), te = sp_excl_end_npy->value();
        int64_t s = ts, e = std::max<int64_t>(ts + 1, n_full_npy - te);
        *current_ptr = std::vector<float>(tstat_ptr->begin() + s, tstat_ptr->begin() + e);
        *skip_npy = true; pw->clearCropRanges(); *skip_npy = false;
        pw->replaceMemTrace(0, current_ptr);
        upd_lbl_npy();
    };
    // Cut confirmed: physically splice the selected range out of current_ptr
    connect(pw, &PlotWidget::cropRangesChanged, dlg, [=]() {
        if (*skip_npy) return;
        const auto& cr = pw->cropRanges();
        if (cr.empty()) return;
        std::vector<std::pair<int64_t,int64_t>> cuts(cr.begin(), cr.end());
        std::sort(cuts.begin(), cuts.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        for (const auto& r : cuts) {
            auto s = std::max<int64_t>(0, r.first);
            auto e = std::min<int64_t>((int64_t)current_ptr->size(), r.second);
            if (s < e) current_ptr->erase(current_ptr->begin() + s, current_ptr->begin() + e);
        }
        *skip_npy = true; pw->clearCropRanges(); *skip_npy = false;
        pw->replaceMemTrace(0, current_ptr);
        upd_lbl_npy();
    });
    connect(sp_excl_start_npy, QOverload<int>::of(&QSpinBox::valueChanged), dlg, [=](int){ do_trim_npy(); });
    connect(sp_excl_end_npy,   QOverload<int>::of(&QSpinBox::valueChanged), dlg, [=](int){ do_trim_npy(); });
    upd_lbl_npy();

    trim_l_npy->addWidget(new QLabel("Exclude:"));
    trim_l_npy->addWidget(new QLabel("start"));
    trim_l_npy->addWidget(sp_excl_start_npy);
    trim_l_npy->addWidget(new QLabel("end"));
    trim_l_npy->addWidget(sp_excl_end_npy);
    trim_l_npy->addWidget(new QLabel("samples  —"));
    trim_l_npy->addWidget(lbl_vis_npy);
    trim_l_npy->addStretch();

    auto* vl = new QVBoxLayout(dlg);
    vl->setContentsMargins(4, 4, 4, 4); vl->setSpacing(4);
    vl->addWidget(ctrl);
    vl->addWidget(trim_row_npy);
    vl->addWidget(pw, 1);
    dlg->show();
}

void MainWindow::onLoadNpyHeatmap() {
    QString path = QFileDialog::getOpenFileName(
        this, "Load heatmap NPY", recentDir("npy"), "NumPy files (*.npy);;All files (*)");
    if (path.isEmpty()) return;
    updateRecentDir("npy", path);

    std::vector<float> data;
    std::vector<int64_t> shape;
    QString err;
    if (!loadNpy(path, data, shape, err)) {
        QMessageBox::critical(this, "Load failed", err); return;
    }
    if (shape.size() != 2 || shape[0] != shape[1]) {
        QMessageBox::critical(this, "Load failed",
            shape.size() != 2
                ? QString("Expected a 2-D array, got %1-D.").arg(shape.size())
                : QString("Expected a square matrix, got %1×%2.").arg(shape[0]).arg(shape[1]));
        return;
    }
    int32_t M = static_cast<int32_t>(shape[0]);

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(QString("Heatmap  M=%1 — %2").arg(M).arg(QFileInfo(path).fileName()));
    dlg->resize(820, 760);

    auto* heatmap = new HeatmapWidget(dlg);
    heatmap->setMatrix(data, M);

    auto* lbl_hover = new QLabel("Hover over matrix to inspect values");
    lbl_hover->setTextInteractionFlags(Qt::TextSelectableByMouse);

    auto* lbl_vmin = new QLabel("Color min:");
    auto* lbl_vmax = new QLabel("Color max:");
    auto* sp_vmin  = new QDoubleSpinBox;
    auto* sp_vmax  = new QDoubleSpinBox;
    sp_vmin->setRange(-1e9, 1e9); sp_vmin->setDecimals(4); sp_vmin->setSingleStep(0.1);
    sp_vmax->setRange(-1e9, 1e9); sp_vmax->setDecimals(4); sp_vmax->setSingleStep(0.1);

    {
        float dmin = 1e38f, dmax = -1e38f;
        for (float v : data) { dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
        float abs_max = std::max(std::abs(dmin), std::abs(dmax));
        sp_vmin->setValue(static_cast<double>(-abs_max));
        sp_vmax->setValue(static_cast<double>( abs_max));
        heatmap->setColorRange(-abs_max, abs_max);
    }

    connect(sp_vmin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setColorRange(static_cast<float>(v), static_cast<float>(sp_vmax->value()));
    });
    connect(sp_vmax, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setColorRange(static_cast<float>(sp_vmin->value()), static_cast<float>(v));
    });

    connect(heatmap, &HeatmapWidget::hoverInfo, dlg, [lbl_hover](int s1, int s2, float val) {
        lbl_hover->setText(
            QString("C[%1, %2] = %3").arg(s1).arg(s2)
                .arg(static_cast<double>(val), 0, 'g', 6));
    });

    // Processing controls
    auto* lbl_scheme  = new QLabel("Color scheme:");
    auto* combo_scheme = new QComboBox;
    combo_scheme->addItems({"RdBu", "Grayscale", "Hot", "Viridis", "Plasma"});
    connect(combo_scheme, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int idx) {
        heatmap->setColorScheme(static_cast<ColorScheme>(idx));
    });

    auto* lbl_sigma   = new QLabel("Gaussian σ:");
    auto* sp_sigma    = new QDoubleSpinBox;
    sp_sigma->setRange(0.0, 50.0); sp_sigma->setDecimals(1); sp_sigma->setSingleStep(0.5);
    sp_sigma->setValue(0.0); sp_sigma->setSpecialValueText("off");
    connect(sp_sigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setGaussianSigma(static_cast<float>(v));
    });

    auto* chk_abs    = new QCheckBox("Abs value");
    connect(chk_abs, &QCheckBox::toggled, [=](bool on) {
        heatmap->setAbsValue(on);
        // Abs collapses all values to [0, vmax] — snap vmin to 0 so the
        // colour range is correct; restore symmetric range when unchecked.
        if (on) {
            sp_vmin->setValue(0.0);
            heatmap->setColorRange(0.0f, static_cast<float>(sp_vmax->value()));
        } else {
            double vm = sp_vmax->value();
            sp_vmin->setValue(-vm);
            heatmap->setColorRange(static_cast<float>(-vm), static_cast<float>(vm));
        }
    });

    auto* lbl_gamma  = new QLabel("Power γ:");
    auto* sp_gamma   = new QDoubleSpinBox;
    // min=1.0 so setSpecialValueText shows "off" at the off-state (gamma=1)
    sp_gamma->setRange(1.0, 10.0); sp_gamma->setDecimals(2); sp_gamma->setSingleStep(0.1);
    sp_gamma->setValue(1.0); sp_gamma->setSpecialValueText("off");
    connect(sp_gamma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setPowerGamma(static_cast<float>(v));
    });

    auto* chk_thresh  = new QCheckBox("Binary threshold |v|≥");
    auto* sp_thresh   = new QDoubleSpinBox;
    sp_thresh->setRange(0.0, 1e9); sp_thresh->setDecimals(4);
    sp_thresh->setSingleStep(0.05); sp_thresh->setValue(0.5);
    sp_thresh->setEnabled(false);
    connect(chk_thresh, &QCheckBox::toggled, [=](bool on) {
        sp_thresh->setEnabled(on);
        heatmap->setBinaryThreshold(on, static_cast<float>(sp_thresh->value()));
    });
    connect(sp_thresh, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        if (chk_thresh->isChecked())
            heatmap->setBinaryThreshold(true, static_cast<float>(v));
    });

    auto* btn_reset_view = new QPushButton("Reset View");
    connect(btn_reset_view, &QPushButton::clicked, heatmap, &HeatmapWidget::resetView);

    auto* btn_autoclip = new QPushButton("Auto-clip 98%");
    connect(btn_autoclip, &QPushButton::clicked, dlg, [=]() {
        float cmin, cmax;
        heatmap->computeClipRange(0.98f, cmin, cmax);
        sp_vmin->setValue(static_cast<double>(cmin));
        sp_vmax->setValue(static_cast<double>(cmax));
        heatmap->setColorRange(cmin, cmax);
    });

    // Keep a shared_ptr so the export lambda can safely capture the data
    auto data_ptr = std::make_shared<std::vector<float>>(std::move(data));

    auto* btn_exp_png = new QPushButton("Export PNG…");
    connect(btn_exp_png, &QPushButton::clicked, dlg, [=]() {
        QString p = QFileDialog::getSaveFileName(dlg, "Export as PNG", MainWindow::recentDir("export"), "PNG images (*.png)");
        if (p.isEmpty()) return;
        MainWindow::updateRecentDir("export", p);
        if (!heatmap->exportPng(p))
            QMessageBox::critical(dlg, "Export failed", "Could not write:\n" + p);
        else
            QMessageBox::information(dlg, "Saved", "Saved: " + p);
    });

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, dlg, [=]() {
        QString p = QFileDialog::getSaveFileName(dlg, "Export as NumPy", MainWindow::recentDir("npy"), "NumPy files (*.npy)");
        if (p.isEmpty()) return;
        MainWindow::updateRecentDir("npy", p);
        FILE* fp = std::fopen(p.toLocal8Bit().constData(), "wb");
        if (!fp) { QMessageBox::critical(dlg, "Export failed", "Cannot create:\n" + p); return; }
        const uint8_t magic[] = {0x93,'N','U','M','P','Y',0x01,0x00};
        std::fwrite(magic, 1, 8, fp);
        std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                           std::to_string(M) + ", " + std::to_string(M) + "), }";
        size_t content_len = dict.size() + 1;
        size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
        dict.resize(header_len - 1, ' '); dict += '\n';
        uint16_t hl = static_cast<uint16_t>(header_len);
        uint8_t hl_b[2] = {uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
        std::fwrite(hl_b, 1, 2, fp);
        std::fwrite(dict.c_str(), 1, dict.size(), fp);
        std::fwrite(data_ptr->data(), sizeof(float),
                    static_cast<size_t>(M) * static_cast<size_t>(M), fp);
        std::fclose(fp);
        QMessageBox::information(dlg, "Saved", "Saved: " + p);
    });

    auto* row1   = new QWidget(dlg);
    auto* row1_l = new QHBoxLayout(row1);
    row1_l->setContentsMargins(4, 2, 4, 2);
    row1_l->addWidget(lbl_hover, 1);
    row1_l->addWidget(lbl_vmin);
    row1_l->addWidget(sp_vmin);
    row1_l->addWidget(lbl_vmax);
    row1_l->addWidget(sp_vmax);
    row1_l->addWidget(btn_autoclip);
    row1_l->addWidget(btn_reset_view);
    row1_l->addWidget(btn_exp_png);
    row1_l->addWidget(btn_exp_npy);

    auto* row2   = new QWidget(dlg);
    auto* row2_l = new QHBoxLayout(row2);
    row2_l->setContentsMargins(4, 2, 4, 2);
    row2_l->addWidget(lbl_scheme);
    row2_l->addWidget(combo_scheme);
    row2_l->addSpacing(12);
    row2_l->addWidget(lbl_sigma);
    row2_l->addWidget(sp_sigma);
    row2_l->addSpacing(8);
    row2_l->addWidget(chk_abs);
    row2_l->addSpacing(8);
    row2_l->addWidget(lbl_gamma);
    row2_l->addWidget(sp_gamma);
    row2_l->addSpacing(8);
    row2_l->addWidget(chk_thresh);
    row2_l->addWidget(sp_thresh);
    row2_l->addStretch();

    auto* vl = new QVBoxLayout(dlg);
    vl->setContentsMargins(4, 4, 4, 4); vl->setSpacing(4);
    vl->addWidget(row1);
    vl->addWidget(row2);
    vl->addWidget(heatmap, 1);
    dlg->show();
}

// ---------------------------------------------------------------------------
// Open NPY / NPZ file as a trace set (loads into the main viewer)
// ---------------------------------------------------------------------------
void MainWindow::onOpenNpyTraces() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open NPY/NPZ as traces",
        recentDir("npy"), "NumPy files (*.npy *.npz);;All files (*)");
    if (path.isEmpty()) return;
    updateRecentDir("npy", path);

    std::vector<float> traces_flat;
    std::vector<float> data_flat;
    int64_t n_traces = 0, n_samples = 0;
    int64_t data_cols = 0;
    QString err;

    struct NamedCol { std::string name; int elem_size; std::vector<uint8_t> payload; };
    std::vector<NamedCol> named_cols;

    if (path.endsWith(".npz", Qt::CaseInsensitive)) {
        std::map<std::string, std::vector<uint8_t>> entries;
        if (!loadNpz(path, entries, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }

        // --- Find the traces entry (2-D float array) ---
        auto it = entries.find("traces.npy");
        if (it == entries.end()) it = entries.find("traces");
        if (it == entries.end()) {
            for (auto& kv : entries) {
                std::vector<int64_t> sh; std::vector<float> tmp;
                if (parseNpyBytes(kv.second, tmp, sh, err) && sh.size() == 2) {
                    it = entries.find(kv.first); break;
                }
            }
        }
        if (it == entries.end()) {
            QMessageBox::critical(this, "Load failed",
                "No 2-D traces array found. Expected an entry named 'traces' or 'traces.npy'.");
            return;
        }
        std::vector<int64_t> sh;
        if (!parseNpyBytes(it->second, traces_flat, sh, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }
        if (sh.size() != 2) {
            QMessageBox::critical(this, "Load failed",
                QString("Traces entry is %1-D; expected 2-D.").arg(sh.size()));
            return;
        }
        n_traces = sh[0]; n_samples = sh[1];
        std::string traces_key = it->first;

        // --- Also check for legacy 2-D data array ---
        for (const char* dname : {"data.npy","data","labels.npy","labels","textin.npy","textin","plaintext.npy","plaintext"}) {
            auto di = entries.find(dname);
            if (di == entries.end() || di->first == traces_key) continue;
            std::vector<int64_t> dsh;
            if (parseNpyBytes(di->second, data_flat, dsh, err) && dsh.size() == 2 && dsh[0] == n_traces) {
                data_cols = dsh[1];
                break;
            }
        }

        // --- Scan all remaining entries for 1-D arrays matching n_traces ---
        static const char* legacy_names[] = {
            "data.npy","data","labels.npy","labels",
            "textin.npy","textin","plaintext.npy","plaintext", nullptr
        };
        for (auto& kv : entries) {
            if (kv.first == traces_key) continue;
            // skip if already picked up as 2-D data
            bool skip = false;
            if (data_cols > 0) {
                for (int li = 0; legacy_names[li]; li++)
                    if (kv.first == legacy_names[li]) { skip = true; break; }
            }
            if (skip) continue;

            std::vector<int64_t> csh; int esz; std::vector<uint8_t> cpayload; QString cerr;
            if (!parseNpyRaw(kv.second, csh, esz, cpayload, cerr)) continue;
            if (csh.size() != 1 || csh[0] != n_traces) continue;
            if (static_cast<int64_t>(cpayload.size()) < n_traces * esz) continue;

            // strip ".npy" suffix from name
            std::string colname = kv.first;
            if (colname.size() > 4 && colname.substr(colname.size()-4) == ".npy")
                colname = colname.substr(0, colname.size()-4);

            named_cols.push_back({colname, esz, std::move(cpayload)});
        }
    } else {
        // Plain .npy: memory-map directly rather than reading the whole file into
        // RAM (a copy here plus the copy TrsFile::openFromArray used to make meant
        // multi-GB files needed ~2x their size in RAM and could get OOM-killed).
        auto f = std::make_unique<TrsFile>();
        std::string npy_err;
        if (!f->openNpy(path.toStdString(), npy_err)) {
            QMessageBox::critical(this, "Load failed", QString::fromStdString(npy_err)); return;
        }
        prefetchWithProgress(f.get(), "Reading " + QFileInfo(path).fileName() + "…");

        Dataset ds;
        ds.file         = std::move(f);
        ds.display_name = QFileInfo(path).fileName();
        ds.plot_widget  = createPlotWidgetForTab();
        datasets_.push_back(std::move(ds));
        active_idx_ = static_cast<int>(datasets_.size()) - 1;

        tab_bar_->addTab(datasets_[static_cast<size_t>(active_idx_)].display_name);
        tab_bar_->setCurrentIndex(active_idx_);

        rebuildTransformList();
        plotWidget()->setTransforms(activeDs().pipeline);

        int n = activeDs().file->header().num_traces;
        spin_first_->setMaximum(std::max(0, n - 1));
        spin_first_->setValue(0);
        spin_count_->setMaximum(n);
        spin_count_->setValue(1);

        updateFileInfo();
        onApplyTraces();
        updateViewLayout();
        return;
    }

    if (n_traces < 1 || n_samples < 1) {
        QMessageBox::critical(this, "Load failed", "Empty array."); return;
    }

    // Build the in-memory TrsFile
    auto f = std::make_unique<TrsFile>();
    std::vector<uint8_t> data_bytes;
    int16_t data_length = 0;
    std::map<std::string, TrsTraceParam> param_map;

    if (!named_cols.empty()) {
        // Pack named 1-D columns as raw bytes; build param_map
        uint16_t offset = 0;
        for (auto& col : named_cols) {
            if (offset + col.elem_size > 32767) break; // safety cap
            TrsTraceParam p;
            p.offset = offset;
            p.length = static_cast<uint16_t>(col.elem_size);
            // type field: 1=u8/i8, 2=i16, 4=i32, 8=i64
            switch (col.elem_size) {
                case 1: p.type = 1; break;
                case 2: p.type = 2; break;
                case 4: p.type = 4; break;
                case 8: p.type = 8; break;
                default: p.type = 0; break;
            }
            param_map[col.name] = p;
            offset += static_cast<uint16_t>(col.elem_size);
        }
        data_length = static_cast<int16_t>(offset);
        data_bytes.resize(static_cast<size_t>(n_traces) * static_cast<size_t>(data_length), 0);
        for (auto& col : named_cols) {
            auto pit = param_map.find(col.name);
            if (pit == param_map.end()) continue;
            uint16_t off = pit->second.offset;
            int esz = col.elem_size;
            for (int64_t ti = 0; ti < n_traces; ti++) {
                const uint8_t* src = col.payload.data() + static_cast<size_t>(ti) * esz;
                uint8_t* dst = data_bytes.data() + static_cast<size_t>(ti) * data_length + off;
                std::memcpy(dst, src, esz);
            }
        }
    } else if (data_cols > 0) {
        data_length = static_cast<int16_t>(std::min(data_cols, (int64_t)32767));
        data_bytes.resize(static_cast<size_t>(n_traces) * static_cast<size_t>(data_length));
        for (int64_t ti = 0; ti < n_traces; ti++) {
            for (int16_t bi = 0; bi < data_length; bi++) {
                float v = data_flat[static_cast<size_t>(ti) * static_cast<size_t>(data_cols)
                                  + static_cast<size_t>(bi)];
                data_bytes[static_cast<size_t>(ti) * static_cast<size_t>(data_length)
                         + static_cast<size_t>(bi)] = static_cast<uint8_t>(
                    static_cast<int>(v) & 0xFF);
            }
        }
    }
    f->openFromArray(traces_flat.data(),
                     static_cast<int32_t>(n_traces),
                     static_cast<int32_t>(n_samples),
                     path.toStdString(),
                     data_bytes.empty() ? nullptr : data_bytes.data(),
                     data_length);
    if (!param_map.empty())
        f->setParamMap(param_map);

    Dataset ds;
    ds.file         = std::move(f);
    ds.display_name = QFileInfo(path).fileName();
    ds.plot_widget  = createPlotWidgetForTab();
    datasets_.push_back(std::move(ds));
    active_idx_ = static_cast<int>(datasets_.size()) - 1;

    tab_bar_->addTab(datasets_[static_cast<size_t>(active_idx_)].display_name);
    tab_bar_->setCurrentIndex(active_idx_);

    rebuildTransformList();
    plotWidget()->setTransforms(activeDs().pipeline);

    int n = activeDs().file->header().num_traces;
    spin_first_->setMaximum(std::max(0, n - 1));
    spin_first_->setValue(0);
    spin_count_->setMaximum(n);
    spin_count_->setValue(1);

    updateFileInfo();
    onApplyTraces();
    updateViewLayout();
}

// ---------------------------------------------------------------------------
// Export traces → 2-D NPY matrix (n_traces × n_samples, pipeline + alignment
// shifts applied, traces marked kAlignDiscardShift dropped). Non-interactive
// core shared by onExportNpy()'s dialog and the Chain "Export" step.
// Returns false and sets err_out on failure. *n_written_out (if non-null)
// receives the actual number of traces written on success.
// ---------------------------------------------------------------------------
static bool exportTracesToNpy(
    const QString& out_path,
    TrsFile* src,
    int32_t first_trace, int32_t count,
    const std::vector<int32_t>& shifts,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    QWidget* msg_parent,
    QProgressDialog* progress,
    QString& err_out,
    int32_t* n_written_out = nullptr)
{
    const TrsHeader& h = src->header();
    int32_t n_req = std::min(count, h.num_traces - first_trace);
    if (n_req <= 0) { err_out = "No traces to export."; return false; }

    std::vector<int32_t> keep;
    keep.reserve(static_cast<size_t>(n_req));
    for (int32_t i = 0; i < n_req; i++) {
        int32_t shift = (i < static_cast<int32_t>(shifts.size())) ? shifts[i] : 0;
        if (shift == kAlignDiscardShift) continue;
        keep.push_back(i);
    }
    if (keep.empty()) { err_out = "No traces to export (all discarded by alignment)."; return false; }
    int32_t n_out = static_cast<int32_t>(keep.size());
    if (n_written_out) *n_written_out = n_out;

    int64_t out_samples = h.num_samples;
    for (const auto& t : pipeline) out_samples = t->transformedCount(out_samples);

    // Allocate full matrix (may be large)
    const size_t total = static_cast<size_t>(n_out) * static_cast<size_t>(out_samples);
    double mem_mb = total * sizeof(float) / (1024.0 * 1024.0);
    if (mem_mb > 2048.0) {
        if (QMessageBox::warning(msg_parent, "Memory warning",
                QString("Output matrix will require ~%1 GB.\nContinue?").arg(mem_mb/1024.0, 0,'f',1),
                QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes) {
            err_out = "Export cancelled.";
            return false;
        }
    }

    std::vector<float> matrix(total);
    // buf must hold both the raw read (h.num_samples) AND the post-pipeline output
    // (out_samples); expanding transforms like STFT can produce more samples than input.
    std::vector<float> buf(static_cast<size_t>(std::max((int64_t)h.num_samples, out_samples)));

    for (int32_t ki = 0; ki < n_out; ki++) {
        if (progress) {
            if (progress->wasCanceled()) { err_out = "Export cancelled."; return false; }
            progress->setValue(ki);
            QApplication::processEvents();
        }
        int32_t i     = keep[static_cast<size_t>(ki)];
        int32_t src_idx = first_trace + i;
        int32_t shift = shifts.empty() ? 0 : shifts[static_cast<size_t>(i)];
        readAlignedWindow(src, src_idx, 0, h.num_samples, shift, h.num_samples, buf.data());
        for (const auto& t : pipeline) t->reset();
        int64_t n_samp = h.num_samples;
        for (const auto& t : pipeline) n_samp = t->apply(buf.data(), n_samp, 0);
        std::copy(buf.begin(), buf.begin() + static_cast<size_t>(n_samp),
                  matrix.begin() + static_cast<ptrdiff_t>(ki) * static_cast<ptrdiff_t>(out_samples));
    }
    if (progress) progress->setValue(n_out);

    return saveNpy2D(out_path, matrix.data(), n_out, out_samples, err_out);
}

// ---------------------------------------------------------------------------
// Export traces → NPZ (traces.npy + data.npy if the source has data bytes),
// same shift/discard handling as exportTracesToNpy(). Non-interactive core
// shared by onExportNpz()'s dialog and the Chain "Export" step.
// ---------------------------------------------------------------------------
static bool exportTracesToNpz(
    const QString& out_path,
    TrsFile* src,
    int32_t first_trace, int32_t count,
    const std::vector<int32_t>& shifts,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    QProgressDialog* progress,
    QString& err_out,
    int32_t* n_written_out = nullptr)
{
    const TrsHeader& h = src->header();
    int32_t n_req = std::min(count, h.num_traces - first_trace);
    if (n_req <= 0) { err_out = "No traces to export."; return false; }

    std::vector<int32_t> keep;
    keep.reserve(static_cast<size_t>(n_req));
    for (int32_t i = 0; i < n_req; i++) {
        int32_t shift = (i < static_cast<int32_t>(shifts.size())) ? shifts[i] : 0;
        if (shift == kAlignDiscardShift) continue;
        keep.push_back(i);
    }
    if (keep.empty()) { err_out = "No traces to export (all discarded by alignment)."; return false; }
    int32_t n_out = static_cast<int32_t>(keep.size());
    if (n_written_out) *n_written_out = n_out;

    int64_t out_samples = h.num_samples;
    for (const auto& t : pipeline) out_samples = t->transformedCount(out_samples);

    std::vector<float> traces(static_cast<size_t>(n_out) * static_cast<size_t>(out_samples));
    // buf must hold both the raw read (h.num_samples) AND the post-pipeline output
    // (out_samples); expanding transforms like STFT can produce more samples than input.
    std::vector<float> buf(static_cast<size_t>(std::max((int64_t)h.num_samples, out_samples)));

    for (int32_t ki = 0; ki < n_out; ki++) {
        if (progress) {
            if (progress->wasCanceled()) { err_out = "Export cancelled."; return false; }
            progress->setValue(ki);
            QApplication::processEvents();
        }
        int32_t i     = keep[static_cast<size_t>(ki)];
        int32_t src_idx = first_trace + i;
        int32_t shift = shifts.empty() ? 0 : shifts[static_cast<size_t>(i)];
        readAlignedWindow(src, src_idx, 0, h.num_samples, shift, h.num_samples, buf.data());
        for (const auto& t : pipeline) t->reset();
        int64_t n_samp = h.num_samples;
        for (const auto& t : pipeline) n_samp = t->apply(buf.data(), n_samp, 0);
        std::copy(buf.begin(), buf.begin() + static_cast<size_t>(n_samp),
                  traces.begin() + static_cast<ptrdiff_t>(ki) * static_cast<ptrdiff_t>(out_samples));
    }
    if (progress) progress->setValue(n_out);

    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
    entries.push_back({"traces.npy",
        buildNpyBytes("<f4", n_out, out_samples, traces.data(),
                      static_cast<size_t>(n_out) * static_cast<size_t>(out_samples) * sizeof(float))});

    if (h.data_length > 0) {
        // Build data matrix (n_traces × data_length, uint8)
        std::vector<uint8_t> data_mat(static_cast<size_t>(n_out) * static_cast<size_t>(h.data_length));
        for (int32_t ki = 0; ki < n_out; ki++) {
            auto db = src->readData(first_trace + keep[static_cast<size_t>(ki)]);
            size_t copy_n = std::min(static_cast<size_t>(h.data_length), db.size());
            std::copy(db.begin(), db.begin() + static_cast<ptrdiff_t>(copy_n),
                      data_mat.begin() + static_cast<ptrdiff_t>(ki) * h.data_length);
        }
        entries.push_back({"data.npy",
            buildNpyBytes("|u1", n_out, h.data_length, data_mat.data(), data_mat.size())});
    }

    return saveNpz(out_path, entries, err_out);
}

// ---------------------------------------------------------------------------
// Export traces → 2-D NPY (n_traces × n_samples, pipeline applied)
// ---------------------------------------------------------------------------
void MainWindow::onExportNpy() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Export NPY", "No file loaded."); return;
    }
    const TrsHeader& h = activeDs().file->header();

    // Config dialog: same range picker as TRS export
    QDialog cfg(this);
    cfg.setWindowTitle("Export as NPY — configuration");
    auto* fl     = new QFormLayout(&cfg);
    auto* sp_first = new QSpinBox; sp_first->setRange(0, std::max(0, h.num_traces-1)); sp_first->setValue(0);
    auto* sp_count = new QSpinBox; sp_count->setRange(1, h.num_traces); sp_count->setValue(h.num_traces);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);

    const bool has_alignment = (activeDs().align_n_samples > 0);
    const int  n_discarded = static_cast<int>(std::count(
        activeDs().align_shifts.begin(), activeDs().align_shifts.end(), kAlignDiscardShift));
    QCheckBox* chk_align = addAlignmentGroup(fl, has_alignment,
        activeDs().align_first_trace, static_cast<int>(activeDs().align_shifts.size()),
        n_discarded, sp_first, sp_count);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first = static_cast<int32_t>(sp_first->value());
    int32_t count = static_cast<int32_t>(sp_count->value());
    const bool use_alignment = chk_align->isChecked();
    const std::vector<int32_t> shifts = use_alignment ? activeDs().align_shifts
                                                        : std::vector<int32_t>{};

    QString path = QFileDialog::getSaveFileName(
        this, "Export traces as NPY", recentDir("npy"), "NumPy files (*.npy)");
    if (path.isEmpty()) return;
    updateRecentDir("npy", path);

    QProgressDialog prog("Exporting…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    QString err;
    int32_t n_written = 0;
    bool ok = exportTracesToNpy(path, activeDs().file.get(), first, count, shifts,
                                 activeDs().pipeline, this, &prog, err, &n_written);
    if (!ok) {
        if (err != "Export cancelled.") QMessageBox::critical(this, "Export failed", err);
    } else {
        int64_t out_samples = h.num_samples;
        for (const auto& t : activeDs().pipeline) out_samples = t->transformedCount(out_samples);
        QMessageBox::information(this, "Exported",
            QString("Saved %1 × %2 matrix to:\n%3").arg(n_written).arg(out_samples).arg(path));
    }
}

// ---------------------------------------------------------------------------
// Export traces → NPZ (traces.npy + data.npy if TRS has data bytes)
// ---------------------------------------------------------------------------
void MainWindow::onExportNpz() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Export NPZ", "No file loaded."); return;
    }
    const TrsHeader& h = activeDs().file->header();

    QDialog cfg(this);
    cfg.setWindowTitle("Export as NPZ — configuration");
    auto* fl       = new QFormLayout(&cfg);
    auto* sp_first = new QSpinBox; sp_first->setRange(0, std::max(0, h.num_traces-1)); sp_first->setValue(0);
    auto* sp_count = new QSpinBox; sp_count->setRange(1, h.num_traces); sp_count->setValue(h.num_traces);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);

    const bool has_alignment = (activeDs().align_n_samples > 0);
    const int  n_discarded = static_cast<int>(std::count(
        activeDs().align_shifts.begin(), activeDs().align_shifts.end(), kAlignDiscardShift));
    QCheckBox* chk_align = addAlignmentGroup(fl, has_alignment,
        activeDs().align_first_trace, static_cast<int>(activeDs().align_shifts.size()),
        n_discarded, sp_first, sp_count);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    if (h.data_length > 0)
        fl->addRow(new QLabel(QString("<i>Will also export data.npy (%1 bytes/trace)</i>")
                                  .arg(h.data_length)));
    fl->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first = static_cast<int32_t>(sp_first->value());
    int32_t count = static_cast<int32_t>(sp_count->value());
    const bool use_alignment = chk_align->isChecked();
    const std::vector<int32_t> shifts = use_alignment ? activeDs().align_shifts
                                                        : std::vector<int32_t>{};

    QString path = QFileDialog::getSaveFileName(
        this, "Export traces as NPZ", recentDir("npy"), "NumPy archives (*.npz)");
    if (path.isEmpty()) return;
    updateRecentDir("npy", path);

    QProgressDialog prog("Exporting…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    QString err;
    int32_t n_written = 0;
    bool ok = exportTracesToNpz(path, activeDs().file.get(), first, count, shifts,
                                 activeDs().pipeline, &prog, err, &n_written);
    if (!ok) {
        if (err != "Export cancelled.") QMessageBox::critical(this, "Export failed", err);
    } else {
        int64_t out_samples = h.num_samples;
        for (const auto& t : activeDs().pipeline) out_samples = t->transformedCount(out_samples);
        QMessageBox::information(this, "Exported",
            QString("Saved %1 × %2 traces%3 to:\n%4")
                .arg(n_written).arg(out_samples)
                .arg(h.data_length > 0 ? " + data" : "")
                .arg(path));
    }
}

// ---------------------------------------------------------------------------
// Parallelised Welch t-test accumulation + compute — shared by
// showTTestDialog() and the Chain "Run T-test" step. shifts empty => no
// alignment; shift == kAlignDiscardShift excludes that trace. Returns false
// on failure: err == "Cancelled." for a user cancel or a declined memory
// warning (not worth reporting as an error), otherwise a real failure
// message from TTestAccumulator::compute().
// ---------------------------------------------------------------------------
bool MainWindow::computeTTest(int32_t eff_first, int32_t eff_count,
                               int64_t eff_first_sample, int64_t eff_n_samples,
                               int32_t byte_idx, const std::vector<int32_t>& shifts, bool abs_value,
                               QWidget* msg_parent, std::shared_ptr<TTestAccumulator>& acc_out,
                               std::vector<float>& tstat_out, int64_t& n0_out, int64_t& n1_out,
                               QString& err)
{
    const TrsHeader& h = activeDs().file->header();
    const std::vector<int32_t>& use_shifts = shifts;

    // Effective raw sample count for the window
    const int64_t raw_ns = (eff_n_samples == 0)
        ? (h.num_samples - eff_first_sample)
        : std::min<int64_t>(eff_n_samples, h.num_samples - eff_first_sample);

    // Effective sample count after pipeline
    int64_t effective_samples = raw_ns;
    for (const auto& t : activeDs().pipeline)
        effective_samples = t->transformedCount(effective_samples);

    // Accumulation is parallelised across traces, one TTestAccumulator + one
    // cloned pipeline per worker thread (merged together at the end) — the
    // shared activeDs().pipeline transforms aren't safe to call apply() on
    // concurrently since they may cache internal state (e.g. FFT plans).
    // Capped at 8 threads since each accumulator costs 4×effective_samples
    // doubles; see the memory estimate below.
#ifdef _OPENMP
    const int n_threads = std::clamp(omp_get_max_threads(), 1, 8);
#else
    const int n_threads = 1;
#endif

    // Memory estimate warning (one accumulator per worker thread)
    int64_t mem_bytes = effective_samples * 4LL * static_cast<int64_t>(sizeof(double))
                      * n_threads;
    if (mem_bytes > 2LL * 1024 * 1024 * 1024) {
        if (QMessageBox::warning(msg_parent, "Memory warning",
                QString("Accumulators will require ~%1 GB across %2 worker thread(s).\nContinue?")
                    .arg(double(mem_bytes) / (1024.0*1024*1024), 0, 'f', 1).arg(n_threads),
                QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes) {
            err = "Cancelled.";
            return false;
        }
    }

    // --- Accumulation ---
    auto acc_ptr = std::make_shared<TTestAccumulator>(static_cast<int32_t>(effective_samples));

    QProgressDialog prog("Accumulating traces…", "Cancel", 0, eff_count, msg_parent);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    std::vector<TTestAccumulator> thread_accs(
        static_cast<size_t>(n_threads), TTestAccumulator(static_cast<int32_t>(effective_samples)));
    std::vector<std::vector<std::shared_ptr<ITransform>>> thread_pipelines(
        static_cast<size_t>(n_threads));
    for (auto& tp : thread_pipelines)
        for (const auto& t : activeDs().pipeline) tp.push_back(t->clone());
    std::vector<std::vector<float>> thread_bufs(
        static_cast<size_t>(n_threads),
        std::vector<float>(static_cast<size_t>(std::max(raw_ns, effective_samples))));
    std::vector<int32_t> thread_skipped(static_cast<size_t>(n_threads), 0);
    std::vector<int32_t> thread_discarded(static_cast<size_t>(n_threads), 0);

    // Process in batches so the GUI thread can update the progress bar and
    // check for cancellation between parallel regions (Qt calls are only
    // ever made from the main thread, never from inside the omp for below).
    // Capped at 256 for parallel efficiency, but also capped so there are at
    // least ~20 progress updates regardless of trace count — otherwise a run
    // with fewer traces than the batch size reports progress exactly once,
    // at the very end, which looks indistinguishable from a hang.
    const int32_t kBatchSize = std::clamp(eff_count / 20, 1, 256);
    bool cancelled = false;

    for (int32_t batch_start = 0; batch_start < eff_count && !cancelled; batch_start += kBatchSize) {
        int32_t batch_end = std::min(batch_start + kBatchSize, eff_count);

        #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
        for (int32_t ti = batch_start; ti < batch_end; ti++) {
#ifdef _OPENMP
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t tid = 0;
#endif
            TTestAccumulator&                        acc_local  = thread_accs[tid];
            std::vector<std::shared_ptr<ITransform>>& pipeline_local = thread_pipelines[tid];
            std::vector<float>&                      trace_buf  = thread_bufs[tid];

            int32_t src_idx = eff_first + ti;
            auto data_bytes = activeDs().file->readData(src_idx);
            if (byte_idx >= static_cast<int32_t>(data_bytes.size())) { thread_skipped[tid]++; continue; }
            int group = (data_bytes[byte_idx] != 0) ? 1 : 0;

            // Read window with per-trace shift, zero-pad out of bounds
            int32_t shift = (ti < static_cast<int32_t>(use_shifts.size())) ? use_shifts[ti] : 0;
            if (shift == kAlignDiscardShift) { thread_discarded[tid]++; continue; }
            const int64_t adj_start = eff_first_sample + shift;
            std::fill(trace_buf.begin(), trace_buf.end(), 0.0f);
            if (adj_start < h.num_samples && adj_start + raw_ns > 0) {
                int64_t src_start = std::max<int64_t>(0, adj_start);
                int64_t src_end   = std::min<int64_t>(h.num_samples, adj_start + raw_ns);
                int64_t dst_off   = src_start - adj_start;
                int64_t got = activeDs().file->readSamples(src_idx, src_start, src_end - src_start,
                                                      trace_buf.data() + dst_off);
                if (got <= 0) { thread_skipped[tid]++; continue; }
            }
            for (const auto& t : pipeline_local) t->reset();
            int64_t n_out = raw_ns;
            for (const auto& t : pipeline_local)
                n_out = t->apply(trace_buf.data(), n_out, 0);
            acc_local.addTrace(group, trace_buf.data(), static_cast<int32_t>(n_out));
        }

        prog.setLabelText(QString("Accumulating trace %1 / %2…").arg(batch_end).arg(eff_count));
        prog.setValue(batch_end);
        QApplication::processEvents();
        if (prog.wasCanceled()) cancelled = true;
    }
    if (cancelled) { err = "Cancelled."; return false; }

    TTestAccumulator& acc = *acc_ptr;
    int32_t skipped = 0, discarded_by_align = 0;
    for (int t = 0; t < n_threads; t++) {
        acc.mergeFrom(thread_accs[static_cast<size_t>(t)]);
        skipped            += thread_skipped[static_cast<size_t>(t)];
        discarded_by_align += thread_discarded[static_cast<size_t>(t)];
    }
    prog.setValue(eff_count);

    if (skipped > 0 || discarded_by_align > 0) {
        QStringList msg;
        if (skipped > 0)
            msg << QString("%1 traces skipped (data byte out of range).").arg(skipped);
        if (discarded_by_align > 0)
            msg << QString("%1 traces excluded (below alignment correlation threshold).")
                       .arg(discarded_by_align);
        QMessageBox::warning(msg_parent, "T-test", msg.join("\n"));
    }

    // --- Compute ---
    std::vector<float> tstat;
    std::string cerr;
    if (!acc.compute(tstat, cerr)) {
        err = QString::fromStdString(cerr);
        return false;
    }
    if (abs_value)
        for (float& v : tstat) v = std::fabs(v);

    acc_out   = acc_ptr;
    tstat_out = std::move(tstat);
    n0_out    = acc.countGroup(0);
    n1_out    = acc.countGroup(1);
    return true;
}

void MainWindow::onRunTTest() {
    showTTestDialog(nullptr);
}

void MainWindow::showTTestDialog(std::function<void(const ChainStep&)> onAddToChain)
{
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "T-test", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();
    if (h.data_length <= 0) {
        QMessageBox::critical(this, "T-test",
            "This TRS file has no per-trace data bytes.\n"
            "Group assignment requires at least 1 data byte per trace.");
        return;
    }

    // Check whether the file has a named "ttest" parameter in its parameter map
    bool have_ttest_param = h.param_map.count("ttest") > 0;
    int32_t auto_byte_idx = have_ttest_param
        ? static_cast<int32_t>(h.param_map.at("ttest").offset)
        : 0;

    // --- Configuration dialog ---
    int n_total = h.num_traces;
    QDialog cfg(this);
    cfg.setWindowTitle("Welch t-test — configuration");
    auto* fl        = new QFormLayout(&cfg);
    auto* sp_first  = new QSpinBox; sp_first->setRange(0, std::max(0, n_total-1)); sp_first->setValue(0);
    auto* sp_count  = new QSpinBox; sp_count->setRange(2, n_total); sp_count->setValue(n_total);
    auto* sp_s_first = new QSpinBox; sp_s_first->setRange(0, std::max(0, (int)h.num_samples - 1)); sp_s_first->setValue(0);
    auto* sp_s_count = new QSpinBox; sp_s_count->setRange(0, (int)h.num_samples); sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);
    fl->addRow("First sample:", sp_s_first);
    fl->addRow("Sample count (0=all):", sp_s_count);

    // Alignment group
    const bool has_alignment = (activeDs().align_n_samples > 0);
    const int  n_align_discarded = static_cast<int>(std::count(
        activeDs().align_shifts.begin(), activeDs().align_shifts.end(), kAlignDiscardShift));
    auto* grp_align  = new QGroupBox("Alignment");
    auto* fl_align   = new QFormLayout(grp_align);
    auto* chk_shifts = new QCheckBox("Apply last alignment shifts");
    chk_shifts->setChecked(has_alignment);
    chk_shifts->setEnabled(has_alignment);
    chk_shifts->setToolTip(has_alignment
        ? QString("Use shifts from the last alignment run (%1 traces, first_sample=%2, n_samples=%3)."
                   "%4")
              .arg(activeDs().align_shifts.size()).arg(activeDs().align_first_sample).arg(activeDs().align_n_samples)
              .arg(n_align_discarded > 0
                       ? QString(" %1 of those are marked discarded and will be skipped.").arg(n_align_discarded)
                       : QString())
        : "No alignment has been applied to the main view yet.");
    fl_align->addRow(chk_shifts);
    auto applyAlignmentToSpinboxes = [&](bool on) {
        if (on) {
            sp_first->setValue(activeDs().align_first_trace);
            sp_count->setValue(static_cast<int>(activeDs().align_shifts.size()));
            sp_s_first->setValue(static_cast<int>(activeDs().align_first_sample));
            sp_s_count->setValue(static_cast<int>(activeDs().align_n_samples));
        }
        sp_first->setEnabled(!on);
        sp_count->setEnabled(!on);
        sp_s_first->setEnabled(!on);
        sp_s_count->setEnabled(!on);
    };
    connect(chk_shifts, &QCheckBox::toggled, [&](bool on){ applyAlignmentToSpinboxes(on); });
    if (has_alignment) applyAlignmentToSpinboxes(true);
    fl->addRow(grp_align);

    QSpinBox* sp_byte = nullptr;
    if (have_ttest_param) {
        auto* lbl = new QLabel(
            QString("Group assignment: <b>ttest</b> parameter (data byte offset %1)")
                .arg(auto_byte_idx));
        lbl->setTextFormat(Qt::RichText);
        fl->addRow(lbl);
    } else {
        sp_byte = new QSpinBox;
        sp_byte->setRange(0, h.data_length - 1);
        sp_byte->setValue(0);
        sp_byte->setToolTip("Index of the data byte used to assign groups.\n"
                            "0 → group 0,  non-zero → group 1.");
        fl->addRow("Group byte index:", sp_byte);
    }

    auto* chk_abs = new QCheckBox("Absolute value |t|");
    chk_abs->setToolTip("Report |t| instead of the signed t-statistic — only magnitude "
                        "matters against the significance threshold, so this makes peaks "
                        "symmetric and comparable regardless of leakage direction.");
    fl->addRow(chk_abs);

    QCheckBox* chk_add_chain = nullptr;
    if (onAddToChain) {
        chk_add_chain = new QCheckBox("Add to Chain");
        chk_add_chain->setChecked(true);
        chk_add_chain->setToolTip("Also capture these exact parameters as a Run T-test step "
                                  "in the chain being built.");
        fl->addRow(chk_add_chain);
    }

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow(cfg_bb);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first    = static_cast<int32_t>(sp_first->value());
    int32_t count    = static_cast<int32_t>(sp_count->value());
    int32_t byte_idx = have_ttest_param ? auto_byte_idx
                                        : static_cast<int32_t>(sp_byte->value());

    const bool use_alignment = chk_shifts->isChecked();
    const int32_t eff_first  = use_alignment ? activeDs().align_first_trace : first;
    const int32_t eff_count  = use_alignment ? static_cast<int32_t>(activeDs().align_shifts.size()) : count;
    const int64_t eff_first_sample = static_cast<int64_t>(sp_s_first->value());
    const int64_t eff_n_samples    = static_cast<int64_t>(sp_s_count->value()); // 0 = all
    const std::vector<int32_t> use_shifts = use_alignment ? activeDs().align_shifts : std::vector<int32_t>{};

    const bool abs_value = chk_abs->isChecked();

    if (onAddToChain && chk_add_chain->isChecked()) {
        // first/count (not eff_first/eff_count) so a step authored without
        // alignment yet still has a sensible fallback trace range recorded —
        // runTTestChainStep() re-derives eff_first/eff_count itself from
        // use_last_alignment + whatever alignment exists when it actually runs.
        ChainStep s;
        s.kind                = ChainStep::Kind::RunTTest;
        s.first_trace          = first;
        s.trace_count           = count;
        s.use_last_alignment      = use_alignment;
        s.ttest_first_sample        = eff_first_sample;
        s.ttest_n_samples             = eff_n_samples;
        s.ttest_byte_idx                = byte_idx;
        s.ttest_abs                       = abs_value;
        onAddToChain(s);
    }

    std::shared_ptr<TTestAccumulator> acc_ptr;
    std::vector<float> tstat;
    int64_t n0 = 0, n1 = 0;
    QString compute_err;
    if (!computeTTest(eff_first, eff_count, eff_first_sample, eff_n_samples, byte_idx, use_shifts,
                       abs_value, this, acc_ptr, tstat, n0, n1, compute_err)) {
        if (!compute_err.isEmpty() && compute_err != "Cancelled.")
            QMessageBox::critical(this, "T-test failed", compute_err);
        return;
    }

    buildTTestResultTab(acc_ptr, std::move(tstat), n0, n1, eff_count, abs_value);
}

// ---------------------------------------------------------------------------
// The full interactive t-test result view (threshold line, Calc TH, Style,
// Export PDF/PNG/NPY/TRS, trim controls) built from an already-computed
// result — the second half of showTTestDialog(), split out so the Chain
// "Run T-test" step gets the exact same rich result tab as the menu action.
// ---------------------------------------------------------------------------
void MainWindow::buildTTestResultTab(const std::shared_ptr<TTestAccumulator>& acc_ptr,
                                      std::vector<float> tstat, int64_t n0, int64_t n1,
                                      int32_t eff_count, bool abs_value)
{
    auto tstat_ptr  = std::make_shared<std::vector<float>>(std::move(tstat));
    auto current_ptr = std::make_shared<std::vector<float>>(*tstat_ptr);
    auto df_orig    = std::make_shared<std::vector<double>>();
    acc_ptr->computeWelchDf(*df_orig);
    auto current_df = std::make_shared<std::vector<double>>(*df_orig);

    const QString value_label = abs_value ? "|t|-value" : "t-value";
    QString tab_title = QString("Welch t-test — %1 traces (G0:%2  G1:%3)%4")
                            .arg(eff_count).arg(n0).arg(n1)
                            .arg(abs_value ? QString(" |t|") : QString());
    addResultTab(*current_ptr, tab_title, QColor("#1f77b4"), value_label);
    PlotWidget* pw = plotWidget();
    pw->setTheme(PlotTheme::light());
    pw->setTraceFilled(0, true);
    pw->setAxisLabels("Sample Index", value_label);
    pw->setThresholds(true, 4.5, -4.5);
    pw->resetView();

    // Controls row
    auto* lbl_groups = new QLabel(
        QString("Group 0: <b>%1</b> traces    Group 1: <b>%2</b> traces")
            .arg(n0).arg(n1));
    lbl_groups->setTextFormat(Qt::RichText);

    auto* lbl_thr  = new QLabel("Threshold ±:");
    auto* spin_thr = new QDoubleSpinBox;
    spin_thr->setRange(0.1, 1000.0);
    spin_thr->setValue(4.5);
    spin_thr->setDecimals(2);
    spin_thr->setSingleStep(0.5);

    auto* chk_onesided = new QCheckBox("One-sided (+)");
    chk_onesided->setToolTip("Show only the positive threshold (use when signal is preprocessed with abs())");
    connect(chk_onesided, &QCheckBox::toggled, pw, [pw, lbl_thr](bool on) {
        pw->setThresholdOneSided(on);
        lbl_thr->setText(on ? "Threshold +:" : "Threshold ±:");
    });
    // |t| is never negative, so the lower threshold half is meaningless —
    // default to one-sided display when the result was rectified.
    if (abs_value) chk_onesided->setChecked(true);

    auto* btn_exp_trs = new QPushButton("Export TRS…");
    auto* btn_exp_npy = new QPushButton("Export .npz…");

    connect(spin_thr, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [pw](double v) { pw->setThresholds(true, v, -v); });

    connect(btn_exp_trs, &QPushButton::clicked, pw, [this, current_ptr]() {
        QString path = QFileDialog::getSaveFileName(this, "Export t-test as TRS",
                                                    MainWindow::recentDir("trs"), "TRS files (*.trs)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("trs", path);
        FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
        if (!fp) { QMessageBox::critical(this, "Export failed", "Cannot create file."); return; }
        int32_t ns = static_cast<int32_t>(current_ptr->size());
        auto wle32 = [&](int32_t v) {
            uint8_t b[4] = {uint8_t(v),uint8_t(v>>8),uint8_t(v>>16),uint8_t(v>>24)};
            std::fwrite(b, 1, 4, fp);
        };
        auto wtlv = [&](uint8_t tag, uint8_t len) { std::fputc(tag,fp); std::fputc(len,fp); };
        wtlv(0x41, 4); wle32(1);
        wtlv(0x42, 4); wle32(ns);
        wtlv(0x43, 1); std::fputc(0x14, fp);
        std::fputc(0x5F, fp); std::fputc(0x00, fp);
        std::fwrite(current_ptr->data(), sizeof(float), static_cast<size_t>(ns), fp);
        std::fclose(fp);
        QMessageBox::information(this, "Export complete", "Saved: " + path);
    });

    connect(btn_exp_npy, &QPushButton::clicked, pw, [this, current_ptr, current_df]() {
        QString path = QFileDialog::getSaveFileName(this, "Export t-test as NumPy",
                                                    MainWindow::recentDir("npy"), "NumPy archive (*.npz)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("npy", path);
        std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
        entries.push_back({"tstat.npy", buildNpy1DBytes("<f4", current_ptr->data(),
                                                         static_cast<int64_t>(current_ptr->size()),
                                                         sizeof(float))});
        entries.push_back({"df.npy", buildNpy1DBytes("<f8", current_df->data(),
                                                      static_cast<int64_t>(current_df->size()),
                                                      sizeof(double))});
        QString err;
        if (!saveNpz(path, entries, err))
            QMessageBox::critical(this, "Export failed", err);
        else
            QMessageBox::information(this, "Export complete", "Saved: " + path);
    });

    auto* btn_calc_th = new QPushButton("Calc TH…");
    connect(btn_calc_th, &QPushButton::clicked, pw, [=]() {
        auto* cd = new QDialog(pw);
        cd->setWindowTitle("Threshold Calculator");
        cd->setWindowModality(Qt::WindowModal);
        auto* fl = new QFormLayout(cd);

        auto* sp_alpha = new QDoubleSpinBox;
        sp_alpha->setRange(1e-6, 0.5); sp_alpha->setDecimals(6);
        sp_alpha->setValue(0.001);      sp_alpha->setSingleStep(0.01);

        int64_t n_L = static_cast<int64_t>(tstat_ptr->size());
        auto* lbl_nL  = new QLabel(QString::number(n_L));
        auto* lbl_nA  = new QLabel(QString::number(n0));
        auto* lbl_nB  = new QLabel(QString::number(n1));

        auto* lbl_ath = new QLabel;
        auto* lbl_nu  = new QLabel;
        auto* lbl_th  = new QLabel;
        lbl_th->setTextFormat(Qt::RichText);

        // Compute median Welch df from the accumulator (data-driven)
        std::vector<double> df_vec;
        acc_ptr->computeWelchDf(df_vec);
        std::vector<double> df_sorted = df_vec;
        std::sort(df_sorted.begin(), df_sorted.end());
        double median_nu = df_sorted.empty()
            ? static_cast<double>(n0 + n1 - 2)
            : df_sorted[df_sorted.size() / 2];

        auto recalc = [=]() {
            double a    = sp_alpha->value();
            double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
            double th   = invTCdf(1.0 - a_th / 2.0, median_nu);
            lbl_ath->setText(QString::number(a_th, 'g', 4));
            lbl_nu ->setText(QString::number(median_nu, 'f', 1));
            lbl_th ->setText(QString("<b>%1</b>").arg(th, 0, 'f', 4));
        };
        connect(sp_alpha, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                cd, [=](double) { recalc(); });
        recalc();

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Apply | QDialogButtonBox::Close);
        connect(bb->button(QDialogButtonBox::Apply), &QPushButton::clicked, cd, [=]() {
            double a    = sp_alpha->value();
            double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
            spin_thr->setValue(invTCdf(1.0 - a_th / 2.0, median_nu));
        });
        connect(bb, &QDialogButtonBox::rejected, cd, &QDialog::close);

        fl->addRow("Significance level α:", sp_alpha);
        fl->addRow("Trace length n_L:",      lbl_nL);
        fl->addRow("Group A  n_A:",           lbl_nA);
        fl->addRow("Group B  n_B:",           lbl_nB);
        fl->addRow(new QLabel);
        fl->addRow("Šidák α_TH:",              lbl_ath);
        fl->addRow("Median Welch ν̂:",         lbl_nu);
        fl->addRow("Threshold TH:",           lbl_th);
        fl->addRow(bb);
        cd->show();
    });

    // Style dialog button
    auto* btn_style = new QPushButton("Style…");
    connect(btn_style, &QPushButton::clicked, pw, [=]() {
        auto* sd = new QDialog(pw);
        sd->setWindowTitle("Plot Style");
        sd->setWindowModality(Qt::NonModal);
        auto* fl2 = new QFormLayout(sd);

        auto* le_title = new QLineEdit(pw->windowTitle());
        le_title->setPlaceholderText("e.g. Welch t-test — AES-128 key byte 0");
        connect(le_title, &QLineEdit::textChanged, sd, [pw](const QString& t) { pw->setTitle(t); });

        auto* sp_width = new QDoubleSpinBox;
        sp_width->setRange(0.5, 6.0); sp_width->setValue(1.5); sp_width->setSingleStep(0.5);
        connect(sp_width, QOverload<double>::of(&QDoubleSpinBox::valueChanged), sd,
                [pw](double v) { pw->setTraceWidth(static_cast<float>(v)); });

        auto* btn_color = new QPushButton("Pick color…");
        btn_color->setStyleSheet(QString("background:%1").arg(QColor("#4fc3f7").name()));
        connect(btn_color, &QPushButton::clicked, sd, [=]() {
            QColor c = QColorDialog::getColor(pw->palette().color(QPalette::Window), sd);
            if (!c.isValid()) return;
            pw->setTraceColor(0, c);
            btn_color->setStyleSheet(QString("background:%1").arg(c.name()));
        });

        auto* btn_dark  = new QPushButton("Dark theme");
        auto* btn_light = new QPushButton("Light theme");
        connect(btn_dark,  &QPushButton::clicked, sd, [pw]() { pw->setTheme(PlotTheme::dark()); });
        connect(btn_light, &QPushButton::clicked, sd, [pw]() { pw->setTheme(PlotTheme::light()); });

        auto* bb2 = new QDialogButtonBox(QDialogButtonBox::Close);
        connect(bb2, &QDialogButtonBox::rejected, sd, &QDialog::close);

        fl2->addRow("Title:",      le_title);
        fl2->addRow("Line width:", sp_width);
        fl2->addRow("Trace color:", btn_color);
        auto* theme_row = new QWidget; auto* trl = new QHBoxLayout(theme_row);
        trl->setContentsMargins(0,0,0,0); trl->addWidget(btn_dark); trl->addWidget(btn_light);
        fl2->addRow("Theme:", theme_row);
        fl2->addRow(bb2);
        sd->show();
    });

    // PDF / PNG export buttons
    auto* btn_exp_pdf = new QPushButton("Export PDF…");
    connect(btn_exp_pdf, &QPushButton::clicked, pw, [=]() {
        QString path = QFileDialog::getSaveFileName(this, "Export t-test as PDF", MainWindow::recentDir("export"), "PDF files (*.pdf)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPdfWriter writer(path);
        writer.setPageSize(QPageSize(QPageSize::A4));
        writer.setPageOrientation(QPageLayout::Landscape);
        writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout::Millimeter);
        QPainter painter(&writer);
        double sx = static_cast<double>(writer.width())  / pw->width();
        double sy = static_cast<double>(writer.height()) / pw->height();
        double sc = std::min(sx, sy);
        painter.scale(sc, sc);
        pw->render(&painter);
        painter.end();
        QMessageBox::information(this, "Exported", "Saved: " + path);
    });
    auto* btn_exp_png = new QPushButton("Export PNG…");
    connect(btn_exp_png, &QPushButton::clicked, pw, [=]() {
        QString path = QFileDialog::getSaveFileName(this, "Export t-test as PNG", MainWindow::recentDir("export"), "PNG images (*.png)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPixmap pix = pw->grab();
        if (!pix.save(path, "PNG"))
            QMessageBox::warning(this, "Export PNG", "Could not save:\n" + path);
        else
            QMessageBox::information(this, "Exported", "Saved: " + path);
    });

    auto* ctrl = new QWidget;
    auto* ctrl_l = new QHBoxLayout(ctrl);
    ctrl_l->setContentsMargins(4, 2, 4, 2);
    ctrl_l->setSpacing(6);

    // Mode buttons
    auto* btn_pan_tt  = new QPushButton("Pan");
    auto* btn_meas_tt = new QPushButton("Measure");
    auto* btn_bz_tt   = new QPushButton("⬚ Zoom");
    auto* btn_crop_tt = new QPushButton("✂ Cut");
    auto* btn_rst_tt  = new QPushButton("Reset");
    btn_pan_tt->setCheckable(true);  btn_pan_tt->setChecked(true);
    btn_meas_tt->setCheckable(true);
    btn_bz_tt->setCheckable(true);
    btn_crop_tt->setCheckable(true);
    btn_pan_tt->setToolTip("Drag to pan, scroll to zoom");
    btn_meas_tt->setToolTip("Click two points to measure distance");
    btn_bz_tt->setToolTip("Drag to rubber-band zoom");
    btn_crop_tt->setToolTip("Drag to select a region to cut (exclude from export)");
    btn_rst_tt->setToolTip("Reset view");
    auto* mg_tt = new QButtonGroup(pw);
    mg_tt->addButton(btn_pan_tt,  0);
    mg_tt->addButton(btn_meas_tt, 1);
    mg_tt->addButton(btn_bz_tt,   2);
    mg_tt->addButton(btn_crop_tt, 3);
    connect(mg_tt, QOverload<int>::of(&QButtonGroup::idClicked), pw, [pw](int id) {
        InteractionMode m = id == 0 ? InteractionMode::Pan
                          : id == 1 ? InteractionMode::Measure
                          : id == 2 ? InteractionMode::BoxZoom
                                    : InteractionMode::CropSelect;
        pw->setMode(m);
    });
    connect(btn_rst_tt, &QPushButton::clicked, pw, [pw]() { pw->resetView(); });

    ctrl_l->addWidget(btn_pan_tt);
    ctrl_l->addWidget(btn_meas_tt);
    ctrl_l->addWidget(btn_bz_tt);
    ctrl_l->addWidget(btn_crop_tt);
    ctrl_l->addWidget(btn_rst_tt);
    ctrl_l->addSpacing(8);
    ctrl_l->addWidget(lbl_groups);
    ctrl_l->addStretch();
    auto* btn_yzi = new QPushButton("↑ Amp");
    auto* btn_yzo = new QPushButton("↓ Amp");
    btn_yzi->setToolTip("Zoom in Y (Ctrl/Shift+scroll up)");
    btn_yzo->setToolTip("Zoom out Y / shorter traces (Ctrl/Shift+scroll down)");
    connect(btn_yzi, &QPushButton::clicked, pw, [pw](){ pw->zoomInY(); });
    connect(btn_yzo, &QPushButton::clicked, pw, [pw](){ pw->zoomOutY(); });

    ctrl_l->addWidget(lbl_thr);
    ctrl_l->addWidget(spin_thr);
    ctrl_l->addWidget(chk_onesided);
    ctrl_l->addWidget(btn_calc_th);
    ctrl_l->addSpacing(8);
    ctrl_l->addWidget(btn_yzi);
    ctrl_l->addWidget(btn_yzo);
    ctrl_l->addSpacing(8);
    ctrl_l->addWidget(btn_style);
    ctrl_l->addStretch();
    ctrl_l->addWidget(btn_exp_trs);
    ctrl_l->addWidget(btn_exp_npy);
    ctrl_l->addWidget(btn_exp_pdf);
    ctrl_l->addWidget(btn_exp_png);

    // Trim row
    int64_t n_full_tt = static_cast<int64_t>(tstat_ptr->size());
    auto* trim_row = new QWidget;
    auto* trim_l   = new QHBoxLayout(trim_row);
    trim_l->setContentsMargins(4, 0, 4, 0);
    auto* sp_excl_start = new QSpinBox; sp_excl_start->setRange(0, static_cast<int>(n_full_tt / 2));
    auto* sp_excl_end   = new QSpinBox; sp_excl_end->setRange(0, static_cast<int>(n_full_tt / 2));
    sp_excl_start->setToolTip("Samples to exclude from the start");
    sp_excl_end->setToolTip("Samples to exclude from the end");
    auto* lbl_vis_tt = new QLabel; lbl_vis_tt->setTextFormat(Qt::RichText);

    auto upd_lbl_tt = [=]() {
        lbl_vis_tt->setText(QString("Exporting <b>%1</b> / %2 samples")
            .arg(current_ptr->size()).arg(n_full_tt));
    };
    auto skip_tt = std::make_shared<bool>(false);
    auto do_trim_tt = [=]() {
        int ts = sp_excl_start->value(), te = sp_excl_end->value();
        int64_t s = ts, e = std::max<int64_t>(ts + 1, n_full_tt - te);
        *current_ptr = std::vector<float>(tstat_ptr->begin() + s, tstat_ptr->begin() + e);
        auto ds = std::min(s, (int64_t)df_orig->size());
        auto de = std::min(e, (int64_t)df_orig->size());
        *current_df = std::vector<double>(df_orig->begin() + ds, df_orig->begin() + de);
        *skip_tt = true; pw->clearCropRanges(); *skip_tt = false;
        pw->replaceMemTrace(0, current_ptr);
        upd_lbl_tt();
    };
    connect(pw, &PlotWidget::cropRangesChanged, pw, [=]() {
        if (*skip_tt) return;
        const auto& cr = pw->cropRanges();
        if (cr.empty()) return;
        std::vector<std::pair<int64_t,int64_t>> cuts(cr.begin(), cr.end());
        std::sort(cuts.begin(), cuts.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        for (const auto& r : cuts) {
            auto s = std::max<int64_t>(0, r.first);
            auto e = std::min<int64_t>((int64_t)current_ptr->size(), r.second);
            if (s < e) {
                current_ptr->erase(current_ptr->begin() + s, current_ptr->begin() + e);
                auto ds = std::min(s,  (int64_t)current_df->size());
                auto de = std::min(e,  (int64_t)current_df->size());
                if (ds < de) current_df->erase(current_df->begin() + ds, current_df->begin() + de);
            }
        }
        *skip_tt = true; pw->clearCropRanges(); *skip_tt = false;
        pw->replaceMemTrace(0, current_ptr);
        upd_lbl_tt();
    });
    connect(sp_excl_start, QOverload<int>::of(&QSpinBox::valueChanged), pw, [=](int){ do_trim_tt(); });
    connect(sp_excl_end,   QOverload<int>::of(&QSpinBox::valueChanged), pw, [=](int){ do_trim_tt(); });
    upd_lbl_tt();

    trim_l->addWidget(new QLabel("Exclude:"));
    trim_l->addWidget(new QLabel("start"));
    trim_l->addWidget(sp_excl_start);
    trim_l->addWidget(new QLabel("end"));
    trim_l->addWidget(sp_excl_end);
    trim_l->addWidget(new QLabel("samples  —"));
    trim_l->addWidget(lbl_vis_tt);
    trim_l->addStretch();

    auto* extra = new QWidget;
    auto* extra_l = new QVBoxLayout(extra);
    extra_l->setContentsMargins(0, 0, 0, 0);
    extra_l->setSpacing(4);
    extra_l->addWidget(ctrl);
    extra_l->addWidget(trim_row);
    activeDs().extra_toolbar = extra;
    updateViewLayout();
}

// ---------------------------------------------------------------------------
// Crop & Merge
// ---------------------------------------------------------------------------

void MainWindow::onCropEdit() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Crop & Merge", "No file loaded.");
        return;
    }

    // Restore Pan mode when dialog is closed
    InteractionMode prev_mode = plotWidget()->mode();

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle("Crop & Merge — Range Editor");
    dlg->resize(500, 380);

    // ---- widgets ----
    auto* lbl_hint = new QLabel(
        "Drag on the plot (in drag-select mode) or click "
        "<b>Add current view</b> to add sample ranges.\n"
        "Ranges are concatenated per trace on export, unless "
        "<b>Export ranges separately</b> is checked below.");
    lbl_hint->setTextFormat(Qt::RichText);
    lbl_hint->setWordWrap(true);

    auto* btn_drag = new QPushButton("Enable drag-select on plot");
    btn_drag->setCheckable(true);
    btn_drag->setToolTip("When enabled, drag on the plot to add ranges");

    auto* list = new QListWidget;
    list->setSelectionMode(QAbstractItemView::SingleSelection);

    auto* lbl_total = new QLabel("Total: 0 samples across 0 ranges");

    auto* btn_add_view = new QPushButton("Add current view");
    auto* btn_remove   = new QPushButton("Remove selected");
    auto* btn_clear    = new QPushButton("Clear all");
    auto* btn_export   = new QPushButton("Export TRS…");
    auto* btn_close    = new QPushButton("Close");

    // Repeat generator: takes the selected list row (or the last range if
    // none selected) as the base window and stamps out N-1 more same-width
    // copies of it, `period` samples apart — for marking every occurrence of
    // a repeating operation (e.g. AES rounds) once you've hand-picked one.
    auto* grp_repeat  = new QGroupBox("Repeat selected range");
    auto* fl_repeat   = new QFormLayout(grp_repeat);
    auto* sp_rep_count  = new QSpinBox;
    sp_rep_count->setRange(1, 100000);
    sp_rep_count->setValue(1);
    sp_rep_count->setToolTip("Total number of copies, including the base range.");
    auto* sp_rep_period = new QSpinBox;
    sp_rep_period->setRange(1, 2000000000);
    sp_rep_period->setValue(1);
    sp_rep_period->setToolTip("Samples between the start of each successive copy.");
    auto* btn_apply_repeat = new QPushButton("Add repeats");
    fl_repeat->addRow("Count:",           sp_rep_count);
    fl_repeat->addRow("Period (samples):", sp_rep_period);
    fl_repeat->addRow(btn_apply_repeat);

    auto* chk_separate = new QCheckBox("Export ranges separately (one output trace per range, not merged)");
    chk_separate->setToolTip(
        "Off: each source trace's ranges are concatenated into one output trace (old behavior).\n"
        "On: every range becomes its own output trace, so N traces × K ranges → N×K output traces.\n"
        "Requires all ranges to be the same length.");

    // ---- layout ----
    auto* vl = new QVBoxLayout(dlg);
    vl->addWidget(lbl_hint);
    vl->addWidget(btn_drag);
    vl->addWidget(list, 1);
    vl->addWidget(lbl_total);
    vl->addWidget(grp_repeat);
    vl->addWidget(chk_separate);

    auto* btns_l = new QHBoxLayout;
    btns_l->addWidget(btn_add_view);
    btns_l->addWidget(btn_remove);
    btns_l->addWidget(btn_clear);
    btns_l->addStretch();
    btns_l->addWidget(btn_export);
    btns_l->addWidget(btn_close);
    vl->addLayout(btns_l);

    // ---- helpers ----
    auto rebuildList = [=]() {
        list->clear();
        const auto& ranges = plotWidget()->cropRanges();
        int64_t total = 0;
        for (int i = 0; i < static_cast<int>(ranges.size()); i++) {
            int64_t len = ranges[i].second - ranges[i].first;
            total += len;
            list->addItem(
                QString("#%1   %2 – %3   (%4 samples)")
                    .arg(i + 1)
                    .arg(ranges[i].first)
                    .arg(ranges[i].second)
                    .arg(len));
        }
        lbl_total->setText(
            QString("Total: <b>%1</b> samples across <b>%2</b> range(s)")
                .arg(total).arg(ranges.size()));
        btn_export->setEnabled(!ranges.empty());
    };
    rebuildList();

    // ---- connections ----
    connect(plotWidget(), &PlotWidget::cropRangesChanged, dlg, rebuildList);

    connect(btn_drag, &QPushButton::toggled, dlg, [this](bool on) {
        plotWidget()->setMode(on ? InteractionMode::CropSelect : InteractionMode::Pan);
    });

    connect(btn_add_view, &QPushButton::clicked, dlg, [this]() {
        plotWidget()->addCropRange(plotWidget()->viewStart(), plotWidget()->viewEnd());
    });

    connect(btn_remove, &QPushButton::clicked, dlg, [=]() {
        int row = list->currentRow();
        if (row >= 0) plotWidget()->removeCropRangeAt(row);
    });

    connect(btn_apply_repeat, &QPushButton::clicked, dlg, [this, dlg, list, sp_rep_count, sp_rep_period]() {
        const auto& ranges = plotWidget()->cropRanges();
        if (ranges.empty()) {
            QMessageBox::information(dlg, "Repeat range", "Add a base range first (drag on the plot or click \"Add current view\").");
            return;
        }
        // Base window: the selected row, or the last range if none is selected.
        int base_idx = list->currentRow();
        if (base_idx < 0) base_idx = static_cast<int>(ranges.size()) - 1;
        auto [base_start, base_end] = ranges[base_idx];
        int64_t width  = base_end - base_start;
        if (width <= 0) return;
        int64_t period = sp_rep_period->value();
        int     count  = sp_rep_count->value();
        int64_t total  = plotWidget()->totalSamples();
        int     added = 0;
        for (int i = 1; i < count; i++) {
            int64_t s = base_start + static_cast<int64_t>(i) * period;
            int64_t e = s + width;
            if (e > total) break; // stop once a copy would run past the end of the trace
            plotWidget()->addCropRange(s, e, /*is_repeat=*/true);
            added++;
        }
        if (added < count - 1) {
            QMessageBox::information(dlg, "Repeat range",
                QString("Added %1 of %2 requested repeats; the rest would run past the end of the trace.")
                    .arg(added).arg(count - 1));
        }
    });

    connect(btn_clear, &QPushButton::clicked, dlg, [this]() {
        plotWidget()->clearCropRanges();
    });

    connect(btn_close, &QPushButton::clicked, dlg, &QDialog::close);

    connect(dlg, &QDialog::finished, this, [this, prev_mode]() {
        // Restore previous interaction mode and clear any pending rubber-band
        plotWidget()->setMode(prev_mode);
    });

    // ---- export ----
    connect(btn_export, &QPushButton::clicked, dlg, [this, dlg, chk_separate]() {
        const auto& ranges = plotWidget()->cropRanges();
        if (ranges.empty()) return;

        bool separate = chk_separate->isChecked();

        // Separate mode needs one fixed sample count for every output trace,
        // so every range must be the same width (TRS files have a single
        // global NUMBER_SAMPLES for all traces).
        int64_t range_width = ranges[0].second - ranges[0].first;
        if (separate) {
            for (const auto& r : ranges) {
                if (r.second - r.first != range_width) {
                    QMessageBox::warning(dlg, "Export failed",
                        "\"Export ranges separately\" requires every range to be the same "
                        "length. Use the repeat generator, or fix up ranges so their widths match.");
                    return;
                }
            }
        }

        // Compute total samples per output trace
        int64_t total_samples = 0;
        for (const auto& r : ranges) total_samples += r.second - r.first;
        if (total_samples <= 0) return;

        const TrsHeader& h = activeDs().file->header();

        QString path = QFileDialog::getSaveFileName(
            dlg, "Export cropped TRS", MainWindow::recentDir("trs"), "TRS files (*.trs)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("trs", path);

        int     n_src_traces = h.num_traces;
        int64_t out_samples  = separate ? range_width : total_samples;
        int     n_out_traces = separate
            ? static_cast<int>(static_cast<int64_t>(n_src_traces) * static_cast<int64_t>(ranges.size()))
            : n_src_traces;

        QProgressDialog prog("Exporting traces…", "Cancel", 0, n_src_traces, dlg);
        prog.setWindowModality(Qt::WindowModal);
        prog.setMinimumDuration(400);

        FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
        if (!fp) {
            QMessageBox::critical(dlg, "Export failed", "Cannot create:\n" + path);
            return;
        }

        // Header helpers
        auto fputle16 = [&](int16_t v) {
            uint8_t b[2] = { uint8_t(v & 0xFF), uint8_t((v >> 8) & 0xFF) };
            std::fwrite(b, 1, 2, fp);
        };
        auto fputle32 = [&](int32_t v) {
            uint8_t b[4] = {uint8_t(v),uint8_t(v>>8),uint8_t(v>>16),uint8_t(v>>24)};
            std::fwrite(b, 1, 4, fp);
        };
        auto tlv = [&](uint8_t tag, uint8_t len) {
            std::fputc(tag, fp); std::fputc(len, fp);
        };

        // Write TRS header
        tlv(0x41, 4); fputle32(n_out_traces);                    // NUMBER_TRACES
        tlv(0x42, 4); fputle32(static_cast<int32_t>(out_samples)); // NUMBER_SAMPLES
        tlv(0x43, 1); std::fputc(0x14, fp);                     // float32
        if (h.data_length > 0) {
            tlv(0x44, 2); fputle16(h.data_length);              // DATA_LENGTH
        }
        std::fputc(0x5F, fp); std::fputc(0x00, fp);             // TRACE_BLOCK

        constexpr int64_t CHUNK = 256 * 1024;
        std::vector<float> buf(CHUNK);
        bool cancelled = false;

        // Writes samples [s, end) of source trace ti, zero-filling on a short read.
        auto writeSampleRange = [&](int ti, int64_t s, int64_t end) {
            while (s < end) {
                int64_t chunk = std::min(CHUNK, end - s);
                int64_t read  = activeDs().file->readSamples(ti, s, chunk, buf.data());
                if (read <= 0) {
                    int64_t remain = end - s;
                    std::fill(buf.begin(), buf.begin() + remain, 0.0f);
                    std::fwrite(buf.data(), sizeof(float),
                                static_cast<size_t>(remain), fp);
                    break;
                }
                std::fwrite(buf.data(), sizeof(float),
                            static_cast<size_t>(read), fp);
                s += read;
            }
        };

        for (int ti = 0; ti < n_src_traces && !cancelled; ti++) {
            if (prog.wasCanceled()) { cancelled = true; break; }
            prog.setLabelText(QString("Exporting trace %1 / %2…").arg(ti + 1).arg(n_src_traces));
            prog.setValue(ti);
            QApplication::processEvents();

            std::vector<uint8_t> data;
            if (h.data_length > 0) data = activeDs().file->readData(ti);

            if (separate) {
                // One output trace per range: aux data (if any) is repeated
                // for each range copy, since every output trace needs its own.
                for (const auto& r : ranges) {
                    if (h.data_length > 0) std::fwrite(data.data(), 1, data.size(), fp);
                    writeSampleRange(ti, r.first, r.second);
                }
            } else {
                // Concatenate each range into one output trace.
                if (h.data_length > 0) std::fwrite(data.data(), 1, data.size(), fp);
                for (const auto& r : ranges) writeSampleRange(ti, r.first, r.second);
            }
        }

        prog.setValue(n_src_traces);
        std::fclose(fp);

        if (cancelled) {
            QFile::remove(path);
            QMessageBox::information(dlg, "Cancelled", "Export was cancelled.");
        } else if (separate) {
            QMessageBox::information(dlg, "Export complete",
                QString("Saved %1 trace(s) (%2 source × %3 ranges), %4 samples each, to:\n%5")
                    .arg(n_out_traces).arg(n_src_traces).arg(ranges.size()).arg(out_samples).arg(path));
        } else {
            QMessageBox::information(dlg, "Export complete",
                QString("Saved %1 trace(s) with %2 samples each to:\n%3")
                    .arg(n_out_traces).arg(out_samples).arg(path));
        }
    });

    dlg->show();
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-Correlation SCA
// ---------------------------------------------------------------------------

void MainWindow::onRunXCorr() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Cross-Correlation", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();
    int n_total   = h.num_traces;
    int n_samples = h.num_samples;

    // Reference/template window pickers below are pipeline-processed sample
    // units, same rationale and conversion helpers as onAlignTraces() — see
    // the comment there. Only the small "match this pattern" windows
    // (Two-Window's reference, Template Match's template) get this
    // treatment; the general "Samples" range shared by every method (matrix
    // size, search region) stays raw since it isn't a search/template window.
    const int64_t proc_total_xcorr = processedCountForRaw(activeDs().pipeline, h.num_samples);

    struct TemplateSpec { int32_t trace; int64_t first; int64_t len; };
    // Heap-allocated (like draw_target below): cfg is non-modal, so onRunXCorr()
    // returns immediately and every lambda below must keep this alive by holding
    // a copy of the shared_ptr rather than a reference to a local that's gone.
    auto template_list = std::make_shared<QVector<TemplateSpec>>();

    // ---- Config dialog ----
    // Non-modal (show(), not exec()): QDialog::setVisible(false) tears down an
    // exec() event loop as soon as it's hidden, which made the old hide()-while-
    // drawing trick close this dialog outright instead of just stepping aside.
    // Non-modal avoids that entirely — the main plot is always interactive.
    auto* cfg = new QDialog(this);
    cfg->setAttribute(Qt::WA_DeleteOnClose);
    cfg->setWindowTitle("Cross-Correlation — configuration");
    auto* vl_cfg = new QVBoxLayout(cfg);

    // Trace range
    auto* grp_traces = new QGroupBox("Traces");
    auto* fl_traces  = new QFormLayout(grp_traces);
    auto* sp_first   = new QSpinBox; sp_first->setRange(0, std::max(0, n_total - 1)); sp_first->setValue(0);
    auto* sp_count   = new QSpinBox; sp_count->setRange(1, n_total); sp_count->setValue(n_total);
    fl_traces->addRow("First trace:", sp_first);
    fl_traces->addRow("Count:",       sp_count);

    // Sample range
    auto* grp_samples = new QGroupBox("Samples");
    auto* fl_samples  = new QFormLayout(grp_samples);
    auto* sp_s_first  = new QSpinBox; sp_s_first->setRange(0, std::max(0, n_samples - 1)); sp_s_first->setValue(0);
    auto* sp_s_count  = new QSpinBox; sp_s_count->setRange(0, n_samples); sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");
    fl_samples->addRow("First sample:", sp_s_first);
    fl_samples->addRow("Count (0=all):", sp_s_count);

    // Stride / downsampling
    auto* grp_ds   = new QGroupBox("Downsampling");
    auto* fl_ds    = new QFormLayout(grp_ds);
    auto* sp_stride = new QSpinBox; sp_stride->setRange(1, 10000); sp_stride->setValue(1);
    sp_stride->setToolTip("Take every Nth sample; M = ceil(num_samples / stride).\n"
                          "Increase to reduce memory and computation time.");
    auto* lbl_M    = new QLabel;
    auto* lbl_mem  = new QLabel;
    fl_ds->addRow("Stride:", sp_stride);
    fl_ds->addRow("Output M (samples):", lbl_M);
    fl_ds->addRow("Matrix memory:", lbl_mem);

    // Method
    auto* grp_method  = new QGroupBox("Method");
    auto* vl_method   = new QVBoxLayout(grp_method);
    auto* rb_baseline = new QRadioButton("Baseline  (direct M×M outer products)");
    auto* rb_dual     = new QRadioButton("Dual Matrix  (via n×n Gram eigendecomposition)");
    auto* rb_mp       = new QRadioButton("MP-Cleaned  (Marchenko-Pastur denoising)");
    rb_baseline->setChecked(true);
    auto* lbl_dual_warn = new QLabel(
        "<small><i>Dual / MP methods use Eigen SelfAdjointEigenSolver + OpenMP.\n"
        "Memory limit: ~4 GB. Practical limit: n ≤ 5000 for reasonable speed.</i></small>");
    lbl_dual_warn->setTextFormat(Qt::RichText);
    lbl_dual_warn->setWordWrap(true);
    lbl_dual_warn->setEnabled(false);
    auto* rb_twowin   = new QRadioButton("Two-Window Template Match  (search × ref rectangular C)");
    auto* rb_template = new QRadioButton("Template Match  (fixed template, sliding NCC per trace)");
    // MP threshold scale (only shown for MP-Cleaned)
    auto* mp_scale_row   = new QWidget;
    auto* mp_scale_hl    = new QHBoxLayout(mp_scale_row);
    mp_scale_hl->setContentsMargins(20, 0, 0, 0);
    auto* mp_scale_lbl   = new QLabel("Threshold scale (×λ+):");
    auto* sp_mp_scale    = new QDoubleSpinBox;
    sp_mp_scale->setRange(1.0, 10.0);
    sp_mp_scale->setSingleStep(0.1);
    sp_mp_scale->setDecimals(2);
    sp_mp_scale->setValue(1.0);
    sp_mp_scale->setToolTip("Multiply λ+ by this factor before thresholding.\n"
                            "1.0 = standard MP cutoff.\n"
                            ">1.0 = keep only stronger signal components (more denoising).");
    mp_scale_hl->addWidget(mp_scale_lbl);
    mp_scale_hl->addWidget(sp_mp_scale);
    mp_scale_row->setVisible(false);

    vl_method->addWidget(rb_baseline);
    vl_method->addWidget(rb_dual);
    vl_method->addWidget(rb_mp);
    vl_method->addWidget(mp_scale_row);
    vl_method->addWidget(rb_twowin);
    vl_method->addWidget(rb_template);
    vl_method->addWidget(lbl_dual_warn);

    connect(rb_mp,   &QRadioButton::toggled, mp_scale_row, &QWidget::setVisible);
    connect(rb_dual, &QRadioButton::toggled, lbl_dual_warn, &QLabel::setEnabled);
    connect(rb_mp,   &QRadioButton::toggled, [lbl_dual_warn, rb_dual, rb_mp]() {
        lbl_dual_warn->setEnabled(rb_dual->isChecked() || rb_mp->isChecked());
    });

    // Alignment group
    const bool has_alignment_xcorr = (activeDs().align_n_samples > 0);
    const int  n_align_discarded_xcorr = static_cast<int>(std::count(
        activeDs().align_shifts.begin(), activeDs().align_shifts.end(), kAlignDiscardShift));
    auto* grp_align_xcorr  = new QGroupBox("Alignment");
    auto* fl_align_xcorr   = new QFormLayout(grp_align_xcorr);
    auto* chk_shifts_xcorr = new QCheckBox("Apply last alignment shifts");
    chk_shifts_xcorr->setChecked(has_alignment_xcorr);
    chk_shifts_xcorr->setEnabled(has_alignment_xcorr);
    chk_shifts_xcorr->setToolTip(has_alignment_xcorr
        ? QString("Use shifts from the last alignment run (%1 traces, first_sample=%2, n_samples=%3)."
                   "%4")
              .arg(activeDs().align_shifts.size()).arg(activeDs().align_first_sample).arg(activeDs().align_n_samples)
              .arg(n_align_discarded_xcorr > 0
                       ? QString(" %1 of those are marked discarded and will be skipped.").arg(n_align_discarded_xcorr)
                       : QString())
        : "No alignment has been applied to the main view yet.");
    fl_align_xcorr->addRow(chk_shifts_xcorr);
    // NOTE: [=] (not [&]) throughout this dialog — cfg is non-modal, so
    // onRunXCorr() returns as soon as cfg->show() runs, and these lambdas keep
    // firing long after that. Capturing locals by reference would dangle.
    auto applyAlignmentToSpinboxesXCorr = [=](bool on) {
        if (on) {
            sp_first->setValue(activeDs().align_first_trace);
            sp_count->setValue(static_cast<int>(activeDs().align_shifts.size()));
            sp_s_first->setValue(static_cast<int>(activeDs().align_first_sample));
            sp_s_count->setValue(static_cast<int>(activeDs().align_n_samples));
        }
        sp_first->setEnabled(!on);
        sp_count->setEnabled(!on);
        sp_s_first->setEnabled(!on);
        sp_s_count->setEnabled(!on);
    };
    connect(chk_shifts_xcorr, &QCheckBox::toggled, cfg, [=](bool on){ applyAlignmentToSpinboxesXCorr(on); });
    if (has_alignment_xcorr) applyAlignmentToSpinboxesXCorr(true);

    // Reference window (Two-Window mode only)
    auto* grp_ref   = new QGroupBox("Reference Window (Two-Window mode)");
    auto* fl_ref    = new QFormLayout(grp_ref);
    auto* sp_r_first = new QSpinBox;
    sp_r_first->setRange(0, static_cast<int>(std::max<int64_t>(0, proc_total_xcorr - 1)));
    sp_r_first->setValue(0);
    sp_r_first->setToolTip("Pipeline-processed sample index.");
    auto* sp_r_count = new QSpinBox;
    sp_r_count->setRange(1, static_cast<int>(std::max<int64_t>(1, proc_total_xcorr)));
    sp_r_count->setValue(static_cast<int>(std::min<int64_t>(512, proc_total_xcorr)));
    sp_r_count->setToolTip("Length in pipeline-processed samples.");
    fl_ref->addRow("Ref first sample (pipeline samples):", sp_r_first);
    fl_ref->addRow("Ref count (pipeline samples):",        sp_r_count);
    grp_ref->setVisible(false);
    connect(rb_twowin, &QRadioButton::toggled, grp_ref, &QWidget::setVisible);

    // Template region(s) (Template Match mode only): step 1 — pick one or more templates.
    // Each entry queued in the list below is matched independently; a separate
    // result window opens for every template once the run completes.
    auto* grp_tmpl   = new QGroupBox("1 · Templates  (Template Match mode)");
    auto* vl_tmpl    = new QVBoxLayout(grp_tmpl);
    auto* fl_tmpl    = new QFormLayout;
    vl_tmpl->addLayout(fl_tmpl);
    auto* sp_tmpl_trace = new QSpinBox;
    sp_tmpl_trace->setRange(0, std::max(0, n_total - 1));
    sp_tmpl_trace->setValue(spin_first_->value());
    sp_tmpl_trace->setToolTip("Absolute trace index the template pattern is extracted from.");
    auto* sp_tmpl_first = new QSpinBox;
    sp_tmpl_first->setRange(0, static_cast<int>(std::max<int64_t>(0, proc_total_xcorr - 1)));
    sp_tmpl_first->setValue(0);
    sp_tmpl_first->setToolTip("Pipeline-processed sample index.");
    auto* sp_tmpl_len   = new QSpinBox;
    sp_tmpl_len->setRange(2, static_cast<int>(std::max<int64_t>(2, proc_total_xcorr)));
    sp_tmpl_len->setValue(static_cast<int>(std::min<int64_t>(512, proc_total_xcorr)));
    sp_tmpl_len->setToolTip("Length in pipeline-processed samples.");

    // Seed from first crop range if one exists (crop ranges are always raw —
    // the plot's own coordinate system — so convert to processed here).
    if (!plotWidget()->cropRanges().empty()) {
        auto [cs, ce] = plotWidget()->cropRanges()[0];
        int64_t pcs = processedCountForRaw(activeDs().pipeline, cs);
        int64_t pce = processedCountForRaw(activeDs().pipeline, ce);
        sp_tmpl_first->setValue(static_cast<int>(pcs));
        sp_tmpl_len->setValue(static_cast<int>(std::max<int64_t>(2, pce - pcs)));
    }

    auto* btn_draw_tmpl = new QPushButton("Draw on plot →");
    btn_draw_tmpl->setToolTip("Switch the main plot to crop-select mode.\n"
                              "Drag to mark the template region, press Enter to confirm\n"
                              "(Escape cancels), then this dialog comes back.");

    auto* tmpl_region_row = new QWidget;
    auto* tmpl_region_hl  = new QHBoxLayout(tmpl_region_row);
    tmpl_region_hl->setContentsMargins(0, 0, 0, 0);
    tmpl_region_hl->addWidget(new QLabel("First:"));
    tmpl_region_hl->addWidget(sp_tmpl_first);
    tmpl_region_hl->addWidget(new QLabel("Length:"));
    tmpl_region_hl->addWidget(sp_tmpl_len);
    tmpl_region_hl->addWidget(btn_draw_tmpl);
    tmpl_region_hl->addStretch();

    fl_tmpl->addRow("Template trace:",  sp_tmpl_trace);
    fl_tmpl->addRow("Template region (pipeline samples):", tmpl_region_row);

    // Queued templates — each drawn region is appended here automatically;
    // "Add to list" lets you queue the current spin-box values without drawing.
    auto* lw_templates = new QListWidget;
    lw_templates->setToolTip("Each entry here is matched independently against the search\n"
                             "region — one result window opens per template.\n"
                             "Empty list = run once, using the values above.");
    auto* hl_tmpl_btns    = new QHBoxLayout;
    auto* btn_tmpl_add    = new QPushButton("Add to list");
    auto* btn_tmpl_remove = new QPushButton("Remove");
    btn_tmpl_remove->setEnabled(false);
    hl_tmpl_btns->addWidget(btn_tmpl_add);
    hl_tmpl_btns->addWidget(btn_tmpl_remove);
    hl_tmpl_btns->addStretch();
    vl_tmpl->addWidget(lw_templates);
    vl_tmpl->addLayout(hl_tmpl_btns);

    auto tmplSummary = [](const TemplateSpec& s) {
        return QString("Trace %1   [%2, %3)   len %4")
            .arg(s.trace).arg(s.first).arg(s.first + s.len).arg(s.len);
    };
    auto addTemplateToList = [=](const TemplateSpec& spec) {
        template_list->push_back(spec);
        new QListWidgetItem(tmplSummary(spec), lw_templates);
    };

    connect(lw_templates, &QListWidget::currentRowChanged, cfg, [=](int row) {
        btn_tmpl_remove->setEnabled(row >= 0);
    });
    connect(btn_tmpl_add, &QPushButton::clicked, cfg, [=]() {
        // sp_tmpl_first/sp_tmpl_len are pipeline-processed; TemplateSpec (and
        // everything that consumes it) expects raw sample positions/counts.
        const auto& pl = activeDs().pipeline;
        int64_t proc_first = static_cast<int64_t>(sp_tmpl_first->value());
        int64_t proc_end    = proc_first + static_cast<int64_t>(sp_tmpl_len->value());
        int64_t raw_first  = rawCountForProcessedCount(pl, proc_first, h.num_samples);
        int64_t raw_end     = rawCountForProcessedCount(pl, proc_end,   h.num_samples);
        addTemplateToList({ static_cast<int32_t>(sp_tmpl_trace->value()),
                            raw_first, std::max<int64_t>(1, raw_end - raw_first) });
    });
    connect(btn_tmpl_remove, &QPushButton::clicked, cfg, [=]() {
        int row = lw_templates->currentRow();
        if (row < 0) return;
        template_list->remove(row);
        delete lw_templates->takeItem(row);
    });

    grp_tmpl->setVisible(false);
    connect(rb_template, &QRadioButton::toggled, grp_tmpl, &QWidget::setVisible);

    // Search region draw button, appended next to the existing "Samples" group
    // (used as the search window in Template Match mode).
    auto* btn_draw_search = new QPushButton("Draw on plot →");
    btn_draw_search->setToolTip("Switch the main plot to crop-select mode.\n"
                                "Drag to mark the search region, press Enter to confirm\n"
                                "(Escape cancels), then this dialog comes back.");
    btn_draw_search->setVisible(false);
    fl_samples->addRow("", btn_draw_search);
    connect(rb_template, &QRadioButton::toggled, btn_draw_search, &QWidget::setVisible);
    connect(rb_template, &QRadioButton::toggled, cfg, [=](bool on) {
        grp_samples->setTitle(on ? "2 · Search Region" : "Samples");
    });

    // ── Interactive drag-select wiring, shared between the two drawable
    //    regions (template / search) — only one can be "active" at a time.
    enum class DrawTarget { None, Template, Search };
    auto draw_target = std::make_shared<DrawTarget>(DrawTarget::None);

    // cfg is non-modal (show(), not exec()), so the main plot stays interactive
    // the whole time — no need to hide/show this dialog to "step out of the way".
    auto startDraw = [=](DrawTarget target) {
        *draw_target = target;
        plotWidget()->clearCropRanges();
        plotWidget()->setMode(InteractionMode::CropSelect);
        btn_draw_tmpl->setText("Drag, then Enter…");
        btn_draw_search->setText("Drag, then Enter…");
        btn_draw_tmpl->setEnabled(false);
        btn_draw_search->setEnabled(false);
    };
    connect(btn_draw_tmpl,   &QPushButton::clicked, cfg, [=]() { startDraw(DrawTarget::Template); });
    connect(btn_draw_search, &QPushButton::clicked, cfg, [=]() { startDraw(DrawTarget::Search); });

    auto* crop_guard = new QObject(cfg);
    connect(plotWidget(), &PlotWidget::cropRangesChanged, crop_guard, [=]() {
        const auto& ranges = plotWidget()->cropRanges();
        if (ranges.empty() || *draw_target == DrawTarget::None) return;
        auto [s, e] = ranges.back();
        if (*draw_target == DrawTarget::Template) {
            sp_tmpl_first->setValue(static_cast<int>(s));
            sp_tmpl_len->setValue(static_cast<int>(std::max<int64_t>(2, e - s)));
            // Each confirmed drag queues its own template — draw again for another.
            addTemplateToList({ static_cast<int32_t>(sp_tmpl_trace->value()),
                                s, std::max<int64_t>(2, e - s) });
        } else {
            sp_s_first->setValue(static_cast<int>(s));
            sp_s_count->setValue(static_cast<int>(std::max<int64_t>(2, e - s)));
        }
        *draw_target = DrawTarget::None;
        plotWidget()->setMode(InteractionMode::Pan);
        btn_draw_tmpl->setText("Draw on plot →");
        btn_draw_search->setText("Draw on plot →");
        btn_draw_tmpl->setEnabled(true);
        btn_draw_search->setEnabled(true);
        cfg->raise();
        cfg->activateWindow();
    });

    // Restore Pan mode (and bring the dialog back) if it's closed/hidden mid-draw
    connect(cfg, &QDialog::finished, cfg, [=](int) {
        if (plotWidget()->mode() == InteractionMode::CropSelect)
            plotWidget()->setMode(InteractionMode::Pan);
    });
    // QDialog::accept()/reject() call done(), which hide()s rather than close()s
    // the dialog — WA_DeleteOnClose alone would never fire, so clean up explicitly.
    connect(cfg, &QDialog::finished, cfg, &QObject::deleteLater);

    // Update output-size / memory estimate labels (accounts for pipeline decimation).
    // Branches on Template Match (n_traces × n_lags) vs. matrix methods (M × M).
    auto updateEstimate = [=]() {
        int64_t ns = sp_s_count->value() == 0 ? n_samples : sp_s_count->value();
        for (const auto& t : activeDs().pipeline) ns = t->transformedCount(ns);
        int st = sp_stride->value();
        double mem_mb;
        if (rb_template->isChecked()) {
            // sp_tmpl_len is already pipeline-processed — unlike ns above,
            // it must NOT be run through transformedCount() again.
            int64_t tmpl_ns = static_cast<int64_t>(sp_tmpl_len->value());
            int64_t n_lags = std::max<int64_t>(1, (ns - tmpl_ns) / std::max(1, st) + 1);
            lbl_M->setText(QString("%1 lags").arg(n_lags));
            mem_mb = static_cast<double>(sp_count->value()) * static_cast<double>(n_lags) * 4.0
                   / (1024.0 * 1024.0);
        } else {
            int64_t M = (ns + st - 1) / st;
            lbl_M->setText(QString::number(M));
            mem_mb = static_cast<double>(M) * M * 4.0 / (1024.0 * 1024.0);
        }
        if (mem_mb >= 1024.0)
            lbl_mem->setText(QString("%1 GB").arg(mem_mb / 1024.0, 0, 'f', 2));
        else
            lbl_mem->setText(QString("%1 MB").arg(mem_mb, 0, 'f', 1));
    };
    connect(sp_stride,    QOverload<int>::of(&QSpinBox::valueChanged), cfg, [=](int) { updateEstimate(); });
    connect(sp_s_count,   QOverload<int>::of(&QSpinBox::valueChanged), cfg, [=](int) { updateEstimate(); });
    connect(sp_count,     QOverload<int>::of(&QSpinBox::valueChanged), cfg, [=](int) { updateEstimate(); });
    connect(sp_tmpl_len,  QOverload<int>::of(&QSpinBox::valueChanged), cfg, [=](int) { updateEstimate(); });
    connect(rb_template,  &QRadioButton::toggled, cfg, [=](bool) { updateEstimate(); });
    updateEstimate();

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(cfg_bb, &QDialogButtonBox::accepted, cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, cfg, &QDialog::reject);

    vl_cfg->addWidget(grp_traces);
    vl_cfg->addWidget(grp_samples);
    vl_cfg->addWidget(grp_align_xcorr);
    vl_cfg->addWidget(grp_ref);
    vl_cfg->addWidget(grp_ds);
    vl_cfg->addWidget(grp_method);
    vl_cfg->addWidget(grp_tmpl);
    vl_cfg->addWidget(cfg_bb);

    // Everything below used to run synchronously after `cfg.exec()` returned
    // Accepted. Now that cfg is non-modal, it runs instead when the user clicks
    // OK, via QDialog::accepted() (emitted from within accept(), so this fires
    // right after cfg_bb's "accepted" -> cfg->accept() connection above runs).
    connect(cfg, &QDialog::accepted, cfg, [=]() {

    int32_t first_trace  = static_cast<int32_t>(sp_first->value());
    int32_t num_traces   = static_cast<int32_t>(sp_count->value());
    int64_t first_sample = static_cast<int64_t>(sp_s_first->value());
    int64_t num_samples_req = static_cast<int64_t>(sp_s_count->value()); // 0 = all
    int32_t stride       = static_cast<int32_t>(sp_stride->value());
    bool    is_twowin    = rb_twowin->isChecked();
    bool    is_template  = rb_template->isChecked();
    double  mp_scale     = sp_mp_scale->value();
    XCorrMethod method   = rb_mp->isChecked()     ? XCorrMethod::MPCleaned
                         : rb_dual->isChecked()    ? XCorrMethod::DualMatrix
                         : is_twowin               ? XCorrMethod::TwoWindow
                         : is_template             ? XCorrMethod::TemplateMatch
                                                   : XCorrMethod::Baseline;

    const bool use_alignment_xcorr = chk_shifts_xcorr->isChecked();
    if (use_alignment_xcorr) {
        first_trace      = activeDs().align_first_trace;
        num_traces       = static_cast<int32_t>(activeDs().align_shifts.size());
        first_sample     = activeDs().align_first_sample;
        num_samples_req  = activeDs().align_n_samples;
    }
    std::vector<int32_t> use_shifts = use_alignment_xcorr ? activeDs().align_shifts : std::vector<int32_t>{};

    // Template Match: the queued list, or (if empty) the currently staged
    // trace/first/length spin-box values as a single implicit template.
    std::vector<TemplateSpec> template_specs;
    if (is_template) {
        for (const auto& s : *template_list) template_specs.push_back(s);
        if (template_specs.empty()) {
            // sp_tmpl_first/sp_tmpl_len are pipeline-processed; convert to
            // raw, same as the "Add to list" button does.
            const auto& pl = activeDs().pipeline;
            int64_t proc_first = static_cast<int64_t>(sp_tmpl_first->value());
            int64_t proc_end    = proc_first + static_cast<int64_t>(sp_tmpl_len->value());
            int64_t raw_first  = rawCountForProcessedCount(pl, proc_first, h.num_samples);
            int64_t raw_end     = rawCountForProcessedCount(pl, proc_end,   h.num_samples);
            template_specs.push_back({ static_cast<int32_t>(sp_tmpl_trace->value()),
                                       raw_first, std::max<int64_t>(1, raw_end - raw_first) });
        }
    }

    // Memory warning for large matrices (effective M accounts for pipeline).
    // For Template Match this is the worst case across all queued templates —
    // each is computed (and shown) one at a time, so memory isn't cumulative.
    {
        int64_t ns = num_samples_req == 0 ? (n_samples - first_sample) : num_samples_req;
        for (const auto& t : activeDs().pipeline) ns = t->transformedCount(ns);
        double mem_mb = 0.0;
        if (is_template) {
            for (const auto& spec : template_specs) {
                int64_t tmpl_ns = spec.len;
                for (const auto& t : activeDs().pipeline) tmpl_ns = t->transformedCount(tmpl_ns);
                int64_t n_lags = std::max<int64_t>(1, (ns - tmpl_ns) / std::max(1, stride) + 1);
                double mb = static_cast<double>(num_traces) * static_cast<double>(n_lags) * 4.0
                          / (1024.0 * 1024.0);
                mem_mb = std::max(mem_mb, mb);
            }
        } else {
            int64_t M = (ns + stride - 1) / stride;
            mem_mb = static_cast<double>(M) * M * 4.0 / (1024.0 * 1024.0);
        }
        if (mem_mb > 2048.0) {
            if (QMessageBox::warning(this, "Memory warning",
                    QString("The largest output matrix will require ~%1 GB.\nContinue?")
                        .arg(mem_mb / 1024.0, 0, 'f', 1),
                    QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
                return;
        }
    }

    // ---- Progress dialog ----
    QProgressDialog prog("Initialising…", "Cancel", 0, 100, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(0);
    prog.setValue(0);
    QApplication::processEvents();

    // ---- Result window (called once per template for Template Match,
    //      once overall for the matrix methods) ----
    auto showResultWindow = [=](XCorrResult result_in, const QString& extra_title) {
    auto result_ptr = std::make_shared<XCorrResult>(std::move(result_in));

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    QString method_str = result_ptr->method == XCorrMethod::MPCleaned      ? "MP-Cleaned"
                       : result_ptr->method == XCorrMethod::DualMatrix      ? "Dual Matrix"
                       : result_ptr->method == XCorrMethod::TwoWindow       ? "Two-Window"
                       : result_ptr->method == XCorrMethod::TemplateMatch   ? "Template Match"
                                                                             : "Baseline";
    QString title = (result_ptr->method == XCorrMethod::TemplateMatch)
        ? QString("Template Match  template_len=%1  traces=%2  lags=%3  lag_stride=%4")
              .arg(result_ptr->tm_template_len).arg(result_ptr->rows).arg(result_ptr->cols)
              .arg(result_ptr->tm_lag_stride)
        : (result_ptr->method == XCorrMethod::TwoWindow)
        ? QString("Two-Window Match  search=%1  ref=%2  n=%3")
              .arg(result_ptr->rows).arg(result_ptr->cols).arg(result_ptr->n_traces)
        : QString("Cross-Correlation [%1]  M=%2  n=%3")
              .arg(method_str).arg(result_ptr->M).arg(result_ptr->n_traces);
    if (!extra_title.isEmpty()) title += "  " + extra_title;
    if (result_ptr->method == XCorrMethod::MPCleaned) {
        title += QString("  λ+=%1").arg(result_ptr->lambda_plus, 0, 'g', 4);
        if (result_ptr->mp_threshold_scale > 1.0)
            title += QString("×%1").arg(result_ptr->mp_threshold_scale, 0, 'f', 2);
        title += QString("  signal=%1").arg(result_ptr->n_signal);
    }
    dlg->setWindowTitle(title);
    dlg->resize(820, 760);

    auto* heatmap = new HeatmapWidget(dlg);
    heatmap->setMatrix(result_ptr->matrix, result_ptr->rows, result_ptr->cols);

    // Controls row
    auto* lbl_hover   = new QLabel("Hover over matrix to inspect values");
    lbl_hover->setTextInteractionFlags(Qt::TextSelectableByMouse);

    auto* lbl_vmin    = new QLabel("Color min:");
    auto* lbl_vmax    = new QLabel("Color max:");
    auto* sp_vmin     = new QDoubleSpinBox;
    auto* sp_vmax     = new QDoubleSpinBox;
    sp_vmin->setRange(-1e9, 1e9); sp_vmin->setDecimals(4); sp_vmin->setValue(-1.0);
    sp_vmax->setRange(-1e9, 1e9); sp_vmax->setDecimals(4); sp_vmax->setValue( 1.0);
    sp_vmin->setSingleStep(0.1);
    sp_vmax->setSingleStep(0.1);

    auto* btn_reset_view     = new QPushButton("Reset View");
    auto* btn_exp_png        = new QPushButton("Export PNG…");
    auto* btn_exp_npy        = new QPushButton("Export matrix .npy…");
    auto* btn_show_traces    = new QPushButton("Show corr traces…");
    auto* btn_exp_corr_trs   = new QPushButton("Export corr traces .trs…");

    // Compute actual data range for sensible default colour bounds
    {
        float dmin =  1e38f, dmax = -1e38f;
        for (float v : result_ptr->matrix) {
            if (v < dmin) dmin = v;
            if (v > dmax) dmax = v;
        }
        float abs_max = std::max(std::abs(dmin), std::abs(dmax));
        sp_vmin->setValue(static_cast<double>(-abs_max));
        sp_vmax->setValue(static_cast<double>( abs_max));
        heatmap->setColorRange(-abs_max, abs_max);
    }

    connect(sp_vmin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setColorRange(static_cast<float>(v), static_cast<float>(sp_vmax->value()));
    });
    connect(sp_vmax, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setColorRange(static_cast<float>(sp_vmin->value()), static_cast<float>(v));
    });
    connect(btn_reset_view, &QPushButton::clicked, heatmap, &HeatmapWidget::resetView);

    connect(heatmap, &HeatmapWidget::hoverInfo, dlg, [lbl_hover, result_ptr](int s1, int s2, float val) {
        if (result_ptr->method == XCorrMethod::TemplateMatch) {
            int64_t abs_sample = result_ptr->tm_search_first_sample
                               + static_cast<int64_t>(s2) * result_ptr->tm_lag_stride;
            lbl_hover->setText(
                QString("Trace %1  lag %2 (sample %3)  NCC = %4").arg(s1).arg(s2).arg(abs_sample)
                    .arg(static_cast<double>(val), 0, 'g', 6));
        } else {
            lbl_hover->setText(
                QString("C[%1, %2] = %3").arg(s1).arg(s2)
                    .arg(static_cast<double>(val), 0, 'g', 6));
        }
    });

    // Processing controls
    auto* lbl_scheme2   = new QLabel("Color scheme:");
    auto* combo_scheme2 = new QComboBox;
    combo_scheme2->addItems({"RdBu", "Grayscale", "Hot", "Viridis", "Plasma", "Lukasz"});
    connect(combo_scheme2, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int idx) {
        heatmap->setColorScheme(static_cast<ColorScheme>(idx));
    });

    auto* lbl_sigma2  = new QLabel("Gaussian σ:");
    auto* sp_sigma2   = new QDoubleSpinBox;
    sp_sigma2->setRange(0.0, 50.0); sp_sigma2->setDecimals(1); sp_sigma2->setSingleStep(0.5);
    sp_sigma2->setValue(0.0); sp_sigma2->setSpecialValueText("off");
    connect(sp_sigma2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setGaussianSigma(static_cast<float>(v));
    });

    auto* chk_abs2   = new QCheckBox("Abs value");
    connect(chk_abs2, &QCheckBox::toggled, [=](bool on) {
        heatmap->setAbsValue(on);
        if (on) {
            sp_vmin->setValue(0.0);
            heatmap->setColorRange(0.0f, static_cast<float>(sp_vmax->value()));
        } else {
            double vm = sp_vmax->value();
            sp_vmin->setValue(-vm);
            heatmap->setColorRange(static_cast<float>(-vm), static_cast<float>(vm));
        }
    });

    // Two-Window / Template Match: default to Lukasz colormap + abs value for
    // template-match look
    if (result_ptr->method == XCorrMethod::TwoWindow ||
        result_ptr->method == XCorrMethod::TemplateMatch) {
        combo_scheme2->setCurrentIndex(5);  // Lukasz (black → neon green)
        chk_abs2->setChecked(true);         // abs: collapses to [0,1], snaps vmin→0
    }

    auto* lbl_gamma2 = new QLabel("Power γ:");
    auto* sp_gamma2  = new QDoubleSpinBox;
    sp_gamma2->setRange(1.0, 10.0); sp_gamma2->setDecimals(2); sp_gamma2->setSingleStep(0.1);
    sp_gamma2->setValue(1.0); sp_gamma2->setSpecialValueText("off");
    connect(sp_gamma2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        heatmap->setPowerGamma(static_cast<float>(v));
    });

    auto* chk_thresh2 = new QCheckBox("Binary threshold |v|≥");
    auto* sp_thresh2  = new QDoubleSpinBox;
    sp_thresh2->setRange(0.0, 1e9); sp_thresh2->setDecimals(4);
    sp_thresh2->setSingleStep(0.05); sp_thresh2->setValue(0.5);
    sp_thresh2->setEnabled(false);
    connect(chk_thresh2, &QCheckBox::toggled, [=](bool on) {
        sp_thresh2->setEnabled(on);
        heatmap->setBinaryThreshold(on, static_cast<float>(sp_thresh2->value()));
    });
    connect(sp_thresh2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [=](double v) {
        if (chk_thresh2->isChecked())
            heatmap->setBinaryThreshold(true, static_cast<float>(v));
    });

    auto* btn_autoclip2 = new QPushButton("Auto-clip 98%");
    connect(btn_autoclip2, &QPushButton::clicked, dlg, [=]() {
        float cmin, cmax;
        heatmap->computeClipRange(0.98f, cmin, cmax);
        sp_vmin->setValue(static_cast<double>(cmin));
        sp_vmax->setValue(static_cast<double>(cmax));
        heatmap->setColorRange(cmin, cmax);
    });

    connect(btn_exp_png, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export heatmap as PNG",
                                                    MainWindow::recentDir("export"), "PNG images (*.png)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        if (!heatmap->exportPng(path))
            QMessageBox::critical(dlg, "Export failed", "Could not write:\n" + path);
        else
            QMessageBox::information(dlg, "Saved", "Saved: " + path);
    });

    connect(btn_exp_npy, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export matrix as NumPy",
                                                    MainWindow::recentDir("npy"), "NumPy files (*.npy)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("npy", path);
        // Write 2-D float32 .npy
        FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
        if (!fp) { QMessageBox::critical(dlg, "Export failed", "Cannot create:\n" + path); return; }
        const uint8_t magic[] = {0x93,'N','U','M','P','Y',0x01,0x00};
        std::fwrite(magic, 1, 8, fp);
        int32_t rows = result_ptr->rows, cols = result_ptr->cols;
        std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                           std::to_string(rows) + ", " + std::to_string(cols) + "), }";
        size_t content_len = dict.size() + 1;
        size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
        dict.resize(header_len - 1, ' ');
        dict += '\n';
        uint16_t hl = static_cast<uint16_t>(header_len);
        uint8_t hl_bytes[2] = {uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
        std::fwrite(hl_bytes, 1, 2, fp);
        std::fwrite(dict.c_str(), 1, dict.size(), fp);
        std::fwrite(result_ptr->matrix.data(), sizeof(float),
                    static_cast<size_t>(rows) * static_cast<size_t>(cols), fp);
        std::fclose(fp);
        QMessageBox::information(dlg, "Saved", "Saved: " + path);
    });

    // ── Show correlation traces in a PlotWidget ───────────────────────────
    connect(btn_show_traces, &QPushButton::clicked, dlg, [=]() {
        int32_t rows = result_ptr->rows;
        int32_t cols = result_ptr->cols;
        if (rows <= 0 || cols <= 0) return;

        const bool is_tm = (result_ptr->method == XCorrMethod::TemplateMatch);

        // Rows can number in the thousands (M for matrix methods, n_traces for
        // Template Match) and only NUM_COLORS distinct colours exist — ask which
        // slice to actually plot instead of dumping every row into one view.
        QDialog pick(dlg);
        pick.setWindowTitle("Select rows to plot");
        auto* pfl = new QFormLayout(&pick);
        auto* lbl_avail = new QLabel(is_tm
            ? QString("%1 traces available (trace %2..%3)")
                  .arg(rows).arg(first_trace).arg(first_trace + rows - 1)
            : QString("%1 rows available").arg(rows));
        pfl->addRow(lbl_avail);
        auto* sp_pick_first = new QSpinBox;
        sp_pick_first->setRange(0, rows - 1);
        sp_pick_first->setValue(0);
        auto* sp_pick_count = new QSpinBox;
        sp_pick_count->setRange(1, rows);
        sp_pick_count->setValue(std::min(rows, NUM_COLORS));
        sp_pick_count->setToolTip(QString("Only %1 distinct colours are used; they repeat past that many rows.")
                                       .arg(NUM_COLORS));
        pfl->addRow(is_tm ? "First trace row:" : "First row:", sp_pick_first);
        pfl->addRow("Count:", sp_pick_count);
        auto* pick_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        connect(pick_bb, &QDialogButtonBox::accepted, &pick, &QDialog::accept);
        connect(pick_bb, &QDialogButtonBox::rejected, &pick, &QDialog::reject);
        pfl->addRow(pick_bb);
        if (pick.exec() != QDialog::Accepted) return;

        int32_t sel_first = static_cast<int32_t>(sp_pick_first->value());
        int32_t sel_count = std::min(static_cast<int32_t>(sp_pick_count->value()), rows - sel_first);

        auto* tdlg = new QDialog(dlg);
        tdlg->setWindowTitle(is_tm
            ? QString("Match curves — %1 of %2 traces × %3 lags").arg(sel_count).arg(rows).arg(cols)
            : QString("Correlation traces — %1 of %2 rows × %3 samples").arg(sel_count).arg(rows).arg(cols));
        tdlg->setAttribute(Qt::WA_DeleteOnClose);
        auto* tvl = new QVBoxLayout(tdlg);
        auto* pw  = new PlotWidget(tdlg);
        if (is_tm)
            pw->setAxisLabels("Lag (samples)", "NCC");
        tvl->addWidget(pw);

        const float* mat = result_ptr->matrix.data();
        for (int32_t i = sel_first; i < sel_first + sel_count; i++) {
            auto trace = std::make_shared<std::vector<float>>(
                mat + static_cast<ptrdiff_t>(i) * cols,
                mat + static_cast<ptrdiff_t>(i) * cols + cols);
            pw->addTrace(std::move(trace),
                         TRACE_COLORS[(i - sel_first) % NUM_COLORS],
                         is_tm ? QString("Trace %1").arg(first_trace + i)
                               : QString("C[%1,:]").arg(i));
        }
        pw->resetView();
        tdlg->resize(1100, 500);
        tdlg->show();
    });

    // ── Export correlation traces as TRS ─────────────────────────────────
    connect(btn_exp_corr_trs, &QPushButton::clicked, dlg, [=]() {
        int32_t rows = result_ptr->rows;
        int32_t cols = result_ptr->cols;
        if (rows <= 0 || cols <= 0) return;

        QString path = QFileDialog::getSaveFileName(dlg, "Export correlation traces as TRS",
                                                    MainWindow::recentDir("trs"), "TRS files (*.trs);;All files (*)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("trs", path);

        FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
        if (!fp) {
            QMessageBox::critical(dlg, "Export failed", "Cannot create:\n" + path);
            return;
        }
        // Write TRS header
        auto wle32 = [&](int32_t v) {
            uint8_t b[4] = { uint8_t(v), uint8_t(v>>8), uint8_t(v>>16), uint8_t(v>>24) };
            std::fwrite(b, 1, 4, fp);
        };
        std::fputc(0x41, fp); std::fputc(4, fp); wle32(rows);  // NUMBER_TRACES
        std::fputc(0x42, fp); std::fputc(4, fp); wle32(cols);  // NUMBER_SAMPLES
        std::fputc(0x43, fp); std::fputc(1, fp); std::fputc(0x14, fp); // SAMPLE_CODING: float32
        std::fputc(0x5F, fp); std::fputc(0, fp);               // TRACE_BLOCK
        // Write trace data (each row of the matrix is one trace)
        std::fwrite(result_ptr->matrix.data(), sizeof(float),
                    static_cast<size_t>(rows) * static_cast<size_t>(cols), fp);
        std::fclose(fp);
        QMessageBox::information(dlg, "Saved", "Saved: " + path);
    });

    auto* ctrl = new QWidget(dlg);
    auto* ctrl_l = new QHBoxLayout(ctrl);
    ctrl_l->setContentsMargins(4, 2, 4, 2);
    ctrl_l->addWidget(lbl_hover, 1);
    ctrl_l->addWidget(lbl_vmin);
    ctrl_l->addWidget(sp_vmin);
    ctrl_l->addWidget(lbl_vmax);
    ctrl_l->addWidget(sp_vmax);
    ctrl_l->addWidget(btn_autoclip2);
    ctrl_l->addWidget(btn_reset_view);
    ctrl_l->addWidget(btn_exp_png);
    ctrl_l->addWidget(btn_exp_npy);
    ctrl_l->addWidget(btn_show_traces);
    ctrl_l->addWidget(btn_exp_corr_trs);

    auto* proc_row  = new QWidget(dlg);
    auto* proc_row_l = new QHBoxLayout(proc_row);
    proc_row_l->setContentsMargins(4, 2, 4, 2);
    proc_row_l->addWidget(lbl_scheme2);
    proc_row_l->addWidget(combo_scheme2);
    proc_row_l->addSpacing(12);
    proc_row_l->addWidget(lbl_sigma2);
    proc_row_l->addWidget(sp_sigma2);
    proc_row_l->addSpacing(8);
    proc_row_l->addWidget(chk_abs2);
    proc_row_l->addSpacing(8);
    proc_row_l->addWidget(lbl_gamma2);
    proc_row_l->addWidget(sp_gamma2);
    proc_row_l->addSpacing(8);
    proc_row_l->addWidget(chk_thresh2);
    proc_row_l->addWidget(sp_thresh2);
    proc_row_l->addStretch();

    // Info bar (λ+, n_signal for MP-Cleaned)
    QWidget* info_bar = nullptr;
    if (result_ptr->method == XCorrMethod::MPCleaned) {
        info_bar = new QWidget(dlg);
        auto* il = new QHBoxLayout(info_bar);
        il->setContentsMargins(4, 0, 4, 0);
        {
            QString thr_str = QString("λ+ = <b>%1</b>").arg(result_ptr->lambda_plus, 0, 'g', 5);
            if (result_ptr->mp_threshold_scale > 1.0)
                thr_str += QString("  ×<b>%1</b> (threshold = <b>%2</b>)")
                               .arg(result_ptr->mp_threshold_scale, 0, 'f', 2)
                               .arg(result_ptr->lambda_plus * result_ptr->mp_threshold_scale, 0, 'g', 5);
            thr_str += QString("    Signal eigenvalues above threshold: <b>%1</b>")
                           .arg(result_ptr->n_signal);
            il->addWidget(new QLabel(thr_str));
        }
        auto* l = qobject_cast<QLabel*>(il->itemAt(0)->widget());
        if (l) l->setTextFormat(Qt::RichText);
        il->addStretch();
    }

    // ---- Template Match: project match peaks back onto the main trace plot ----
    // The trivial "best match" is the template matching its own source location
    // (NCC=1) — not useful. This instead finds every recurrence of the pattern
    // above a threshold and highlights each one on the main plot as an amber
    // marker, since the same operation can repeat many times in a trace.
    QWidget* peak_row = nullptr;
    if (result_ptr->method == XCorrMethod::TemplateMatch) {
        peak_row = new QWidget(dlg);
        auto* pl = new QHBoxLayout(peak_row);
        pl->setContentsMargins(4, 2, 4, 2);

        auto* sp_peak_trace = new QSpinBox;
        sp_peak_trace->setRange(0, std::max(0, result_ptr->rows - 1));
        sp_peak_trace->setValue(0);
        sp_peak_trace->setToolTip("Row (trace) in this result whose match curve is scanned for peaks.");

        auto* sp_peak_thr = new QDoubleSpinBox;
        sp_peak_thr->setRange(-1.0, 1.0);
        sp_peak_thr->setDecimals(2);
        sp_peak_thr->setSingleStep(0.05);
        sp_peak_thr->setValue(0.8);
        sp_peak_thr->setToolTip("Local maxima of NCC at or above this value are projected onto\n"
                                "the main trace plot as amber markers — finds every recurrence\n"
                                "of the pattern, not just the trivial self-match.");

        auto* chk_peak_show  = new QCheckBox("Show on main plot");
        auto* lbl_peak_count = new QLabel;

        auto updatePeaks = [=]() {
            if (!chk_peak_show->isChecked()) {
                plotWidget()->clearMatchMarkers();
                lbl_peak_count->setText(QString());
                return;
            }
            int32_t ti = sp_peak_trace->value();
            if (ti < 0 || ti >= result_ptr->rows) return;
            const float* row = result_ptr->matrix.data()
                             + static_cast<size_t>(ti) * static_cast<size_t>(result_ptr->cols);
            float   thr     = static_cast<float>(sp_peak_thr->value());
            // Minimum spacing between accepted peaks, in lag columns: one
            // template length, so a single true hit isn't reported many times.
            int64_t min_sep = std::max<int64_t>(1, result_ptr->tm_template_len / result_ptr->tm_lag_stride);

            std::vector<std::pair<int64_t,int64_t>> ranges;
            int64_t lag = 0;
            while (lag < result_ptr->cols) {
                if (row[lag] >= thr) {
                    int64_t best = lag;
                    int64_t end  = std::min<int64_t>(result_ptr->cols, lag + min_sep);
                    for (int64_t j = lag + 1; j < end; j++)
                        if (row[j] > row[best]) best = j;
                    int64_t abs_start = result_ptr->tm_search_first_sample
                                       + best * result_ptr->tm_lag_stride;
                    ranges.push_back({abs_start, abs_start + result_ptr->tm_template_len});
                    lag = best + min_sep;
                } else {
                    lag++;
                }
            }
            plotWidget()->setMatchMarkers(ranges);
            lbl_peak_count->setText(QString("%1 match%2 found")
                .arg(ranges.size()).arg(ranges.size() == 1 ? "" : "es"));
        };

        connect(sp_peak_trace, QOverload<int>::of(&QSpinBox::valueChanged),    dlg, [=](int)    { updatePeaks(); });
        connect(sp_peak_thr,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), dlg, [=](double) { updatePeaks(); });
        connect(chk_peak_show, &QCheckBox::toggled,                            dlg, [=](bool)   { updatePeaks(); });

        // Clear this window's markers when it closes, but only if it was the
        // one showing them — avoids stomping on another still-open result window.
        connect(dlg, &QObject::destroyed, this, [this, chk_peak_show]() {
            if (chk_peak_show->isChecked()) plotWidget()->clearMatchMarkers();
        });

        pl->addWidget(new QLabel("Trace:"));
        pl->addWidget(sp_peak_trace);
        pl->addSpacing(8);
        pl->addWidget(new QLabel("Threshold ≥:"));
        pl->addWidget(sp_peak_thr);
        pl->addSpacing(8);
        pl->addWidget(chk_peak_show);
        pl->addWidget(lbl_peak_count);
        pl->addStretch();
    }

    auto* vl = new QVBoxLayout(dlg);
    vl->setContentsMargins(4, 4, 4, 4);
    vl->setSpacing(4);
    if (info_bar) vl->addWidget(info_bar);
    if (peak_row) vl->addWidget(peak_row);
    vl->addWidget(ctrl);
    vl->addWidget(proc_row);
    vl->addWidget(heatmap, 1);

    dlg->show();
    };

    // ---- Computation ----
    std::string err;

    if (is_template) {
        int64_t ns = (num_samples_req == 0) ? (h.num_samples - first_sample) : num_samples_req;
        const int n_tmpl = static_cast<int>(template_specs.size());
        for (int ti = 0; ti < n_tmpl; ti++) {
            const auto& spec = template_specs[static_cast<size_t>(ti)];
            QString prefix = n_tmpl > 1 ? QString("Template %1/%2 — ").arg(ti + 1).arg(n_tmpl) : QString();

            auto progCb = [&](int32_t done, int32_t total) -> bool {
                if (prog.wasCanceled()) return false;
                prog.setMaximum(total);
                prog.setValue(done);
                prog.setLabelText(
                    total > 0 ? prefix + QString("Processing trace %1 / %2…").arg(done).arg(total)
                              : prefix + QString("Processing…"));
                QApplication::processEvents();
                return true;
            };

            XCorrResult result;
            bool ok = computeTemplateMatch(
                activeDs().file.get(), spec.trace, spec.first, spec.len,
                first_trace, num_traces,
                first_sample, ns,
                stride, activeDs().pipeline, use_shifts, result, progCb, err);
            prog.setValue(prog.maximum());

            if (!ok) {
                if (!err.empty())
                    QMessageBox::critical(this, "Cross-Correlation failed",
                                          QString::fromStdString(err));
                return;  // cancelled / error — abort remaining templates too
            }

            QString extra = n_tmpl > 1
                ? QString("[template %1/%2: trace %3, %4:%5]")
                      .arg(ti + 1).arg(n_tmpl).arg(spec.trace).arg(spec.first).arg(spec.first + spec.len)
                : QString();
            showResultWindow(std::move(result), extra);
        }
    } else {
        auto progCb = [&](int32_t done, int32_t total) -> bool {
            if (prog.wasCanceled()) return false;
            prog.setMaximum(total);
            prog.setValue(done);
            prog.setLabelText(
                total > 0 ? QString("Processing trace %1 / %2…").arg(done).arg(total)
                          : QString("Processing…"));
            QApplication::processEvents();
            return true;
        };

        XCorrResult result;
        bool ok;
        if (is_twowin) {
            // sp_r_first/sp_r_count are pipeline-processed; convert to raw.
            const auto& pl = activeDs().pipeline;
            int64_t proc_ref_first = static_cast<int64_t>(sp_r_first->value());
            int64_t proc_ref_end    = proc_ref_first + static_cast<int64_t>(sp_r_count->value());
            int64_t ref_first = rawCountForProcessedCount(pl, proc_ref_first, h.num_samples);
            int64_t ref_end    = rawCountForProcessedCount(pl, proc_ref_end,   h.num_samples);
            int64_t ref_count = std::max<int64_t>(1, ref_end - ref_first);
            int64_t ns = (num_samples_req == 0) ? (h.num_samples - first_sample) : num_samples_req;
            ok = computeTwoWindowCorr(
                activeDs().file.get(), first_trace, num_traces,
                ref_first, ref_count,
                first_sample, ns,
                stride, activeDs().pipeline, use_shifts, result, progCb, err);
        } else {
            ok = computeXCorr(
                activeDs().file.get(),
                first_trace, num_traces,
                first_sample, num_samples_req,
                stride, method, activeDs().pipeline, use_shifts, result, progCb, err,
                mp_scale);
        }
        prog.setValue(prog.maximum());

        if (!ok) {
            if (!err.empty())
                QMessageBox::critical(this, "Cross-Correlation failed",
                                      QString::fromStdString(err));
            return;  // cancelled
        }
        showResultWindow(std::move(result), QString());
    }
    });

    cfg->show();
}

// ---------------------------------------------------------------------------

std::shared_ptr<ITransform> MainWindow::createTransform(int idx) {
    switch (idx) {
    case 0: return std::make_shared<AbsTransform>();
    case 1: return std::make_shared<NegateTransform>();
    case 2: {
        bool ok;
        int w = QInputDialog::getInt(this, "Moving Average",
                                     "Window size (samples):",
                                     64, 2, 1'000'000, 1, &ok);
        if (!ok) return nullptr;
        return std::make_shared<MovingAverageTransform>(w);
    }
    case 3: {
        QDialog d(this);
        d.setWindowTitle("Window Resample — parameters");
        auto* fl = new QFormLayout(&d);

        auto* sp_win = new QSpinBox; sp_win->setRange(2, 1'000'000); sp_win->setValue(64);
        sp_win->setToolTip("Samples averaged into each output point.");
        auto* sp_overlap = new QDoubleSpinBox;
        sp_overlap->setRange(0.0, 1.0);
        sp_overlap->setSingleStep(0.05);
        sp_overlap->setDecimals(2);
        sp_overlap->setValue(0.0);
        sp_overlap->setToolTip("Fraction of each window reused by the next one (0 = "
                               "non-overlapping blocks, closer to 1 = windows slide by "
                               "just a few samples at a time).");

        fl->addRow("Window size (samples):", sp_win);
        fl->addRow("Overlap (0-1):",          sp_overlap);

        // Show approximate output size as the user adjusts parameters.
        auto* lbl_out = new QLabel;
        fl->addRow("Output samples/trace:", lbl_out);
        int n_samples = activeDs().file ? activeDs().file->header().num_samples : 0;
        auto updateOut = [&]() {
            int W = sp_win->value();
            int hop = std::max(1, static_cast<int>(
                std::llround(W * (1.0 - sp_overlap->value()))));
            if (n_samples >= W) {
                int64_t nw = (n_samples - W) / hop + 1;
                lbl_out->setText(QString("%1  (hop = %2 samples)").arg(nw).arg(hop));
            } else {
                lbl_out->setText("Window larger than trace — no output");
            }
        };
        updateOut();
        connect(sp_win, QOverload<int>::of(&QSpinBox::valueChanged), [&](int){ updateOut(); });
        connect(sp_overlap, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [&](double){ updateOut(); });

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        fl->addRow(bb);
        connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
        if (d.exec() != QDialog::Accepted) return nullptr;

        return std::make_shared<WindowResampleTransform>(
            sp_win->value(), static_cast<float>(sp_overlap->value()));
    }
    case 4: {
        bool ok;
        int s = QInputDialog::getInt(this, "Stride Resample",
                                     "Stride (keep every Nth sample):",
                                     4, 2, 1'000'000, 1, &ok);
        if (!ok) return nullptr;
        return std::make_shared<StrideResampleTransform>(s);
    }
    case 5: {
        bool ok;
        double v = QInputDialog::getDouble(this, "Offset",
                                           "Value added to every sample:",
                                           0.0, -1e9, 1e9, 6, &ok);
        if (!ok) return nullptr;
        return std::make_shared<OffsetTransform>(static_cast<float>(v));
    }
    case 6: {
        bool ok;
        double v = QInputDialog::getDouble(this, "Scale",
                                           "Factor multiplied with every sample:",
                                           1.0, -1e9, 1e9, 6, &ok);
        if (!ok) return nullptr;
        return std::make_shared<ScaleTransform>(static_cast<float>(v));
    }
    case 7: {
        QStringList wins = { "Rectangular", "Hann", "Hamming", "Blackman" };
        bool ok;
        QString choice = QInputDialog::getItem(this, "FFT Magnitude",
                                               "Window function:", wins, 1, false, &ok);
        if (!ok) return nullptr;
        using W = FFTMagnitudeTransform::Window;
        W win = W::Hann;
        if      (choice == "Rectangular") win = W::Rectangular;
        else if (choice == "Hamming")     win = W::Hamming;
        else if (choice == "Blackman")    win = W::Blackman;
        return std::make_shared<FFTMagnitudeTransform>(win);
    }
    case 8: {
        QDialog d(this);
        d.setWindowTitle("STFT Magnitude — parameters");
        auto* fl = new QFormLayout(&d);

        auto* sp_win = new QSpinBox; sp_win->setRange(4, 1<<20); sp_win->setValue(256);
        sp_win->setToolTip("Samples per FFT window. Powers of 2 are fastest.");
        auto* sp_overlap = new QDoubleSpinBox;
        sp_overlap->setRange(0.0, 1.0);
        sp_overlap->setSingleStep(0.05);
        sp_overlap->setDecimals(2);
        sp_overlap->setValue(0.5);
        sp_overlap->setToolTip("Fraction of each window reused by the next one "
                               "(hop = window × (1 - overlap)).");
        auto* cmb = new QComboBox;
        cmb->addItems({ "Rectangular", "Hann", "Hamming", "Blackman" });
        cmb->setCurrentIndex(1);

        fl->addRow("Window size (samples):", sp_win);
        fl->addRow("Overlap (0-1):",          sp_overlap);
        fl->addRow("Window function:",       cmb);

        // Show the derived hop size and approximate output size as the user
        // adjusts parameters.
        auto* lbl_out = new QLabel;
        fl->addRow("Output bins/trace:", lbl_out);
        int n_samples = activeDs().file ? activeDs().file->header().num_samples : 0;
        auto hopFromOverlap = [&]() {
            int W = sp_win->value();
            return std::max(1, static_cast<int>(std::llround(W * (1.0 - sp_overlap->value()))));
        };
        auto updateOut = [&]() {
            int W = sp_win->value(), H = hopFromOverlap();
            if (n_samples >= W) {
                int64_t nw = (n_samples - W) / H + 1;
                int64_t nb = W / 2 + 1;
                lbl_out->setText(QString("%1 windows × %2 bins = %3 total  (hop = %4 samples)")
                                     .arg(nw).arg(nb).arg(nw * nb).arg(H));
            } else {
                lbl_out->setText("Window larger than trace — no output");
            }
        };
        updateOut();
        connect(sp_win, QOverload<int>::of(&QSpinBox::valueChanged), [&](int){ updateOut(); });
        connect(sp_overlap, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [&](double){ updateOut(); });

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        fl->addRow(bb);
        connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
        if (d.exec() != QDialog::Accepted) return nullptr;

        using W2 = STFTMagnitudeTransform::Window;
        W2 win = W2::Hann;
        switch (cmb->currentIndex()) {
            case 0: win = W2::Rectangular; break;
            case 2: win = W2::Hamming;     break;
            case 3: win = W2::Blackman;    break;
            default: break;
        }
        return std::make_shared<STFTMagnitudeTransform>(sp_win->value(), hopFromOverlap(), win);
    }
    case 9: {
        bool ok;
        double ns = QInputDialog::getDouble(this, "Gaussian Noise",
                                            "Noise level (fraction of per-trace std):\n"
                                            "e.g. 0.1 adds noise with σ = 10 % of the trace's own std",
                                            0.1, 0.0, 1e6, 4, &ok);
        if (!ok) return nullptr;
        return std::make_shared<GaussianNoiseTransform>(static_cast<float>(ns));
    }
    case 10: {
        QDialog d(this);
        d.setWindowTitle("Filter — parameters");
        auto* fl = new QFormLayout(&d);

        auto* cmb_type = new QComboBox;
        cmb_type->addItems({ "Lowpass", "Highpass", "Bandpass", "Notch" });
        fl->addRow("Type:", cmb_type);

        auto* sp_cutoff = new QDoubleSpinBox;
        sp_cutoff->setRange(0.001, 0.999);
        sp_cutoff->setSingleStep(0.01);
        sp_cutoff->setDecimals(3);
        sp_cutoff->setValue(0.1);
        sp_cutoff->setToolTip("Fraction of Nyquist (half the sample rate) — TRS files "
                              "don't reliably carry a real sample rate, so cutoff is "
                              "expressed relative to it instead of Hz.\n"
                              "0.1 = 10% of Nyquist. Lower = more smoothing (lowpass) "
                              "or more high-frequency content removed (highpass).");
        fl->addRow("Cutoff (× Nyquist, 0-1):", sp_cutoff);

        auto* sp_q = new QDoubleSpinBox;
        sp_q->setRange(0.05, 20.0);
        sp_q->setSingleStep(0.1);
        sp_q->setDecimals(2);
        sp_q->setValue(0.707);
        sp_q->setToolTip("Lowpass/Highpass: 0.707 (1/√2) is the maximally-flat "
                         "response; higher peaks near the cutoff.\n"
                         "Bandpass/Notch: higher Q = narrower band/notch.");
        fl->addRow("Q:", sp_q);

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        fl->addRow(bb);
        connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
        if (d.exec() != QDialog::Accepted) return nullptr;

        using FT = BiquadFilterTransform::FilterType;
        FT type = FT::Lowpass;
        switch (cmb_type->currentIndex()) {
            case 1: type = FT::Highpass; break;
            case 2: type = FT::Bandpass; break;
            case 3: type = FT::Notch;    break;
            default: break;
        }
        return std::make_shared<BiquadFilterTransform>(
            type, static_cast<float>(sp_cutoff->value()), static_cast<float>(sp_q->value()));
    }
    default: return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Builds up to max_display aligned traces (raw window per shifts[i]/
// output_mode, pipeline applied) onto pw. Shared by Align Traces' "Show in
// New Window"/"Apply to Main View" buttons and the Chain "Align" step.
// output_mode: 0=avg-pad, 1=zero-pad, 2=crop to common range.
// Returns false (and warns on msg_parent) if crop mode leaves no common
// range, or a large-allocation warning is declined.
// ---------------------------------------------------------------------------
bool MainWindow::buildAlignedTraces(PlotWidget* pw, const std::vector<int32_t>& shifts,
                                     int32_t first_tr, int output_mode, int max_display,
                                     QWidget* msg_parent)
{
    const TrsHeader& h = activeDs().file->header();
    int32_t num_tr = static_cast<int32_t>(shifts.size());

    int64_t out_start, out_len;
    if (output_mode == 2) {
        int64_t crop_start = 0;
        int64_t crop_end   = h.num_samples;
        for (int i = 0; i < num_tr; i++) {
            int64_t s = shifts[static_cast<size_t>(i)];
            if (s == kAlignDiscardShift) continue;   // excluded — ignore for crop range
            crop_start = std::max(crop_start, -s);
            crop_end   = std::min(crop_end, h.num_samples - s);
        }
        if (crop_end <= crop_start) {
            QMessageBox::warning(msg_parent, "Align Traces",
                "No common valid range after cropping (shifts too large).");
            return false;
        }
        out_start = crop_start;
        out_len   = crop_end - crop_start;
    } else {
        out_start = 0;
        out_len   = h.num_samples;
    }

    // Effective sample count after the processing pipeline — an expanding
    // stage (e.g. STFT) needs a larger buffer than out_len.
    int64_t effective_len = out_len;
    for (const auto& t : activeDs().pipeline)
        effective_len = t->transformedCount(effective_len);

    // Memory estimate: an expanding pipeline stage (e.g. STFT) combined with
    // "Full trace" mode (out_len = the entire raw trace) can need far more
    // memory than the raw file itself, especially for "Show in New Window"
    // (and the Chain step) which — unlike "Apply to Main View" — has no cap
    // on how many traces it builds. Warn before allocating instead of
    // crashing partway through.
    {
        int64_t n_kept = 0;
        for (int i = 0; i < num_tr; i++)
            if (shifts[static_cast<size_t>(i)] != kAlignDiscardShift) n_kept++;
        int64_t n_to_build = std::min<int64_t>(n_kept, max_display);
        int64_t bytes_per_trace = std::max(out_len, effective_len)
                                 * static_cast<int64_t>(sizeof(float));
        int64_t total_bytes = bytes_per_trace * n_to_build;
        if (total_bytes > 2LL * 1024 * 1024 * 1024) {
            if (QMessageBox::warning(msg_parent, "Memory warning",
                    QString("Building %1 aligned trace(s) will require ~%2 GB.\nContinue?")
                        .arg(n_to_build)
                        .arg(double(total_bytes) / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1),
                    QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
                return false;
        }
    }

    int shown = 0;
    for (int i = 0; i < num_tr && shown < max_display; i++) {
        int64_t shift = static_cast<int64_t>(shifts[static_cast<size_t>(i)]);
        if (shift == kAlignDiscardShift) continue;   // below correlation threshold

        auto data = std::make_shared<std::vector<float>>(
            static_cast<size_t>(std::max(out_len, effective_len)), 0.0f);

        int64_t raw_start = out_start + shift;
        int64_t raw_end   = raw_start + out_len;
        int64_t src_start = std::max<int64_t>(0, raw_start);
        int64_t src_end   = std::min<int64_t>(h.num_samples, raw_end);
        int64_t dst_off   = src_start - raw_start;

        if (src_start < src_end)
            activeDs().file->readSamples(first_tr + i, src_start,
                                   src_end - src_start,
                                   data->data() + static_cast<size_t>(dst_off));

        if (output_mode == 0) {
            int64_t valid = src_end - src_start;
            if (valid > 0) {
                double sum = 0.0;
                const float* vp = data->data() + static_cast<size_t>(dst_off);
                for (int64_t j = 0; j < valid; j++) sum += vp[j];
                float avg = static_cast<float>(sum / valid);
                if (dst_off > 0)
                    std::fill(data->begin(),
                              data->begin() + static_cast<size_t>(dst_off), avg);
                int64_t tail_off = dst_off + valid;
                if (tail_off < out_len)
                    std::fill(data->begin() + static_cast<size_t>(tail_off),
                              data->begin() + static_cast<size_t>(out_len), avg);
            }
        }

        // Apply the processing pipeline so the displayed/baked trace is the
        // aligned *result of the pipeline* — the same processed signal the
        // alignment search itself matched against, not the raw trace.
        for (const auto& t : activeDs().pipeline) t->reset();
        int64_t n_out = out_len;
        for (const auto& t : activeDs().pipeline)
            n_out = t->apply(data->data(), n_out, 0);
        data->resize(static_cast<size_t>(std::max<int64_t>(0, n_out)));

        pw->addTrace(std::move(data),
                     TRACE_COLORS[shown % NUM_COLORS],
                     QString("T%1 (%2%3)")
                         .arg(first_tr + i)
                         .arg(shift >= 0 ? "+" : "")
                         .arg(shift));
        shown++;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Commits shifts as the active dataset's alignment and bakes a NUM_COLORS
// preview into the main plot — exactly what Align Traces' "Apply to Main
// View" does, factored out so the Chain "Align" step can reuse it after
// running alignByPeak/alignByXCorr itself.
// ---------------------------------------------------------------------------
bool MainWindow::computeAndStoreAlignment(const std::vector<int32_t>& shifts, int32_t first_tr,
                                           int output_mode, QWidget* msg_parent, QString& err)
{
    const TrsHeader& h   = activeDs().file->header();
    const int32_t num_tr = static_cast<int32_t>(shifts.size());

    int64_t out_start = 0, out_len = h.num_samples;
    if (output_mode == 2) {
        int64_t crop_start = 0, crop_end = h.num_samples;
        for (int i = 0; i < num_tr; i++) {
            int32_t sraw = shifts[static_cast<size_t>(i)];
            if (sraw == kAlignDiscardShift) continue;   // excluded — ignore for crop range
            int64_t s = static_cast<int64_t>(sraw);
            crop_start = std::max(crop_start, -s);
            crop_end   = std::min(crop_end, h.num_samples - s);
        }
        if (crop_end > crop_start) { out_start = crop_start; out_len = crop_end - crop_start; }
    }

    // Store alignment state (including any kAlignDiscardShift entries) for
    // later reuse by t-test/CPA/cross-correlation/export.
    saveSnapshot();
    activeDs().align_first_trace  = first_tr;
    activeDs().align_first_sample = out_start;
    activeDs().align_n_samples    = out_len;
    activeDs().align_shifts.assign(shifts.begin(), shifts.end());

    // Show only NUM_COLORS representative traces in the main plot. These
    // are in-memory baked-in traces — mark as not file-backed so drag-align
    // on them won't overwrite the stored alignment state.
    // buildAlignedTraces() already applied activeDs().pipeline to produce
    // them, so the plot's own live transforms must stay empty here —
    // otherwise the pipeline would be applied a second time on top of the
    // already-processed result.
    plotWidget()->clearTraces();
    if (!buildAlignedTraces(plotWidget(), shifts, first_tr, output_mode, NUM_COLORS, msg_parent)) {
        err = "Failed to build aligned preview.";
        return false;
    }
    plotWidget()->setTransforms({});
    plotWidget()->resetView();
    activeDs().plot_file_backed = false;
    updateUndoButton();   // also refreshes btn_unapply_'s enabled state
    return true;
}

// ---------------------------------------------------------------------------
// Writes the (trace,offset) shift table for [first_trace, first_trace +
// shifts.size()) to a CSV file, skipping entries marked kAlignDiscardShift.
// Same format as Align Traces' "Export Shifts…" button (and read back by
// loadShiftsCsv() below / "Load Shifts from File…"). Shared by that button
// and the Chain "Export Shifts" step. Returns false and sets err_out on
// failure.
// ---------------------------------------------------------------------------
static bool saveShiftsCsv(const QString& path, int32_t first_trace,
                           const std::vector<int32_t>& shifts, QString& err_out)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        err_out = "Cannot create file:\n" + path;
        return false;
    }
    QTextStream out(&f);
    out << "# trs-viewer alignment shifts (trace,offset) — omitted traces were "
           "discarded/excluded\n";
    for (int i = 0; i < static_cast<int>(shifts.size()); i++) {
        int32_t shift = shifts[static_cast<size_t>(i)];
        if (shift == kAlignDiscardShift) continue;   // omit — matches the threshold filter
        out << (first_trace + i) << ',' << shift << '\n';
    }
    f.close();
    return true;
}

// Reads a (trace,offset) shift table written by saveShiftsCsv() above (or by
// hand) — trace numbers missing from the file become kAlignDiscardShift
// within the resulting [min,max] range. Shared by "Load Shifts from File…"
// and the Chain "Load Shifts" step. Returns false and sets err_out on
// failure, including "no valid rows found".
static bool loadShiftsCsv(const QString& path, int32_t& first_trace_out,
                           std::vector<int32_t>& shifts_out, QString& err_out,
                           int* bad_lines_out = nullptr)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        err_out = "Cannot open file:\n" + path;
        return false;
    }
    std::map<int32_t, int32_t> pairs;
    QTextStream in(&f);
    int bad_lines = 0;
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith('#')) continue;
        QStringList parts = line.split(',');
        bool ok1 = false, ok2 = false;
        if (parts.size() >= 2) {
            int32_t tr  = parts[0].trimmed().toInt(&ok1);
            int32_t off = parts[1].trimmed().toInt(&ok2);
            if (ok1 && ok2) { pairs[tr] = off; continue; }
        }
        bad_lines++;
    }
    f.close();
    if (bad_lines_out) *bad_lines_out = bad_lines;
    if (pairs.empty()) {
        err_out = "No valid \"trace,offset\" rows found in:\n" + path;
        return false;
    }
    int32_t first_tr = pairs.begin()->first;
    int32_t last_tr  = pairs.rbegin()->first;
    shifts_out.assign(static_cast<size_t>(last_tr - first_tr + 1), kAlignDiscardShift);
    for (const auto& kv : pairs) shifts_out[static_cast<size_t>(kv.first - first_tr)] = kv.second;
    first_trace_out = first_tr;
    return true;
}

// ---------------------------------------------------------------------------
// Trace alignment dialog
// ---------------------------------------------------------------------------
void MainWindow::onAlignTraces() {
    showAlignDialog(nullptr);
}

void MainWindow::showAlignDialog(std::function<void(const ChainStep&)> onAddToChain)
{
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Align Traces", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();

    // Every range control below (reference region, search window, drag-to-
    // select) is expressed in *pipeline-processed* sample units, not raw —
    // with a decimating pipeline stage active, that's the domain the user
    // actually cares about, and it lets the raw window read+processed per
    // candidate trace stay as small as the request needs instead of the
    // caller having to over-provision a raw count "just in case". Converted
    // back to raw right before calling alignByPeak/alignByXCorr, which still
    // take raw sample positions/counts (see rawCountForProcessedCount()).
    // Identity when there's no pipeline, so this is a no-op in the common case.
    const int64_t proc_total = processedCountForRaw(activeDs().pipeline, h.num_samples);

    auto* dlg = new QDialog(this);
    dlg->setWindowTitle("Align Traces");
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    auto* vl = new QVBoxLayout(dlg);

    // ── Parameters ───────────────────────────────────────────────────────────
    auto* grp = new QGroupBox("Parameters");
    auto* fl  = new QFormLayout(grp);

    // Reference trace
    auto* sp_ref = new QSpinBox;
    sp_ref->setRange(0, h.num_traces - 1);
    sp_ref->setValue(spin_first_->value());
    sp_ref->setToolTip("Absolute trace index used as the alignment template.");

    // Reference region — in pipeline-processed sample units (see note above).
    auto* sp_ref_first = new QSpinBox;
    sp_ref_first->setRange(0, static_cast<int>(std::max<int64_t>(0, proc_total - 1)));
    sp_ref_first->setToolTip("Pipeline-processed sample index (after any active pipeline stages).");
    auto* sp_ref_len = new QSpinBox;
    sp_ref_len->setRange(2, static_cast<int>(std::max<int64_t>(2, proc_total)));
    sp_ref_len->setValue(static_cast<int>(std::min<int64_t>(200, proc_total)));
    sp_ref_len->setToolTip("Length in pipeline-processed samples.");

    // Seed from first crop range if one exists (crop ranges are always raw —
    // the plot's own coordinate system — so convert to processed here).
    if (!plotWidget()->cropRanges().empty()) {
        auto [cs, ce] = plotWidget()->cropRanges()[0];
        int64_t pcs = processedCountForRaw(activeDs().pipeline, cs);
        int64_t pce = processedCountForRaw(activeDs().pipeline, ce);
        sp_ref_first->setValue(static_cast<int>(pcs));
        sp_ref_len->setValue(static_cast<int>(std::max<int64_t>(2, pce - pcs)));
    }

    // "Draw on plot" button — puts the main plot into CropSelect mode;
    // when the user draws a region the spinboxes update automatically.
    auto* btn_draw = new QPushButton("Draw on plot →");
    btn_draw->setToolTip("Switch the main plot to crop-select mode.\n"
                         "Drag to mark the reference region — releasing the "
                         "mouse confirms it immediately, no Enter needed.");

    connect(btn_draw, &QPushButton::clicked, dlg, [=]() {
        plotWidget()->clearCropRanges();
        plotWidget()->setCropAutoConfirm(true);
        plotWidget()->setMode(InteractionMode::CropSelect);
        btn_draw->setText("Drawing… (drag on plot)");
        btn_draw->setEnabled(false);
    });

    // A QObject parented to dlg so the connection is torn down when the dialog
    // closes, even if the user never finishes drawing.
    auto* crop_guard = new QObject(dlg);
    connect(plotWidget(), &PlotWidget::cropRangesChanged, crop_guard, [=]() {
        const auto& ranges = plotWidget()->cropRanges();
        if (ranges.empty()) return;
        auto [s, e] = ranges.back();
        int64_t ps = processedCountForRaw(activeDs().pipeline, s);
        int64_t pe = processedCountForRaw(activeDs().pipeline, e);
        sp_ref_first->setValue(static_cast<int>(ps));
        sp_ref_len->setValue(static_cast<int>(std::max<int64_t>(2, pe - ps)));
        // Restore normal mode and re-enable button
        plotWidget()->setCropAutoConfirm(false);
        plotWidget()->setMode(InteractionMode::Pan);
        btn_draw->setText("Draw on plot →");
        btn_draw->setEnabled(true);
    });

    // Restore Pan mode if the dialog is closed mid-draw
    connect(dlg, &QDialog::finished, dlg, [=](int) {
        plotWidget()->setCropAutoConfirm(false);
        if (plotWidget()->mode() == InteractionMode::CropSelect)
            plotWidget()->setMode(InteractionMode::Pan);
    });

    auto* region_row = new QWidget;
    auto* region_hl  = new QHBoxLayout(region_row);
    region_hl->setContentsMargins(0, 0, 0, 0);
    region_hl->addWidget(new QLabel("First:"));
    region_hl->addWidget(sp_ref_first);
    region_hl->addWidget(new QLabel("Length:"));
    region_hl->addWidget(sp_ref_len);
    region_hl->addWidget(btn_draw);
    region_hl->addStretch();

    // Method
    auto* combo_method = new QComboBox;
    combo_method->addItem("Peak alignment");
    combo_method->addItem("Cross-correlation");
    combo_method->setToolTip(
        "Peak: each trace's highest peak within the search window is matched "
        "to the reference peak.\n"
        "Cross-correlation: the reference region is used as a template; "
        "the lag with maximum normalised correlation is used.");

    // Search window — also in pipeline-processed sample units.
    auto* sp_search = new QSpinBox;
    sp_search->setRange(1, static_cast<int>(std::max<int64_t>(1, proc_total / 2)));
    sp_search->setValue(static_cast<int>(std::min<int64_t>(50, std::max<int64_t>(1, proc_total / 2))));
    sp_search->setToolTip("Maximum shift to consider (± pipeline-processed samples "
                          "around the reference position).");

    // Peak mode row (hidden for XCorr)
    auto* peak_row = new QWidget;
    auto* peak_hl  = new QHBoxLayout(peak_row);
    peak_hl->setContentsMargins(0, 0, 0, 0);
    auto* combo_peak = new QComboBox;
    combo_peak->addItem("Absolute max  |v|");
    combo_peak->addItem("Signed max");
    peak_hl->addWidget(new QLabel("Peak mode:"));
    peak_hl->addWidget(combo_peak);
    peak_hl->addStretch();

    // Discard-below-threshold row (Cross-correlation only)
    auto* corr_row = new QWidget;
    auto* corr_hl  = new QHBoxLayout(corr_row);
    corr_hl->setContentsMargins(0, 0, 0, 0);
    auto* chk_discard = new QCheckBox("Discard below correlation:");
    auto* sp_min_corr = new QDoubleSpinBox;
    sp_min_corr->setRange(-1.0, 1.0);
    sp_min_corr->setSingleStep(0.05);
    sp_min_corr->setDecimals(2);
    sp_min_corr->setValue(0.5);
    sp_min_corr->setEnabled(false);
    chk_discard->setToolTip(
        "Traces whose best normalised cross-correlation (NCC, -1..1) against "
        "the reference falls below this value are excluded rather than "
        "shifted — from this dialog's own output, and later from t-test/CPA/"
        "cross-correlation if the stored alignment is reused.");
    connect(chk_discard, &QCheckBox::toggled, sp_min_corr, &QDoubleSpinBox::setEnabled);
    corr_hl->addWidget(chk_discard);
    corr_hl->addWidget(sp_min_corr);
    corr_hl->addStretch();

    connect(combo_method, QOverload<int>::of(&QComboBox::currentIndexChanged),
            dlg, [peak_row, corr_row](int idx) {
                peak_row->setVisible(idx == 0);
                corr_row->setVisible(idx == 1);
            });
    corr_row->setVisible(false);

    // Traces to align
    auto* sp_tr_first = new QSpinBox;
    sp_tr_first->setRange(0, h.num_traces - 1);
    sp_tr_first->setValue(spin_first_->value());
    auto* sp_tr_count = new QSpinBox;
    sp_tr_count->setRange(1, h.num_traces);
    sp_tr_count->setValue(spin_count_->value());

    auto* tr_row = new QWidget;
    auto* tr_hl  = new QHBoxLayout(tr_row);
    tr_hl->setContentsMargins(0, 0, 0, 0);
    tr_hl->addWidget(new QLabel("First:"));
    tr_hl->addWidget(sp_tr_first);
    tr_hl->addWidget(new QLabel("Count:"));
    tr_hl->addWidget(sp_tr_count);
    tr_hl->addStretch();

    fl->addRow("Reference trace:",   sp_ref);
    fl->addRow("Reference region (pipeline samples):", region_row);
    fl->addRow("Method:",            combo_method);
    fl->addRow("Search window ± (pipeline samples):", sp_search);
    fl->addRow(peak_row);
    fl->addRow(corr_row);
    fl->addRow("Traces:",            tr_row);
    vl->addWidget(grp);

    auto* run_row = new QWidget;
    auto* run_hl  = new QHBoxLayout(run_row);
    run_hl->setContentsMargins(0, 0, 0, 0);
    auto* btn_run = new QPushButton("Run");
    auto* btn_load_shifts = new QPushButton("Load Shifts from File…");
    btn_load_shifts->setToolTip(
        "Load a previously-exported (trace,offset) shift table instead of "
        "running a search — e.g. shifts computed elsewhere, or with some "
        "traces manually omitted.");
    run_hl->addWidget(btn_run);
    run_hl->addWidget(btn_load_shifts);
    vl->addWidget(run_row);

    // ── Results (shown after a successful run) ────────────────────────────────
    auto* tbl = new QTableWidget(0, 3);
    tbl->setHorizontalHeaderLabels({"Trace", "Shift (samples)", "Correlation"});
    tbl->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tbl->horizontalHeader()->setStretchLastSection(true);
    tbl->setMaximumHeight(220);
    tbl->hide();
    vl->addWidget(tbl);

    auto* lbl_discard_summary = new QLabel;
    lbl_discard_summary->setWordWrap(true);
    lbl_discard_summary->hide();
    vl->addWidget(lbl_discard_summary);

    // Output mode selector + show button (hidden until run completes)
    auto* output_row = new QWidget;
    auto* output_hl  = new QHBoxLayout(output_row);
    output_hl->setContentsMargins(0, 0, 0, 0);
    auto* combo_output = new QComboBox;
    combo_output->addItem("Full trace — pad with average");
    combo_output->addItem("Full trace — pad with zeros");
    combo_output->addItem("Crop to common range");
    combo_output->setToolTip(
        "Pad with average: fill the shifted-in region with the mean of each trace.\n"
        "Pad with zeros: fill with 0.\n"
        "Crop: trim all traces to the sample range where every trace has real data.");
    auto* btn_show  = new QPushButton("Show in New Window…");
    auto* btn_apply = new QPushButton(onAddToChain ? "Apply to Main View + Add to Chain"
                                                     : "Apply to Main View");
    btn_apply->setToolTip(onAddToChain
        ? "Replace the main plot with the aligned traces, and capture these exact "
          "parameters (method, range, reference, region, search window, threshold, "
          "output mode) as an Align step in the chain being built."
        : "Replace the main plot with the aligned traces.");
    auto* btn_export_shifts = new QPushButton("Export Shifts…");
    btn_export_shifts->setToolTip(
        "Save the (trace,offset) table shown above to a text file — omitted/"
        "discarded traces are left out, so it can later be reloaded with "
        "some traces intentionally excluded.");
    output_hl->addWidget(combo_output);
    output_hl->addWidget(btn_show);
    output_hl->addWidget(btn_apply);
    output_hl->addWidget(btn_export_shifts);
    output_row->hide();
    vl->addWidget(output_row);

    // Shared mutable state between Run and Show
    auto result_ptr = std::make_shared<AlignResult>();

    // Populates the results table + reveals the output controls from
    // whatever's currently in result_ptr — shared by a live Run and by
    // loading a shift table from a file, so both end up in the same state.
    auto showResults = [=](int32_t first_tr) {
        int32_t num_tr = static_cast<int32_t>(result_ptr->shifts.size());
        const auto& scores = result_ptr->scores;
        int n_discarded = 0;
        tbl->setRowCount(0);
        for (int i = 0; i < num_tr; i++) {
            int row = tbl->rowCount();
            tbl->insertRow(row);
            const int32_t shift = result_ptr->shifts[static_cast<size_t>(i)];
            const bool discarded = (shift == kAlignDiscardShift);
            if (discarded) n_discarded++;

            auto* item_trace = new QTableWidgetItem(QString::number(first_tr + i));
            auto* item_shift = new QTableWidgetItem(
                discarded ? QString("discarded") : QString::number(shift));
            auto* item_score = new QTableWidgetItem(
                (i < static_cast<int>(scores.size()))
                    ? QString::number(scores[static_cast<size_t>(i)], 'f', 3)
                    : QString());
            if (discarded) {
                for (auto* item : {item_trace, item_shift, item_score}) {
                    item->setBackground(QColor(255, 200, 200));
                    item->setForeground(QColor(150, 0, 0));
                }
            }
            tbl->setItem(row, 0, item_trace);
            tbl->setItem(row, 1, item_shift);
            tbl->setItem(row, 2, item_score);
        }
        tbl->show();

        if (n_discarded > 0) {
            lbl_discard_summary->setText(
                QString("%1 of %2 traces discarded/omitted.").arg(n_discarded).arg(num_tr));
            lbl_discard_summary->show();
        } else {
            lbl_discard_summary->hide();
        }

        output_row->show();
        dlg->adjustSize();
    };

    // ── Run ──────────────────────────────────────────────────────────────────
    connect(btn_run, &QPushButton::clicked, dlg, [=]() {
        int32_t first_tr = static_cast<int32_t>(sp_tr_first->value());
        int32_t num_tr   = static_cast<int32_t>(sp_tr_count->value());
        num_tr = std::min(num_tr, h.num_traces - first_tr);
        if (num_tr <= 0) {
            QMessageBox::warning(dlg, "Align Traces", "No traces in range.");
            return;
        }

        int32_t ref_abs = static_cast<int32_t>(sp_ref->value());
        int32_t ref_off = ref_abs - first_tr;
        if (ref_off < 0 || ref_off >= num_tr) {
            QMessageBox::warning(dlg, "Align Traces",
                QString("Reference trace %1 is outside the selected range [%2, %3).")
                    .arg(ref_abs).arg(first_tr).arg(first_tr + num_tr));
            return;
        }

        // sp_ref_first/sp_ref_len/sp_search are pipeline-processed sample
        // units (see comment at the top of onAlignTraces); convert to raw
        // sample positions/counts here, right before calling into align.cpp,
        // which — like the rest of the app — works in raw terms and applies
        // the pipeline itself.
        const auto& pipeline_now = activeDs().pipeline;
        int64_t proc_ref_first = static_cast<int64_t>(sp_ref_first->value());
        int64_t proc_ref_end   = proc_ref_first + static_cast<int64_t>(sp_ref_len->value());
        int64_t ref_first = rawCountForProcessedCount(pipeline_now, proc_ref_first, h.num_samples);
        int64_t ref_end   = rawCountForProcessedCount(pipeline_now, proc_ref_end,   h.num_samples);
        int64_t ref_len   = std::max<int64_t>(1, ref_end - ref_first);
        int32_t shalf     = static_cast<int32_t>(std::max<int64_t>(1,
            rawCountForProcessedCount(pipeline_now, sp_search->value(), h.num_samples)));
        bool    use_abs   = (combo_peak->currentIndex() == 0);
        bool    is_peak   = (combo_method->currentIndex() == 0);

        QProgressDialog prog(
            is_peak ? "Finding peaks…" : "Cross-correlating…",
            "Cancel", 0, num_tr, dlg);
        prog.setWindowModality(Qt::WindowModal);
        prog.setMinimumDuration(300);
        prog.setValue(0);

        auto progress_fn = [&](int done, int total) -> bool {
            prog.setValue(done);
            prog.setMaximum(total);
            QApplication::processEvents();
            return !prog.wasCanceled();
        };

        const bool  discard_enabled = !is_peak && chk_discard->isChecked();
        const float min_corr        = discard_enabled
            ? static_cast<float>(sp_min_corr->value()) : -2.0f;

        std::string err;
        bool ok;
        if (is_peak) {
            ok = alignByPeak(activeDs().file.get(), activeDs().pipeline,
                             first_tr, num_tr, ref_off,
                             ref_first, ref_len, shalf, use_abs,
                             *result_ptr, progress_fn, err);
        } else {
            ok = alignByXCorr(activeDs().file.get(), activeDs().pipeline,
                              first_tr, num_tr, ref_off,
                              ref_first, ref_len, shalf, min_corr,
                              *result_ptr, progress_fn, err);
        }
        prog.setValue(num_tr);

        if (!ok) {
            if (!err.empty())
                QMessageBox::critical(dlg, "Alignment failed",
                                      QString::fromStdString(err));
            return;
        }

        showResults(first_tr);
    });

    // ── Load shifts from file ───────────────────────────────────────────────
    // File format: one "trace,offset" pair per line (blank lines and lines
    // starting with '#' are ignored). Trace numbers not present in the file
    // are treated as discarded/omitted, same as a live run's threshold
    // filter — so a hand-edited or filtered export can leave traces out.
    connect(btn_load_shifts, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getOpenFileName(dlg, "Load alignment shifts",
                                                     MainWindow::recentDir("align_shifts"),
                                                     "Shift tables (*.csv *.txt);;All files (*)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("align_shifts", path);

        int32_t first_tr = 0;
        int bad_lines = 0;
        QString err;
        if (!loadShiftsCsv(path, first_tr, result_ptr->shifts, err, &bad_lines)) {
            QMessageBox::critical(dlg, "Load failed", err);
            return;
        }
        result_ptr->scores.clear();   // not available from a file — table shows blank scores

        sp_tr_first->setValue(first_tr);
        sp_tr_count->setValue(static_cast<int>(result_ptr->shifts.size()));
        showResults(first_tr);

        if (bad_lines > 0)
            QMessageBox::warning(dlg, "Load shifts",
                QString("Loaded %1 trace(s); skipped %2 unparseable line(s).")
                    .arg(result_ptr->shifts.size()).arg(bad_lines));
    });

    // ── Export shifts to file ───────────────────────────────────────────────
    connect(btn_export_shifts, &QPushButton::clicked, dlg, [=]() {
        if (result_ptr->shifts.empty()) return;
        QString path = QFileDialog::getSaveFileName(dlg, "Export alignment shifts",
                                                     MainWindow::recentDir("align_shifts"),
                                                     "CSV files (*.csv);;All files (*)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("align_shifts", path);

        int32_t first_tr = static_cast<int32_t>(sp_tr_first->value());
        int32_t n_written = static_cast<int32_t>(std::count_if(
            result_ptr->shifts.begin(), result_ptr->shifts.end(),
            [](int32_t s) { return s != kAlignDiscardShift; }));
        QString err;
        if (!saveShiftsCsv(path, first_tr, result_ptr->shifts, err)) {
            QMessageBox::critical(dlg, "Export failed", err);
            return;
        }
        QMessageBox::information(dlg, "Export complete",
            QString("Saved %1 trace shift(s) to:\n%2").arg(n_written).arg(path));
    });

    // ── Show aligned traces ───────────────────────────────────────────────────
    // Builds aligned trace data into a PlotWidget from the dialog's current
    // result/range/output-mode — thin wrapper around buildAlignedTraces(),
    // which also backs the Chain "Align" step (see mainwindow.h).
    auto buildAligned = [=](PlotWidget* pw, int max_display = INT_MAX) -> bool {
        return buildAlignedTraces(pw, result_ptr->shifts,
                                   static_cast<int32_t>(sp_tr_first->value()),
                                   combo_output->currentIndex(), max_display, dlg);
    };

    connect(btn_show, &QPushButton::clicked, dlg, [=]() {
        if (result_ptr->shifts.empty()) return;
        int32_t num_tr = static_cast<int32_t>(std::count_if(
            result_ptr->shifts.begin(), result_ptr->shifts.end(),
            [](int32_t s) { return s != kAlignDiscardShift; }));

        auto* vdlg = new QDialog(dlg);
        vdlg->setWindowTitle(QString("Aligned traces — %1 traces").arg(num_tr));
        vdlg->setAttribute(Qt::WA_DeleteOnClose);
        auto* vl2 = new QVBoxLayout(vdlg);
        auto* pw  = new PlotWidget(vdlg);
        vl2->addWidget(pw);

        if (!buildAligned(pw)) { vdlg->deleteLater(); return; }

        pw->resetView();
        vdlg->resize(1100, 500);
        vdlg->show();
    });

    connect(btn_apply, &QPushButton::clicked, dlg, [=]() {
        if (result_ptr->shifts.empty()) return;
        QString err;
        if (!computeAndStoreAlignment(result_ptr->shifts,
                                       static_cast<int32_t>(sp_tr_first->value()),
                                       combo_output->currentIndex(), dlg, err))
            return;   // buildAlignedTraces() already showed the warning, if any

        if (onAddToChain) {
            // Same fields btn_run just used to produce result_ptr->shifts —
            // captured from the dialog's own widgets rather than duplicated
            // into a separate blind parameter form, so adding an Align step
            // to a chain means actually running it against the current view
            // first, same as using this dialog normally.
            ChainStep s;
            s.kind            = ChainStep::Kind::Align;
            s.align_method    = combo_method->currentIndex();
            s.first_trace     = static_cast<int32_t>(sp_tr_first->value());
            s.trace_count     = static_cast<int32_t>(sp_tr_count->value());
            s.ref_offset      = static_cast<int32_t>(sp_ref->value()) - s.first_trace;
            s.ref_first       = static_cast<int64_t>(sp_ref_first->value());
            s.ref_len         = static_cast<int64_t>(sp_ref_len->value());
            s.search_half     = static_cast<int32_t>(sp_search->value());
            s.peak_use_abs    = (combo_peak->currentIndex() == 0);
            s.discard_enabled = chk_discard->isChecked();
            s.min_corr        = sp_min_corr->value();
            s.output_mode     = combo_output->currentIndex();
            onAddToChain(s);
        }
        dlg->accept();
    });

    dlg->resize(480, 220);
    dlg->show();
}

// ---------------------------------------------------------------------------
// Chain "Run T-test" step: resolves the trace range from the last
// alignment (if step.use_last_alignment and one exists) or the step's own
// first_trace/trace_count, then computeTTest() + buildTTestResultTab() —
// same two calls showTTestDialog() makes after its config dialog closes.
// ---------------------------------------------------------------------------
bool MainWindow::runTTestChainStep(const ChainStep& step, QWidget* msg_parent, QString& err) {
    const TrsHeader& h = activeDs().file->header();
    if (h.data_length <= 0) {
        err = "This TRS file has no per-trace data bytes.";
        return false;
    }

    bool have_ttest_param = h.param_map.count("ttest") > 0;
    int32_t byte_idx = have_ttest_param
        ? static_cast<int32_t>(h.param_map.at("ttest").offset)
        : step.ttest_byte_idx;

    const bool use_alignment = step.use_last_alignment && activeDs().align_n_samples > 0;
    const int32_t eff_first = use_alignment ? activeDs().align_first_trace : step.first_trace;
    const int32_t eff_count = use_alignment ? static_cast<int32_t>(activeDs().align_shifts.size())
                                              : step.trace_count;
    if (eff_count < 2) {
        err = "Not enough traces in range (need at least 2).";
        return false;
    }
    const std::vector<int32_t> use_shifts = use_alignment ? activeDs().align_shifts
                                                             : std::vector<int32_t>{};

    std::shared_ptr<TTestAccumulator> acc_ptr;
    std::vector<float> tstat;
    int64_t n0 = 0, n1 = 0;
    if (!computeTTest(eff_first, eff_count, step.ttest_first_sample, step.ttest_n_samples,
                       byte_idx, use_shifts, step.ttest_abs, msg_parent, acc_ptr, tstat, n0, n1, err))
        return false;

    buildTTestResultTab(acc_ptr, std::move(tstat), n0, n1, eff_count, step.ttest_abs);
    return true;
}

// ---------------------------------------------------------------------------
// Chain: executes one saved step against the active dataset. Returns false
// (with err set) on failure — the Chain Editor's Run loop stops there.
// msg_parent is used for any warning/save-path dialogs a step needs to pop
// (memory warnings, "no path set" export prompts).
// ---------------------------------------------------------------------------
bool MainWindow::runChainStep(const ChainStep& step, QWidget* msg_parent, QString& err) {
    if (!hasActiveDs() || activeDs().is_result) {
        err = "No file loaded.";
        return false;
    }

    switch (step.kind) {
    case ChainStep::Kind::AddTransform: {
        auto tx = buildTransformFromStep(step);
        if (!tx) { err = "Unknown transform type in step."; return false; }
        // Same tail as onAddTransform(), minus the interactive createTransform() call.
        saveSnapshot();
        activeDs().pipeline.push_back(tx);
        rebuildTransformList();
        updateFileInfo();
        if (activeDs().plot_file_backed)
            plotWidget()->setTransforms(activeDs().pipeline);
        else
            rebakeAlignedView();
        plotWidget()->update();
        return true;
    }

    case ChainStep::Kind::ClearPipeline: {
        saveSnapshot();
        activeDs().pipeline.clear();
        rebuildTransformList();
        updateFileInfo();
        if (activeDs().plot_file_backed)
            plotWidget()->setTransforms({});
        else
            rebakeAlignedView();
        plotWidget()->update();
        return true;
    }

    case ChainStep::Kind::Align: {
        const TrsHeader& h = activeDs().file->header();
        int32_t first_tr = step.first_trace;
        int32_t num_tr   = std::min(step.trace_count, h.num_traces - first_tr);
        if (num_tr <= 0) { err = "No traces in the step's trace range."; return false; }
        if (step.ref_offset < 0 || step.ref_offset >= num_tr) {
            err = QString("Reference trace is outside the step's trace range [%1, %2).")
                      .arg(first_tr).arg(first_tr + num_tr);
            return false;
        }

        // ref_first/ref_len/search_half are stored in pipeline-processed
        // sample units, same convention as the Align Traces dialog (see its
        // top-of-function comment) — convert to raw using the *current*
        // pipeline, right before calling into align.cpp.
        const auto& pipeline_now = activeDs().pipeline;
        int64_t ref_first = rawCountForProcessedCount(pipeline_now, step.ref_first, h.num_samples);
        int64_t ref_end   = rawCountForProcessedCount(pipeline_now, step.ref_first + step.ref_len,
                                                       h.num_samples);
        int64_t ref_len   = std::max<int64_t>(1, ref_end - ref_first);
        int32_t shalf     = static_cast<int32_t>(std::max<int64_t>(1,
            rawCountForProcessedCount(pipeline_now, step.search_half, h.num_samples)));

        QProgressDialog prog(step.align_method == 0 ? "Finding peaks…" : "Cross-correlating…",
                              "Cancel", 0, num_tr, msg_parent);
        prog.setWindowModality(Qt::WindowModal);
        prog.setMinimumDuration(300);
        auto progress_fn = [&](int done, int total) -> bool {
            prog.setValue(done);
            prog.setMaximum(total);
            QApplication::processEvents();
            return !prog.wasCanceled();
        };

        AlignResult result;
        std::string aerr;
        bool ok;
        if (step.align_method == 0) {
            ok = alignByPeak(activeDs().file.get(), pipeline_now, first_tr, num_tr, step.ref_offset,
                              ref_first, ref_len, shalf, step.peak_use_abs, result, progress_fn, aerr);
        } else {
            float min_corr = step.discard_enabled ? static_cast<float>(step.min_corr) : -2.0f;
            ok = alignByXCorr(activeDs().file.get(), pipeline_now, first_tr, num_tr, step.ref_offset,
                               ref_first, ref_len, shalf, min_corr, result, progress_fn, aerr);
        }
        if (!ok) {
            err = !aerr.empty() ? QString::fromStdString(aerr) : "Alignment cancelled.";
            return false;
        }
        return computeAndStoreAlignment(result.shifts, first_tr, step.output_mode, msg_parent, err);
    }

    case ChainStep::Kind::Reload: {
        // "Apply the stored shift to the raw traces": drive the reload from
        // the alignment's own range rather than whatever the sidebar's
        // First trace/Count spinboxes currently hold (unrelated global
        // widgets that may be stale from something else entirely) — see the
        // comment on MainWindow::runChainStep in mainwindow.h.
        if (activeDs().align_n_samples > 0) {
            spin_first_->setValue(activeDs().align_first_trace);
            spin_count_->setValue(static_cast<int>(activeDs().align_shifts.size()));
        }
        onApplyTraces();
        return true;
    }

    case ChainStep::Kind::Export: {
        QString path = step.path;
        if (path.isEmpty()) {
            QString title, filter;
            switch (step.export_format) {
            case 0: title = "Export processed TRS";   filter = "TRS files (*.trs)";        break;
            case 1: title = "Export traces as NPY";    filter = "NumPy files (*.npy)";      break;
            default: title = "Export traces as NPZ";   filter = "NumPy archives (*.npz)";   break;
            }
            path = QFileDialog::getSaveFileName(msg_parent, title, recentDir("chain"), filter);
            if (path.isEmpty()) { err = "Export cancelled."; return false; }
            updateRecentDir("chain", path);
        }

        const std::vector<int32_t> shifts = step.use_last_alignment ? activeDs().align_shifts
                                                                       : std::vector<int32_t>{};
        int32_t n_written = 0;
        bool ok;
        switch (step.export_format) {
        case 0:
            ok = exportTracesToTrs(path, activeDs().file.get(), step.exp_first, step.exp_count,
                                    shifts, activeDs().pipeline, nullptr, err, &n_written);
            // exportTracesToTrs repurposes a non-empty err as a non-fatal
            // clipped-sample-count note on success, not a failure.
            if (ok) err.clear();
            break;
        case 1:
            ok = exportTracesToNpy(path, activeDs().file.get(), step.exp_first, step.exp_count,
                                    shifts, activeDs().pipeline, msg_parent, nullptr, err, &n_written);
            break;
        default:
            ok = exportTracesToNpz(path, activeDs().file.get(), step.exp_first, step.exp_count,
                                    shifts, activeDs().pipeline, nullptr, err, &n_written);
            break;
        }
        return ok;
    }

    case ChainStep::Kind::ExportShifts: {
        if (activeDs().align_shifts.empty()) {
            err = "No alignment shifts to export — run an Align step first.";
            return false;
        }
        QString path = step.path;
        if (path.isEmpty()) {
            path = QFileDialog::getSaveFileName(msg_parent, "Export alignment shifts",
                                                 recentDir("chain"), "CSV files (*.csv)");
            if (path.isEmpty()) { err = "Export cancelled."; return false; }
            updateRecentDir("chain", path);
        }
        return saveShiftsCsv(path, activeDs().align_first_trace, activeDs().align_shifts, err);
    }

    case ChainStep::Kind::LoadShifts: {
        QString path = step.path;
        if (path.isEmpty()) {
            path = QFileDialog::getOpenFileName(msg_parent, "Load alignment shifts",
                recentDir("chain"), "Shift tables (*.csv *.txt);;All files (*)");
            if (path.isEmpty()) { err = "Load cancelled."; return false; }
            updateRecentDir("chain", path);
        }
        int32_t first_tr = 0;
        std::vector<int32_t> shifts;
        if (!loadShiftsCsv(path, first_tr, shifts, err)) return false;

        saveSnapshot();
        activeDs().align_first_trace  = first_tr;
        activeDs().align_first_sample = 0;
        activeDs().align_n_samples    = activeDs().file->header().num_samples;
        activeDs().align_shifts       = std::move(shifts);
        updateUndoButton();
        return true;
    }

    case ChainStep::Kind::RunTTest:
        return runTTestChainStep(step, msg_parent, err);
    }
    err = "Unknown step kind.";
    return false;
}

// ---------------------------------------------------------------------------
// Chain Editor — build, save/load, and run a saved sequence of the above
// steps against the active dataset. See inc/chain.h for the step model.
// ---------------------------------------------------------------------------
void MainWindow::onChainEditor() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Chain Editor", "No file loaded.");
        return;
    }

    auto* dlg = new QDialog(this);
    dlg->setWindowTitle("Chain Editor");
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    auto* vl = new QVBoxLayout(dlg);

    vl->addWidget(new QLabel(
        "A saved, ordered sequence of steps — build it once, then Run it "
        "(or save it and load it back later) instead of redoing the same "
        "clicks by hand every time."));

    auto* list = new QListWidget;
    vl->addWidget(list, 1);

    auto steps = std::make_shared<std::vector<ChainStep>>();
    auto refreshList = [=]() {
        list->clear();
        for (const auto& s : *steps) list->addItem(s.summary());
    };

    // ── Add Transform sub-dialog: reuses combo_transform_'s item list and
    // createTransform() itself (fully interactive, unchanged) — the result
    // is then read back into a step via describeTransformStep() so running
    // it later needs no dialogs. ──────────────────────────────────────────
    auto addTransformStep = [=]() {
        bool ok = false;
        QString choice = QInputDialog::getItem(dlg, "Add Step — Add Transform",
            "Transform type:", [this]() {
                QStringList items;
                for (int i = 0; i < combo_transform_->count(); i++)
                    items << combo_transform_->itemText(i);
                return items;
            }(), 0, false, &ok);
        if (!ok) return;
        int idx = combo_transform_->findText(choice);
        if (idx < 0) return;
        auto tx = createTransform(idx);   // interactive per-type parameter dialog
        if (!tx) return;                  // user cancelled the parameter dialog
        steps->push_back(describeTransformStep(idx, *tx));
        refreshList();
    };

    // ── Align: opens the real, interactive Align Traces dialog (drag-on-plot region,
    // Run, results table) instead of a blind parameter form — remembering
    // good region/threshold numbers without seeing them applied is hard, so
    // adding an Align step means actually running alignment against the
    // current view first (via that dialog's "Apply to Main View + Add to
    // Chain" button) and capturing the exact parameters that just worked.
    auto addAlignStep = [=]() {
        showAlignDialog([=](const ChainStep& s) {
            steps->push_back(s);
            refreshList();
        });
    };

    // ── Export sub-dialog ──────────────────────────────────────────────────
    auto addExportStep = [=]() {
        const TrsHeader& h = activeDs().file->header();

        QDialog d(dlg);
        d.setWindowTitle("Add Step — Export");
        auto* fl = new QFormLayout(&d);

        auto* combo_fmt = new QComboBox;
        combo_fmt->addItems({"TRS", "NPY", "NPZ"});
        fl->addRow("Format:", combo_fmt);

        auto* sp_first = new QSpinBox; sp_first->setRange(0, std::max(0, h.num_traces - 1));
        auto* sp_count = new QSpinBox; sp_count->setRange(1, h.num_traces);
        if (activeDs().align_n_samples > 0) {
            sp_first->setValue(activeDs().align_first_trace);
            sp_count->setValue(static_cast<int>(activeDs().align_shifts.size()));
        } else {
            sp_first->setValue(0);
            sp_count->setValue(h.num_traces);
        }
        fl->addRow("First trace:", sp_first);
        fl->addRow("Count:",       sp_count);

        // A plain checkbox rather than addAlignmentGroup(): at authoring
        // time there may be no alignment yet (e.g. an earlier step in this
        // very chain computes it), so this can't reflect or sync against
        // *current* alignment state the way the interactive export dialogs'
        // checkbox does — it just controls what happens when this step runs.
        auto* chk_align = new QCheckBox("Apply alignment shifts (if any) when this step runs");
        chk_align->setChecked(true);
        chk_align->setToolTip("If the dataset has a stored alignment when this step "
                              "executes (e.g. from an earlier Align step in this chain), "
                              "discarded traces are skipped and the rest are read with "
                              "their shift applied. No effect if there is none.");
        fl->addRow(chk_align);

        auto* le_path = new QLineEdit;
        le_path->setPlaceholderText("(leave blank to prompt for a path each run)");
        auto* btn_browse = new QPushButton("Browse…");
        auto* path_row = new QWidget;
        auto* path_hl  = new QHBoxLayout(path_row);
        path_hl->setContentsMargins(0, 0, 0, 0);
        path_hl->addWidget(le_path);
        path_hl->addWidget(btn_browse);
        fl->addRow("Output path:", path_row);
        connect(btn_browse, &QPushButton::clicked, &d, [&]() {
            QString filter = combo_fmt->currentIndex() == 0 ? "TRS files (*.trs)"
                            : combo_fmt->currentIndex() == 1 ? "NumPy files (*.npy)"
                                                               : "NumPy archives (*.npz)";
            QString p = QFileDialog::getSaveFileName(&d, "Output path", recentDir("chain"), filter);
            if (!p.isEmpty()) le_path->setText(p);
        });

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        fl->addRow(bb);
        connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
        if (d.exec() != QDialog::Accepted) return;

        ChainStep s;
        s.kind                = ChainStep::Kind::Export;
        s.export_format       = combo_fmt->currentIndex();
        s.exp_first            = static_cast<int32_t>(sp_first->value());
        s.exp_count             = static_cast<int32_t>(sp_count->value());
        s.use_last_alignment      = chk_align->isChecked();
        s.path                      = le_path->text();
        steps->push_back(s);
        refreshList();
    };

    // ── Export Shifts / Load Shifts sub-dialog: just a path field + browse
    // button, same "blank = prompt at run time" convention as Export. ─────
    auto addShiftsStep = [=](bool is_export) {
        QDialog d(dlg);
        d.setWindowTitle(is_export ? "Add Step — Export Shifts" : "Add Step — Load Shifts");
        auto* fl = new QFormLayout(&d);

        auto* le_path = new QLineEdit;
        le_path->setPlaceholderText("(leave blank to prompt for a path each run)");
        auto* btn_browse = new QPushButton("Browse…");
        auto* path_row = new QWidget;
        auto* path_hl  = new QHBoxLayout(path_row);
        path_hl->setContentsMargins(0, 0, 0, 0);
        path_hl->addWidget(le_path);
        path_hl->addWidget(btn_browse);
        fl->addRow("File:", path_row);
        connect(btn_browse, &QPushButton::clicked, &d, [&]() {
            QString p = is_export
                ? QFileDialog::getSaveFileName(&d, "Shift table path", recentDir("chain"),
                                                "CSV files (*.csv)")
                : QFileDialog::getOpenFileName(&d, "Shift table path", recentDir("chain"),
                                                "Shift tables (*.csv *.txt);;All files (*)");
            if (!p.isEmpty()) le_path->setText(p);
        });

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
        fl->addRow(bb);
        connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
        connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
        if (d.exec() != QDialog::Accepted) return;

        ChainStep s;
        s.kind = is_export ? ChainStep::Kind::ExportShifts : ChainStep::Kind::LoadShifts;
        s.path = le_path->text();
        steps->push_back(s);
        refreshList();
    };

    // ── Run T-test: reuses the real, interactive config dialog (see
    // showAlignDialog()'s "Apply to Main View + Add to Chain" for why) — its
    // own "Add to Chain" checkbox captures the parameters and actually runs
    // the t-test, showing the same result tab as the menu action. ─────────
    auto addRunTTestStep = [=]() {
        showTTestDialog([=](const ChainStep& s) {
            steps->push_back(s);
            refreshList();
        });
    };

    auto* btn_add_step = new QPushButton("Add step ▾");
    auto* btn_remove   = new QPushButton("Remove");
    auto* btn_up       = new QPushButton("↑");
    auto* btn_down     = new QPushButton("↓");
    auto* btn_load     = new QPushButton("Load…");
    auto* btn_save     = new QPushButton("Save…");
    auto* btn_run      = new QPushButton("Run");

    connect(btn_add_step, &QPushButton::clicked, dlg, [=]() {
        QMenu menu(dlg);
        QAction* act_tx      = menu.addAction("Add Transform…");
        QAction* act_clear   = menu.addAction("Clear Pipeline");
        QAction* act_align   = menu.addAction("Align Traces…");
        QAction* act_reload  = menu.addAction("Reload (Load / Refresh)");
        QAction* act_export  = menu.addAction("Export…");
        QAction* act_exp_sh  = menu.addAction("Export Shifts…");
        QAction* act_load_sh = menu.addAction("Load Shifts…");
        QAction* act_ttest   = menu.addAction("Run T-test…");
        QAction* chosen = menu.exec(btn_add_step->mapToGlobal(QPoint(0, btn_add_step->height())));
        if (chosen == act_tx) {
            addTransformStep();
        } else if (chosen == act_clear) {
            ChainStep s; s.kind = ChainStep::Kind::ClearPipeline;
            steps->push_back(s);
            refreshList();
        } else if (chosen == act_align) {
            addAlignStep();
        } else if (chosen == act_reload) {
            ChainStep s; s.kind = ChainStep::Kind::Reload;
            steps->push_back(s);
            refreshList();
        } else if (chosen == act_export) {
            addExportStep();
        } else if (chosen == act_exp_sh) {
            addShiftsStep(true);
        } else if (chosen == act_load_sh) {
            addShiftsStep(false);
        } else if (chosen == act_ttest) {
            addRunTTestStep();
        }
    });

    connect(btn_remove, &QPushButton::clicked, dlg, [=]() {
        int row = list->currentRow();
        if (row < 0 || row >= static_cast<int>(steps->size())) return;
        steps->erase(steps->begin() + row);
        refreshList();
    });
    connect(btn_up, &QPushButton::clicked, dlg, [=]() {
        int row = list->currentRow();
        if (row <= 0 || row >= static_cast<int>(steps->size())) return;
        std::swap((*steps)[static_cast<size_t>(row)], (*steps)[static_cast<size_t>(row - 1)]);
        refreshList();
        list->setCurrentRow(row - 1);
    });
    connect(btn_down, &QPushButton::clicked, dlg, [=]() {
        int row = list->currentRow();
        if (row < 0 || row + 1 >= static_cast<int>(steps->size())) return;
        std::swap((*steps)[static_cast<size_t>(row)], (*steps)[static_cast<size_t>(row + 1)]);
        refreshList();
        list->setCurrentRow(row + 1);
    });

    connect(btn_load, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getOpenFileName(dlg, "Load Chain",
            recentDir("chain"), "Chain files (*.json);;All files (*)");
        if (path.isEmpty()) return;
        updateRecentDir("chain", path);
        QString err;
        std::vector<ChainStep> loaded;
        if (!loadChain(path, loaded, err)) {
            QMessageBox::critical(dlg, "Load failed", err);
            return;
        }
        *steps = std::move(loaded);
        refreshList();
    });
    connect(btn_save, &QPushButton::clicked, dlg, [=]() {
        if (steps->empty()) return;
        QString path = QFileDialog::getSaveFileName(dlg, "Save Chain",
            recentDir("chain"), "Chain files (*.json)");
        if (path.isEmpty()) return;
        updateRecentDir("chain", path);
        QString err;
        if (!saveChain(path, *steps, err))
            QMessageBox::critical(dlg, "Save failed", err);
    });

    connect(btn_run, &QPushButton::clicked, dlg, [=]() {
        if (steps->empty()) return;
        if (!hasActiveDs() || activeDs().is_result) {
            QMessageBox::information(dlg, "Chain Editor", "No file loaded.");
            return;
        }
        int n = static_cast<int>(steps->size());
        QProgressDialog prog("Running chain…", "Cancel", 0, n, dlg);
        prog.setWindowModality(Qt::WindowModal);
        prog.setMinimumDuration(300);
        for (int i = 0; i < n; i++) {
            if (prog.wasCanceled()) return;
            prog.setLabelText(QString("Step %1/%2: %3").arg(i + 1).arg(n).arg((*steps)[static_cast<size_t>(i)].summary()));
            prog.setValue(i);
            QApplication::processEvents();
            QString err;
            if (!runChainStep((*steps)[static_cast<size_t>(i)], dlg, err)) {
                prog.close();
                QMessageBox::critical(dlg, "Chain failed",
                    QString("Step %1/%2 (%3) failed:\n%4")
                        .arg(i + 1).arg(n).arg((*steps)[static_cast<size_t>(i)].summary()).arg(err));
                return;
            }
        }
        prog.setValue(n);
        QMessageBox::information(dlg, "Chain complete", QString("Ran %1 step(s).").arg(n));
    });

    auto* btns_l = new QHBoxLayout;
    btns_l->addWidget(btn_add_step);
    btns_l->addWidget(btn_remove);
    btns_l->addWidget(btn_up);
    btns_l->addWidget(btn_down);
    btns_l->addStretch();
    btns_l->addWidget(btn_load);
    btns_l->addWidget(btn_save);
    btns_l->addWidget(btn_run);
    vl->addLayout(btns_l);

    dlg->resize(560, 420);
    dlg->show();
}

// ---------------------------------------------------------------------------
// CPA / DPA
// ---------------------------------------------------------------------------
void MainWindow::onRunCpa() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "CPA", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();
    if (h.data_length <= 0) {
        QMessageBox::warning(this, "CPA",
            "This trace set has no per-trace data (data_length = 0).\n"
            "CPA requires plaintext/ciphertext data stored in each trace.");
        return;
    }

    const int n_total   = h.num_traces;
    const int n_samples = h.num_samples;

    // ---- Step 1: Configuration dialog ----
    QDialog cfg(this);
    cfg.setWindowTitle("CPA — Configuration");
    auto* vl_cfg = new QVBoxLayout(&cfg);

    auto* grp_traces = new QGroupBox("Traces");
    auto* fl_traces  = new QFormLayout(grp_traces);
    auto* sp_first   = new QSpinBox; sp_first->setRange(0, std::max(0, n_total - 1)); sp_first->setValue(0);
    auto* sp_count   = new QSpinBox; sp_count->setRange(2, n_total); sp_count->setValue(n_total);
    fl_traces->addRow("First trace:", sp_first);
    fl_traces->addRow("Count:",       sp_count);

    auto* grp_samples = new QGroupBox("Samples");
    auto* fl_samples  = new QFormLayout(grp_samples);
    auto* sp_s_first  = new QSpinBox; sp_s_first->setRange(0, std::max(0, n_samples - 1)); sp_s_first->setValue(0);
    auto* sp_s_count  = new QSpinBox; sp_s_count->setRange(0, n_samples); sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");
    fl_samples->addRow("First sample:", sp_s_first);
    fl_samples->addRow("Count (0=all):", sp_s_count);

    auto* grp_hyp = new QGroupBox("Hypotheses");
    auto* fl_hyp  = new QFormLayout(grp_hyp);
    auto* sp_m = new QSpinBox;
    sp_m->setRange(1, 65536);
    sp_m->setValue(256);
    sp_m->setToolTip("Number of model evaluations (0 to M-1). 256 for a full AES key byte.");
    fl_hyp->addRow("M:", sp_m);

    // Alignment shifts option — populated by the last "Apply to Main View" run
    const bool has_alignment = (activeDs().align_n_samples > 0);
    const int  n_align_discarded = static_cast<int>(std::count(
        activeDs().align_shifts.begin(), activeDs().align_shifts.end(), kAlignDiscardShift));
    auto* grp_align  = new QGroupBox("Alignment");
    auto* fl_align   = new QFormLayout(grp_align);
    auto* chk_shifts = new QCheckBox("Apply last alignment shifts");
    chk_shifts->setChecked(has_alignment);
    chk_shifts->setEnabled(has_alignment);
    chk_shifts->setToolTip(has_alignment
        ? QString("Use shifts from the last alignment run (%1 traces, first_sample=%2, n_samples=%3)."
                   "%4")
              .arg(activeDs().align_shifts.size()).arg(activeDs().align_first_sample).arg(activeDs().align_n_samples)
              .arg(n_align_discarded > 0
                       ? QString(" %1 of those are marked discarded and will be skipped.").arg(n_align_discarded)
                       : QString())
        : "No alignment has been applied to the main view yet.");
    fl_align->addRow(chk_shifts);

    // When alignment is toggled, lock/unlock spinboxes and fill in the stored values.
    // Save originals so we can restore when unchecked.
    auto applyAlignmentToSpinboxes = [&](bool on) {
        sp_first ->setEnabled(!on);
        sp_count ->setEnabled(!on);
        sp_s_first->setEnabled(!on);
        sp_s_count->setEnabled(!on);
        if (on) {
            sp_first ->setValue(activeDs().align_first_trace);
            sp_count ->setValue(static_cast<int>(activeDs().align_shifts.size()));
            sp_s_first->setValue(static_cast<int>(activeDs().align_first_sample));
            sp_s_count->setValue(static_cast<int>(activeDs().align_n_samples));
        }
    };
    connect(chk_shifts, &QCheckBox::toggled, [applyAlignmentToSpinboxes](bool on) {
        applyAlignmentToSpinboxes(on);
    });
    if (has_alignment) applyAlignmentToSpinboxes(true);  // apply immediately if pre-checked

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    vl_cfg->addWidget(grp_traces);
    vl_cfg->addWidget(grp_samples);
    vl_cfg->addWidget(grp_hyp);
    vl_cfg->addWidget(grp_align);
    vl_cfg->addWidget(cfg_bb);
    if (cfg.exec() != QDialog::Accepted) return;

    const int32_t first_trace     = static_cast<int32_t>(sp_first->value());
    const int32_t num_traces      = static_cast<int32_t>(sp_count->value());
    const int64_t first_sample    = static_cast<int64_t>(sp_s_first->value());
    const int64_t num_samples_req = static_cast<int64_t>(sp_s_count->value());
    const int32_t n_hypotheses    = static_cast<int32_t>(sp_m->value());
    // When using stored alignment: override trace range and sample window
    const bool use_alignment = chk_shifts->isChecked();
    const std::vector<int32_t> use_shifts = use_alignment ? activeDs().align_shifts : std::vector<int32_t>{};
    const int32_t eff_first_trace  = use_alignment ? activeDs().align_first_trace  : first_trace;
    const int32_t eff_num_traces   = use_alignment
                                     ? static_cast<int32_t>(activeDs().align_shifts.size())
                                     : num_traces;
    const int64_t eff_first_sample = use_alignment ? activeDs().align_first_sample : first_sample;
    const int64_t eff_n_samples    = use_alignment ? activeDs().align_n_samples    : num_samples_req;

    // ---- Step 2: Initialise Python (once) ----
    {
        std::string py_err;
        if (!LeakageModel::isInitialized() && !LeakageModel::globalInit(py_err)) {
            QMessageBox::critical(this, "CPA",
                "Failed to initialise Python:\n" + QString::fromStdString(py_err));
            return;
        }
    }

    // ---- Step 3: Leakage model editor dialog ----
    LeakageModelDialog model_dlg(activeDs().file.get(), first_trace,
                                 std::min(5, num_traces), this);
    if (model_dlg.exec() != QDialog::Accepted) return;

    // Get the compiled model (re-compile if user didn't click Test first)
    LeakageModel* raw_model = model_dlg.compiledModel();
    std::unique_ptr<LeakageModel> owned_model;
    if (!raw_model) {
        owned_model = std::make_unique<LeakageModel>();
        std::string err;
        if (!owned_model->compile(model_dlg.code(), err)) {
            QMessageBox::critical(this, "CPA",
                "Failed to compile leakage model:\n" + QString::fromStdString(err));
            return;
        }
        raw_model = owned_model.get();
    }

    // Build the leakage callback that calls Python
    LeakageModel* model_ptr = raw_model;
    LeakageFn leakage_fn = [model_ptr](
        const std::vector<uint8_t>& data, int data_len,
        int n_tr, int hypothesis,
        std::vector<float>& out, std::string& err) -> bool
    {
        return model_ptr->evaluate(data, data_len, n_tr, hypothesis, out, err);
    };

    // ---- Step 4: Run CPA ----
    QProgressDialog prog("Loading traces...", "Cancel", 0, eff_num_traces + n_hypotheses, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(0);
    prog.setValue(0);
    QApplication::processEvents();

    CpaResult result;
    std::string err;

    auto progCb = [&](int32_t done, int32_t total) -> bool {
        if (prog.wasCanceled()) return false;
        prog.setMaximum(total);
        prog.setValue(done);
        if (done <= eff_num_traces)
            prog.setLabelText(QString("Loading trace %1 / %2...").arg(done).arg(eff_num_traces));
        else
            prog.setLabelText(QString("Hypothesis %1 / %2...").arg(done - eff_num_traces).arg(n_hypotheses));
        QApplication::processEvents();
        return true;
    };

    bool ok = computeCpa(activeDs().file.get(), eff_first_trace, eff_num_traces,
                         eff_first_sample, eff_n_samples,
                         n_hypotheses, use_shifts,
                         activeDs().pipeline, leakage_fn, result, progCb, err);
    prog.setValue(prog.maximum());

    if (!ok) {
        if (!err.empty())
            QMessageBox::critical(this, "CPA failed", QString::fromStdString(err));
        return;
    }

    // ---- Step 5: Result window ----
    const int32_t NS = result.n_samples;

    // Rank all hypotheses by peak absolute correlation
    struct HypPeak { int32_t hyp; float peak_r; int32_t peak_sample; };
    std::vector<HypPeak> ranked(result.n_hypotheses);
    for (int k = 0; k < result.n_hypotheses; k++) {
        const float* row = result.corr.data() + k * NS;
        float best = 0; int32_t best_s = 0;
        for (int s = 0; s < NS; s++) {
            if (std::abs(row[s]) > best) { best = std::abs(row[s]); best_s = s; }
        }
        ranked[k] = {k, best, best_s};
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const HypPeak& a, const HypPeak& b){ return a.peak_r > b.peak_r; });

    auto result_ptr = std::make_shared<CpaResult>(std::move(result));

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(QString("CPA  n=%1  M=%2  samples=%3  best=h%4 (r=%5)")
        .arg(eff_num_traces).arg(n_hypotheses).arg(NS)
        .arg(ranked[0].hyp).arg(ranked[0].peak_r, 0, 'f', 4));
    dlg->resize(1100, 720);

    auto* hm = new HeatmapWidget(dlg);
    hm->setMatrix(result_ptr->corr, result_ptr->n_hypotheses, NS);
    hm->setColorScheme(ColorScheme::RdBu);

    // Hover label
    auto* lbl_hover = new QLabel("Hover over heatmap to inspect");
    connect(hm, &HeatmapWidget::hoverInfo, dlg, [lbl_hover](int row, int col, float val) {
        if (row >= 0 && col >= 0)
            lbl_hover->setText(QString("hyp=%1  sample=%2  corr=%3")
                .arg(row).arg(col).arg(val, 0, 'g', 5));
        else
            lbl_hover->setText("Hover over heatmap to inspect");
    });

    // Top candidates table
    const int show_n = std::min<int>(static_cast<int>(ranked.size()), 16);
    const bool show_hex = (n_hypotheses <= 256);
    auto* tbl = new QTableWidget(show_n, show_hex ? 4 : 3, dlg);
    tbl->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tbl->setSelectionBehavior(QAbstractItemView::SelectRows);
    tbl->setHorizontalHeaderLabels(
        show_hex ? QStringList{"#", "Hyp", "Hex", "Peak |r|", "Sample"}
                 : QStringList{"#", "Hyp", "Peak |r|", "Sample"});
    tbl->horizontalHeader()->setStretchLastSection(false);
    tbl->verticalHeader()->hide();
    tbl->setColumnCount(show_hex ? 5 : 4);
    // rebuild header with correct count
    tbl->setHorizontalHeaderLabels(
        show_hex ? QStringList{"#", "Hyp", "Hex", "Peak |r|", "Sample"}
                 : QStringList{"#", "Hyp", "Peak |r|", "Sample"});
    for (int r = 0; r < show_n; r++) {
        const auto& p = ranked[r];
        int col = 0;
        auto* rank_item = new QTableWidgetItem(QString::number(r + 1));
        rank_item->setTextAlignment(Qt::AlignCenter);
        tbl->setItem(r, col++, rank_item);
        auto* hyp_item = new QTableWidgetItem(QString::number(p.hyp));
        hyp_item->setTextAlignment(Qt::AlignCenter);
        tbl->setItem(r, col++, hyp_item);
        if (show_hex) {
            auto* hex_item = new QTableWidgetItem(
                QString("0x%1").arg(p.hyp, 2, 16, QLatin1Char('0')).toUpper());
            hex_item->setTextAlignment(Qt::AlignCenter);
            tbl->setItem(r, col++, hex_item);
        }
        auto* r_item = new QTableWidgetItem(QString::number(static_cast<double>(p.peak_r), 'f', 4));
        r_item->setTextAlignment(Qt::AlignCenter);
        if (r == 0) {
            QFont f = r_item->font(); f.setBold(true); r_item->setFont(f);
        }
        tbl->setItem(r, col++, r_item);
        auto* s_item = new QTableWidgetItem(QString::number(p.peak_sample));
        s_item->setTextAlignment(Qt::AlignCenter);
        tbl->setItem(r, col++, s_item);
    }
    tbl->resizeColumnsToContents();
    tbl->setFixedWidth(show_hex ? 280 : 220);
    tbl->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

    // Color range controls
    auto* sp_vmin = new QDoubleSpinBox; sp_vmin->setRange(-1e9, 1e9); sp_vmin->setDecimals(4); sp_vmin->setValue(-1.0);
    auto* sp_vmax = new QDoubleSpinBox; sp_vmax->setRange(-1e9, 1e9); sp_vmax->setDecimals(4); sp_vmax->setValue(1.0);
    {
        float dmin = result_ptr->corr[0], dmax = result_ptr->corr[0];
        for (float v : result_ptr->corr) { dmin = std::min(dmin, v); dmax = std::max(dmax, v); }
        float lim = std::max(std::abs(dmin), std::abs(dmax));
        sp_vmin->setValue(-lim); sp_vmax->setValue(lim);
        hm->setColorRange(-lim, lim);
    }
    connect(sp_vmin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [hm, sp_vmax](double v) { hm->setColorRange(v, sp_vmax->value()); });
    connect(sp_vmax, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [hm, sp_vmin](double v) { hm->setColorRange(sp_vmin->value(), v); });

    // Buttons
    auto* btn_show_key = new QPushButton("Show corr traces...");
    auto* btn_exp_npy  = new QPushButton("Export .npy...");
    auto* btn_close    = new QPushButton("Close");

    // Color scheme combo
    auto* combo_scheme = new QComboBox;
    combo_scheme->addItems({"RdBu", "Grayscale", "Hot", "Viridis", "Plasma"});
    connect(combo_scheme, QOverload<int>::of(&QComboBox::currentIndexChanged), [hm](int i) {
        hm->setColorScheme(static_cast<ColorScheme>(i));
    });

    // Abs value checkbox
    auto* chk_abs = new QCheckBox("Abs value");
    connect(chk_abs, &QCheckBox::toggled, [hm, sp_vmin, sp_vmax](bool on) {
        hm->setAbsValue(on);
        if (on) { sp_vmin->setValue(0); }
    });

    // Show correlation traces for each hypothesis
    connect(btn_show_key, &QPushButton::clicked, dlg, [this, result_ptr, NS, dlg]() {
        auto* td = new QDialog(dlg);
        td->setAttribute(Qt::WA_DeleteOnClose);
        td->setWindowTitle("CPA — Correlation traces per hypothesis");
        td->resize(900, 500);
        auto* pw = new PlotWidget(td);
        const float* mat = result_ptr->corr.data();
        for (int k = 0; k < result_ptr->n_hypotheses; k++) {
            auto trace = std::make_shared<std::vector<float>>(mat + k * NS, mat + k * NS + NS);
            QColor c = TRACE_COLORS[k % NUM_COLORS];
            c.setAlpha(180);
            pw->addTrace(std::move(trace), c, QString("h%1").arg(k));
        }
        pw->setAxisLabels("Sample index", "Hypothesis");
        auto* vl = new QVBoxLayout(td);
        vl->addWidget(pw);
        td->show();
    });

    // Export .npy
    connect(btn_exp_npy, &QPushButton::clicked, dlg, [this, result_ptr, NS]() {
        QString path = QFileDialog::getSaveFileName(this, "Export CPA result", MainWindow::recentDir("npy"), "NumPy (*.npy)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("npy", path);
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly)) return;
        // Write NPY v1.0 header for float32 array shape (M, NS)
        QString desc = QString("{'descr': '<f4', 'fortran_order': False, 'shape': (%1, %2), }").arg(result_ptr->n_hypotheses).arg(NS);
        while ((10 + desc.size()) % 64 != 0) desc += ' ';
        QByteArray hdr;
        hdr.append("\x93NUMPY"); hdr.append('\x01'); hdr.append('\x00');
        uint16_t hlen = static_cast<uint16_t>(desc.size());
        hdr.append(static_cast<char>(hlen & 0xFF)); hdr.append(static_cast<char>(hlen >> 8));
        hdr.append(desc.toLatin1());
        f.write(hdr);
        f.write(reinterpret_cast<const char*>(result_ptr->corr.data()),
                static_cast<qint64>(result_ptr->corr.size() * sizeof(float)));
    });

    connect(btn_close, &QPushButton::clicked, dlg, &QDialog::close);

    // Control row (above heatmap)
    auto* ctrl = new QHBoxLayout;
    ctrl->addWidget(lbl_hover);
    ctrl->addStretch();
    ctrl->addWidget(new QLabel("Min:")); ctrl->addWidget(sp_vmin);
    ctrl->addWidget(new QLabel("Max:")); ctrl->addWidget(sp_vmax);
    ctrl->addWidget(combo_scheme);
    ctrl->addWidget(chk_abs);
    ctrl->addWidget(btn_show_key);
    ctrl->addWidget(btn_exp_npy);
    ctrl->addWidget(btn_close);

    // Left side: ctrl + heatmap
    auto* left_widget = new QWidget(dlg);
    auto* left_vl = new QVBoxLayout(left_widget);
    left_vl->setContentsMargins(0, 0, 0, 0);
    left_vl->addLayout(ctrl);
    left_vl->addWidget(hm, 1);

    // Right side: top candidates table
    auto* right_widget = new QWidget(dlg);
    auto* right_vl = new QVBoxLayout(right_widget);
    right_vl->setContentsMargins(4, 0, 0, 0);
    auto* lbl_top = new QLabel(
        QString("<b>Top %1 candidates</b>").arg(show_n));
    lbl_top->setTextFormat(Qt::RichText);
    right_vl->addWidget(lbl_top);
    right_vl->addWidget(tbl, 1);

    auto* splitter = new QSplitter(Qt::Horizontal, dlg);
    splitter->addWidget(left_widget);
    splitter->addWidget(right_widget);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 0);

    auto* vl_dlg = new QVBoxLayout(dlg);
    vl_dlg->addWidget(splitter, 1);

    dlg->show();
}

// ---------------------------------------------------------------------------
// SNR
// ---------------------------------------------------------------------------

void MainWindow::onRunSNR() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "SNR", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();
    if (h.data_length <= 0) {
        QMessageBox::critical(this, "SNR",
            "This TRS file has no per-trace data bytes.\n"
            "SNR requires at least 1 data byte per trace for class labels.");
        return;
    }

    // --- Configuration dialog ---
    int n_total = h.num_traces;
    QDialog cfg(this);
    cfg.setWindowTitle("SNR — configuration");
    auto* fl       = new QFormLayout(&cfg);

    auto* sp_first  = new QSpinBox; sp_first->setRange(0, std::max(0, n_total-1)); sp_first->setValue(0);
    auto* sp_count  = new QSpinBox; sp_count->setRange(2, n_total);                sp_count->setValue(n_total);
    auto* sp_s_first = new QSpinBox; sp_s_first->setRange(0, std::max(0,(int)h.num_samples-1)); sp_s_first->setValue(0);
    auto* sp_s_count = new QSpinBox; sp_s_count->setRange(0, (int)h.num_samples);  sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");

    fl->addRow("First trace:",            sp_first);
    fl->addRow("Count:",                  sp_count);
    fl->addRow("First sample:",           sp_s_first);
    fl->addRow("Sample count (0=all):",   sp_s_count);

    auto* sp_byte = new QSpinBox;
    sp_byte->setRange(0, h.data_length - 1);
    sp_byte->setValue(0);
    sp_byte->setToolTip("Data byte used to derive the class label.");
    fl->addRow("Data byte index:", sp_byte);

    auto* cmb_mode = new QComboBox;
    cmb_mode->addItem("Raw byte value  (256 classes, 0–255)");
    cmb_mode->addItem("Hamming weight  (9 classes, 0–8)");
    cmb_mode->addItem("AES S-box output  (256 classes)");
    cmb_mode->addItem("AES S-box output — Hamming weight  (9 classes)");
    fl->addRow("Class mode:", cmb_mode);

    auto* sp_key = new QSpinBox;
    sp_key->setRange(0, 255);
    sp_key->setValue(0);
    sp_key->setEnabled(false);
    sp_key->setToolTip("Key byte hypothesis used as XOR input to the S-box.");
    fl->addRow("Key byte hypothesis:", sp_key);

    connect(cmb_mode, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int idx) {
        sp_key->setEnabled(idx == 2 || idx == 3);
    });

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow(cfg_bb);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    const int32_t first    = sp_first->value();
    const int32_t count    = sp_count->value();
    const int32_t byte_idx = sp_byte->value();
    const int     mode     = cmb_mode->currentIndex();
    const uint8_t key_byte = static_cast<uint8_t>(sp_key->value());
    const bool    use_hw   = (mode == 1 || mode == 3);
    const bool    use_sbox = (mode == 2 || mode == 3);
    const int32_t n_classes = use_hw ? 9 : 256;

    const int64_t eff_first_sample = sp_s_first->value();
    const int64_t eff_n_samples    = sp_s_count->value();
    const int64_t raw_ns = (eff_n_samples == 0)
        ? (h.num_samples - eff_first_sample)
        : std::min<int64_t>(eff_n_samples, h.num_samples - eff_first_sample);

    int64_t effective_samples = raw_ns;
    for (const auto& t : activeDs().pipeline)
        effective_samples = t->transformedCount(effective_samples);

    // Memory warning: n_classes * 2 * effective_samples * sizeof(double)
    int64_t mem_bytes = static_cast<int64_t>(n_classes) * 2LL
                      * effective_samples * static_cast<int64_t>(sizeof(double));
    if (mem_bytes > 2LL * 1024 * 1024 * 1024) {
        if (QMessageBox::warning(this, "Memory warning",
                QString("Accumulator will require ~%1 GB.\nContinue?")
                    .arg(double(mem_bytes) / (1024.0*1024*1024), 0, 'f', 1),
                QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
            return;
    }

    // --- Accumulation ---
    SNRAccumulator acc(static_cast<int32_t>(effective_samples), n_classes);

    QProgressDialog prog("Accumulating traces…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    std::vector<float> trace_buf(static_cast<size_t>(std::max(raw_ns, effective_samples)));
    int32_t skipped = 0;

    static const int32_t HW_TABLE[256] = {
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4, 1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, 2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, 2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, 3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5, 2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, 3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6, 3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
        3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7, 4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
    };
    static const uint8_t AES_SBOX[256] = {
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
    };

    for (int32_t ti = 0; ti < count; ti++) {
        if (prog.wasCanceled()) return;
        prog.setValue(ti);
        QApplication::processEvents();

        int32_t src_idx  = first + ti;
        auto data_bytes  = activeDs().file->readData(src_idx);
        if (byte_idx >= static_cast<int32_t>(data_bytes.size())) { skipped++; continue; }

        uint8_t  bval    = data_bytes[byte_idx];
        if (use_sbox) bval = AES_SBOX[bval ^ key_byte];
        int32_t  label   = use_hw ? HW_TABLE[bval] : static_cast<int32_t>(bval);

        const int64_t adj_start = eff_first_sample;
        std::fill(trace_buf.begin(), trace_buf.end(), 0.0f);
        if (adj_start < h.num_samples && adj_start + raw_ns > 0) {
            int64_t src_start = std::max<int64_t>(0, adj_start);
            int64_t src_end   = std::min<int64_t>(h.num_samples, adj_start + raw_ns);
            int64_t dst_off   = src_start - adj_start;
            int64_t got = activeDs().file->readSamples(src_idx, src_start, src_end - src_start,
                                                  trace_buf.data() + dst_off);
            if (got <= 0) { skipped++; continue; }
        }
        for (const auto& t : activeDs().pipeline) t->reset();
        int64_t n_out = raw_ns;
        for (const auto& t : activeDs().pipeline)
            n_out = t->apply(trace_buf.data(), n_out, 0);
        acc.addTrace(label, trace_buf.data(), static_cast<int32_t>(n_out));
    }
    prog.setValue(count);

    if (skipped > 0)
        QMessageBox::warning(this, "SNR",
            QString("%1 traces skipped (data byte out of range).").arg(skipped));

    // --- Compute ---
    std::vector<float> snr;
    std::string        err;
    if (!acc.compute(snr, err)) {
        QMessageBox::critical(this, "SNR failed", QString::fromStdString(err));
        return;
    }

    // --- Result tab ---
    auto snr_ptr = std::make_shared<std::vector<float>>(std::move(snr));

    QString mode_str;
    if      (mode == 0) mode_str = "raw byte";
    else if (mode == 1) mode_str = "HW(byte)";
    else if (mode == 2) mode_str = QString("SBox(byte⊕0x%1)").arg(key_byte, 2, 16, QChar('0'));
    else                mode_str = QString("HW(SBox(byte⊕0x%1))").arg(key_byte, 2, 16, QChar('0'));
    QString tab_title = QString("SNR — %1 traces, %2 (%3 classes)")
                            .arg(acc.totalTraces())
                            .arg(mode_str)
                            .arg(n_classes);
    addResultTab(*snr_ptr, tab_title, QColor("#f4a63a"), "SNR");
    PlotWidget* pw = plotWidget();
    pw->setTraceFilled(0, true);
    pw->setAxisLabels("Sample Index", "SNR");
    pw->setThresholds(false, 0.0, 0.0);
    pw->resetView();

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, pw, [this, snr_ptr]() {
        QString path = QFileDialog::getSaveFileName(this, "Export SNR as NumPy",
                                                    MainWindow::recentDir("npy"), "NumPy files (*.npy)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("npy", path);
        QString e;
        if (!saveNpy(path, snr_ptr->data(), static_cast<int64_t>(snr_ptr->size()), e))
            QMessageBox::critical(this, "Export failed", e);
        else
            QMessageBox::information(this, "Export complete", "Saved: " + path);
    });

    auto* btn_exp_pdf = new QPushButton("Export PDF…");
    connect(btn_exp_pdf, &QPushButton::clicked, pw, [=]() {
        QString path = QFileDialog::getSaveFileName(this, "Export SNR as PDF",
                                                    MainWindow::recentDir("export"), "PDF files (*.pdf)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPixmap px = pw->grab();
        QPdfWriter writer(path);
        writer.setResolution(150);
        writer.setPageSize(QPageSize(QPageSize::A4));
        writer.setPageOrientation(QPageLayout::Landscape);
        writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout::Millimeter);
        QPainter painter(&writer);
        if (painter.isActive())
            painter.drawPixmap(painter.viewport(), px);
    });

    auto* extra = new QWidget;
    auto* hl = new QHBoxLayout(extra);
    hl->setContentsMargins(4, 2, 4, 2);
    hl->addWidget(btn_exp_npy);
    hl->addWidget(btn_exp_pdf);
    hl->addStretch();

    activeDs().extra_toolbar = extra;
    updateViewLayout();
}

// ---------------------------------------------------------------------------
// Static SNR  |μ[s] / σ[s]|
// Run the same operation N times; σ is purely electronic/thermal noise.
// ---------------------------------------------------------------------------

void MainWindow::onRunStaticSNR() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Static SNR", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();

    // --- Configuration dialog ---
    int n_total = h.num_traces;
    QDialog cfg(this);
    cfg.setWindowTitle("Static SNR |μ/σ| — configuration");
    auto* fl = new QFormLayout(&cfg);

    auto* sp_first   = new QSpinBox; sp_first->setRange(0, std::max(0, n_total-1)); sp_first->setValue(0);
    auto* sp_count   = new QSpinBox; sp_count->setRange(2, n_total);                sp_count->setValue(n_total);
    auto* sp_s_first = new QSpinBox; sp_s_first->setRange(0, std::max(0,(int)h.num_samples-1)); sp_s_first->setValue(0);
    auto* sp_s_count = new QSpinBox; sp_s_count->setRange(0, (int)h.num_samples);  sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");

    fl->addRow("First trace:",          sp_first);
    fl->addRow("Count:",                sp_count);
    fl->addRow("First sample:",         sp_s_first);
    fl->addRow("Sample count (0=all):", sp_s_count);

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow(cfg_bb);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    const int32_t first = sp_first->value();
    const int32_t count = sp_count->value();
    const int64_t eff_first_sample = sp_s_first->value();
    const int64_t eff_n_samples    = sp_s_count->value();
    const int64_t raw_ns = (eff_n_samples == 0)
        ? (h.num_samples - eff_first_sample)
        : std::min<int64_t>(eff_n_samples, h.num_samples - eff_first_sample);

    int64_t effective_samples = raw_ns;
    for (const auto& t : activeDs().pipeline)
        effective_samples = t->transformedCount(effective_samples);

    // --- Accumulate sum and sum-of-squares ---
    std::vector<double> sum(static_cast<size_t>(effective_samples), 0.0);
    std::vector<double> sum2(static_cast<size_t>(effective_samples), 0.0);
    int64_t N = 0;

    QProgressDialog prog("Accumulating traces…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    std::vector<float> trace_buf(static_cast<size_t>(std::max(raw_ns, effective_samples)));

    for (int32_t ti = 0; ti < count; ti++) {
        if (prog.wasCanceled()) return;
        prog.setValue(ti);
        QApplication::processEvents();

        int32_t src_idx = first + ti;
        std::fill(trace_buf.begin(), trace_buf.end(), 0.0f);
        const int64_t adj_start = eff_first_sample;
        if (adj_start < h.num_samples && adj_start + raw_ns > 0) {
            int64_t src_start = std::max<int64_t>(0, adj_start);
            int64_t src_end   = std::min<int64_t>(h.num_samples, adj_start + raw_ns);
            int64_t dst_off   = src_start - adj_start;
            int64_t got = activeDs().file->readSamples(src_idx, src_start, src_end - src_start,
                                                  trace_buf.data() + dst_off);
            if (got <= 0) continue;
        }
        for (const auto& t : activeDs().pipeline) t->reset();
        int64_t n_out = raw_ns;
        for (const auto& t : activeDs().pipeline)
            n_out = t->apply(trace_buf.data(), n_out, 0);

        for (int64_t s = 0; s < n_out; s++) {
            double v = static_cast<double>(trace_buf[s]);
            sum[s]  += v;
            sum2[s] += v * v;
        }
        N++;
    }
    prog.setValue(count);

    if (N < 2) {
        QMessageBox::critical(this, "Static SNR", "Need at least 2 traces.");
        return;
    }

    // --- Compute |μ / σ| per sample ---
    std::vector<float> snr(static_cast<size_t>(effective_samples));
    double nd = static_cast<double>(N);
    for (int64_t s = 0; s < effective_samples; s++) {
        double mean = sum[s] / nd;
        double var  = (sum2[s] - sum[s] * mean) / (nd - 1.0);
        if (var < 0.0) var = 0.0;
        double sigma = std::sqrt(var);
        snr[s] = (sigma > 0.0) ? static_cast<float>(std::abs(mean) / sigma) : 0.0f;
    }

    // --- Result window ---
    float snr_min = *std::min_element(snr.begin(), snr.end());
    float snr_max = *std::max_element(snr.begin(), snr.end());
    double snr_sum = std::accumulate(snr.begin(), snr.end(), 0.0);
    float snr_avg = snr.empty() ? 0.f : static_cast<float>(snr_sum / snr.size());

    auto snr_ptr = std::make_shared<std::vector<float>>(std::move(snr));

    QString tab_title = QString("Static SNR |μ/σ| — %1 traces").arg(N);
    addResultTab(*snr_ptr, tab_title, QColor("#4fc3f7"), "|μ/σ|");
    PlotWidget* pw = plotWidget();
    pw->setTraceFilled(0, true);
    pw->setAxisLabels("Sample Index", "|μ/σ|");
    pw->setThresholds(false, 0.0, 0.0);
    pw->resetView();

    auto* lbl_stats = new QLabel(
        QString("Min: <b>%1</b>  Max: <b>%2</b>  Avg: <b>%3</b>")
            .arg(static_cast<double>(snr_min), 0, 'f', 4)
            .arg(static_cast<double>(snr_max), 0, 'f', 4)
            .arg(static_cast<double>(snr_avg), 0, 'f', 4));
    lbl_stats->setTextFormat(Qt::RichText);

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, pw, [this, snr_ptr]() {
        QString path = QFileDialog::getSaveFileName(this, "Export static SNR as NumPy",
                                                    MainWindow::recentDir("npy"), "NumPy files (*.npy)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("npy", path);
        QString e;
        if (!saveNpy(path, snr_ptr->data(), static_cast<int64_t>(snr_ptr->size()), e))
            QMessageBox::critical(this, "Export failed", e);
        else
            QMessageBox::information(this, "Export complete", "Saved: " + path);
    });

    auto* btn_exp_pdf = new QPushButton("Export PDF…");
    connect(btn_exp_pdf, &QPushButton::clicked, pw, [=]() {
        QString path = QFileDialog::getSaveFileName(this, "Export static SNR as PDF",
                                                    MainWindow::recentDir("export"), "PDF files (*.pdf)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPixmap px = pw->grab();
        QPdfWriter writer(path);
        writer.setResolution(150);
        writer.setPageSize(QPageSize(QPageSize::A4));
        writer.setPageOrientation(QPageLayout::Landscape);
        writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout::Millimeter);
        QPainter painter(&writer);
        if (painter.isActive())
            painter.drawPixmap(painter.viewport(), px);
    });

    auto* extra = new QWidget;
    auto* hl = new QHBoxLayout(extra);
    hl->setContentsMargins(4, 2, 4, 2);
    hl->addWidget(lbl_stats);
    hl->addStretch();
    hl->addWidget(btn_exp_npy);
    hl->addWidget(btn_exp_pdf);

    activeDs().extra_toolbar = extra;
    updateViewLayout();
}

// ---------------------------------------------------------------------------
// FFT Spectrum
// ---------------------------------------------------------------------------

void MainWindow::onRunFFT() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "FFT", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();

    // --- Configuration dialog ---
    const int n_total = h.num_traces;
    QDialog cfg(this);
    cfg.setWindowTitle("FFT Spectrum — configuration");
    auto* fl = new QFormLayout(&cfg);

    auto* sp_first   = new QSpinBox; sp_first->setRange(0, std::max(0, n_total-1)); sp_first->setValue(0);
    auto* sp_count   = new QSpinBox; sp_count->setRange(1, n_total);                sp_count->setValue(n_total);
    auto* sp_s_first = new QSpinBox; sp_s_first->setRange(0, std::max(0,(int)h.num_samples-1)); sp_s_first->setValue(0);
    auto* sp_s_count = new QSpinBox; sp_s_count->setRange(0, (int)h.num_samples);  sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");

    fl->addRow("First trace:",          sp_first);
    fl->addRow("Count:",                sp_count);
    fl->addRow("First sample:",         sp_s_first);
    fl->addRow("Sample count (0=all):", sp_s_count);

    auto* cmb_window = new QComboBox;
    cmb_window->addItem("None (rectangular)");
    cmb_window->addItem("Hann");
    cmb_window->addItem("Hamming");
    cmb_window->addItem("Blackman");
    cmb_window->setCurrentIndex(1);  // default: Hann
    fl->addRow("Window function:", cmb_window);

    auto* cmb_output = new QComboBox;
    cmb_output->addItem("Magnitude");
    cmb_output->addItem("Magnitude (dB)");
    cmb_output->addItem("Phase (rad)");
    fl->addRow("Output:", cmb_output);

    auto* chk_envelope = new QCheckBox("Show min/max envelope");
    chk_envelope->setChecked(false);
    fl->addRow("", chk_envelope);

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow(cfg_bb);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    const int32_t first        = sp_first->value();
    const int32_t count        = sp_count->value();
    const int64_t eff_first_sample = sp_s_first->value();
    const int64_t eff_n_samples    = sp_s_count->value();
    const int      win_mode    = cmb_window->currentIndex();
    const int      out_mode    = cmb_output->currentIndex();
    const bool     show_env    = chk_envelope->isChecked();

    const int64_t raw_ns = (eff_n_samples == 0)
        ? (h.num_samples - eff_first_sample)
        : std::min<int64_t>(eff_n_samples, h.num_samples - eff_first_sample);

    int64_t effective_samples = raw_ns;
    for (const auto& t : activeDs().pipeline)
        effective_samples = t->transformedCount(effective_samples);

    if (effective_samples < 2) {
        QMessageBox::critical(this, "FFT", "Too few samples after pipeline.");
        return;
    }

    // Number of FFT output bins (one-sided: N/2+1).
    const int64_t fft_in_n  = effective_samples;
    const int64_t fft_out_n = fft_in_n / 2 + 1;

    // Build window coefficients.
    std::vector<float> window(static_cast<size_t>(fft_in_n));
    {
        const double N1 = static_cast<double>(fft_in_n - 1);
        for (int64_t i = 0; i < fft_in_n; ++i) {
            double w = 1.0;
            const double phi = 2.0 * M_PI * i / N1;
            if (win_mode == 1)      w = 0.5 * (1.0 - std::cos(phi));           // Hann
            else if (win_mode == 2) w = 0.54 - 0.46 * std::cos(phi);           // Hamming
            else if (win_mode == 3) w = 0.42 - 0.5*std::cos(phi) + 0.08*std::cos(2.0*phi); // Blackman
            window[i] = static_cast<float>(w);
        }
    }

    // Accumulators for average, min, max.
    std::vector<double> acc_sum(static_cast<size_t>(fft_out_n), 0.0);
    std::vector<float>  acc_min(static_cast<size_t>(fft_out_n),  std::numeric_limits<float>::max());
    std::vector<float>  acc_max(static_cast<size_t>(fft_out_n), -std::numeric_limits<float>::max());
    int64_t n_ok = 0;

    QProgressDialog prog("Computing FFT…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    std::vector<float> trace_buf(static_cast<size_t>(std::max(raw_ns, effective_samples)));
    Eigen::FFT<float>  fft_engine;
    fft_engine.SetFlag(Eigen::FFT<float>::HalfSpectrum);  // only positive freqs
    std::vector<std::complex<float>> freq_buf;

    for (int32_t ti = 0; ti < count; ++ti) {
        if (prog.wasCanceled()) return;
        prog.setValue(ti);
        QApplication::processEvents();

        int32_t src_idx = first + ti;
        std::fill(trace_buf.begin(), trace_buf.end(), 0.0f);
        if (eff_first_sample < h.num_samples && eff_first_sample + raw_ns > 0) {
            int64_t src_start = std::max<int64_t>(0, eff_first_sample);
            int64_t src_end   = std::min<int64_t>(h.num_samples, eff_first_sample + raw_ns);
            int64_t dst_off   = src_start - eff_first_sample;
            int64_t got = activeDs().file->readSamples(src_idx, src_start, src_end - src_start,
                                                  trace_buf.data() + dst_off);
            if (got <= 0) continue;
        }

        for (const auto& t : activeDs().pipeline) t->reset();
        int64_t n_out = raw_ns;
        for (const auto& t : activeDs().pipeline)
            n_out = t->apply(trace_buf.data(), n_out, 0);

        if (n_out < 2) continue;

        // Apply window (in-place).
        for (int64_t s = 0; s < n_out; ++s)
            trace_buf[s] *= window[s];

        // FFT.
        std::vector<float> in_vec(trace_buf.begin(), trace_buf.begin() + n_out);
        fft_engine.fwd(freq_buf, in_vec);

        // Accumulate magnitude / phase.
        const float norm = 1.0f / static_cast<float>(n_out);
        for (int64_t k = 0; k < fft_out_n; ++k) {
            float val;
            if (out_mode == 2) {
                val = std::arg(freq_buf[k]);
            } else {
                float mag = std::abs(freq_buf[k]) * norm;
                // Double all bins except DC and Nyquist to get one-sided amplitude.
                if (k > 0 && k < fft_out_n - 1) mag *= 2.0f;
                if (out_mode == 1) {
                    val = (mag > 0.0f) ? 20.0f * std::log10(mag) : -200.0f;
                } else {
                    val = mag;
                }
            }
            acc_sum[k] += static_cast<double>(val);
            if (val < acc_min[k]) acc_min[k] = val;
            if (val > acc_max[k]) acc_max[k] = val;
        }
        ++n_ok;
    }
    prog.setValue(count);

    if (n_ok == 0) {
        QMessageBox::critical(this, "FFT", "No traces could be processed.");
        return;
    }

    // Build average spectrum.
    auto avg_ptr = std::make_shared<std::vector<float>>(static_cast<size_t>(fft_out_n));
    for (int64_t k = 0; k < fft_out_n; ++k)
        (*avg_ptr)[k] = static_cast<float>(acc_sum[k] / static_cast<double>(n_ok));

    // Build x-axis label and frequency axis (for display only via setAxisLabels).
    // scale_x is seconds/sample in TRS files; if meaningful, show Hz.
    const float scale_x = h.scale_x;
    const bool  has_freq_axis = (scale_x > 0.0f && scale_x < 1.0f);  // typical oscilloscope range
    QString x_label = has_freq_axis ? "Frequency (Hz)" : "Frequency Bin";
    QString y_label;
    if (out_mode == 0) y_label = "Amplitude";
    else if (out_mode == 1) y_label = "Magnitude (dB)";
    else y_label = "Phase (rad)";

    // If we have a real frequency axis, scale the avg data x-range via a dummy
    // transform is not needed — PlotWidget currently uses sample-index x-axis.
    // We store the frequency values as a second shared_ptr and display them;
    // the user can read the Hz ticks from the axis once we set the x-scale.
    // PlotWidget supports setXScale(double scale, double offset) via resetView.
    // For now, pass the frequency-per-bin scale to PlotWidget's x-axis.

    // --- Result window ---
    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(QString("FFT Spectrum — %1 traces, %2")
                            .arg(n_ok)
                            .arg(cmb_window->currentText()));
    dlg->resize(1200, 560);

    auto* pw = new PlotWidget(dlg);

    // Set x-axis scale so ticks show Hz instead of bin indices.
    if (has_freq_axis) {
        const double fs   = 1.0 / static_cast<double>(scale_x);   // sample rate Hz
        const double df   = fs / static_cast<double>(fft_in_n);    // Hz per bin
        pw->setXScale(df, 0.0);
    }

    if (show_env && n_ok > 1) {
        auto min_ptr = std::make_shared<std::vector<float>>(acc_min);
        auto max_ptr = std::make_shared<std::vector<float>>(acc_max);
        pw->addTrace(min_ptr, QColor("#5c9bd6"), "Min");
        pw->addTrace(max_ptr, QColor("#5c9bd6"), "Max");
        pw->addTrace(avg_ptr, QColor("#f4a63a"), "Average");
    } else {
        pw->addTrace(avg_ptr, QColor("#f4a63a"), "Average");
        pw->setTraceFilled(0, true);
    }

    pw->setAxisLabels(x_label, y_label);
    pw->setThresholds(false, 0.0, 0.0);
    pw->resetView();

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, dlg, [dlg, avg_ptr]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export FFT spectrum as NumPy",
                                                    MainWindow::recentDir("npy"), "NumPy files (*.npy)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("npy", path);
        QString e;
        if (!saveNpy(path, avg_ptr->data(), static_cast<int64_t>(avg_ptr->size()), e))
            QMessageBox::critical(dlg, "Export failed", e);
        else
            QMessageBox::information(dlg, "Export complete", "Saved: " + path);
    });

    auto* btn_exp_pdf = new QPushButton("Export PDF…");
    connect(btn_exp_pdf, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export FFT spectrum as PDF",
                                                    MainWindow::recentDir("export"), "PDF files (*.pdf)");
        if (path.isEmpty()) return;
        MainWindow::updateRecentDir("export", path);
        QPixmap px = pw->grab();
        QPdfWriter writer(path);
        writer.setResolution(150);
        writer.setPageSize(QPageSize(QPageSize::A4));
        writer.setPageOrientation(QPageLayout::Landscape);
        writer.setPageMargins(QMarginsF(10, 10, 10, 10), QPageLayout::Millimeter);
        QPainter painter(&writer);
        if (painter.isActive())
            painter.drawPixmap(painter.viewport(), px);
    });

    auto* hl = new QHBoxLayout;
    hl->addWidget(btn_exp_npy);
    hl->addWidget(btn_exp_pdf);
    hl->addStretch();

    auto* vl = new QVBoxLayout(dlg);
    vl->addWidget(pw, 1);
    vl->addLayout(hl);

    dlg->show();
}

// ---------------------------------------------------------------------------
// Dataset export
// ---------------------------------------------------------------------------

struct LabelDef {
    enum Type { RawValue, HammingWeight, Bit, BitRange };

    QString name;
    Type    type     = RawValue;
    int     byte_off = 0;
    int     byte_cnt = 1;    // bytes to read (RawValue, HW, BitRange)
    bool    big_end  = false; // endianness (RawValue, BitRange)
    int     bit_lo   = 0;    // Bit: bit index (0=LSB); BitRange: low bit (inclusive)
    int     bit_hi   = 0;    // BitRange: high bit (inclusive)

    static size_t bitsToElemSize(int bits) {
        if (bits <=  8) return 1;
        if (bits <= 16) return 2;
        if (bits <= 32) return 4;
        return 8;
    }
    static std::string bitsToNpyDtype(int bits) {
        if (bits <=  8) return "|u1";
        if (bits <= 16) return "<u2";
        if (bits <= 32) return "<u4";
        return "<u8";
    }

    std::string dtype() const {
        switch (type) {
            case HammingWeight: case Bit: return "|u1";
            case BitRange:  return bitsToNpyDtype(bit_hi - bit_lo + 1);
            case RawValue:
                switch (byte_cnt) {
                    case 1: return "|u1"; case 2: return "<u2";
                    case 3: case 4: return "<u4"; default: return "<u8";
                }
        }
        return "|u1";
    }
    size_t elem_size() const {
        switch (type) {
            case HammingWeight: case Bit: return 1;
            case BitRange:  return bitsToElemSize(bit_hi - bit_lo + 1);
            case RawValue:
                switch (byte_cnt) {
                    case 1: return 1; case 2: return 2;
                    case 3: case 4: return 4; default: return 8;
                }
        }
        return 1;
    }

    uint64_t readInt(const std::vector<uint8_t>& data, int off, int cnt, bool be) const {
        int avail = static_cast<int>(data.size());
        int n = std::min(cnt, avail - off);
        if (n <= 0) return 0;
        uint64_t val = 0;
        if (be)
            for (int i = 0; i < n; i++)      val = (val << 8) | data[off + i];
        else
            for (int i = n - 1; i >= 0; i--) val = (val << 8) | data[off + i];
        return val;
    }

    uint64_t compute(const std::vector<uint8_t>& data) const {
        int avail = static_cast<int>(data.size());
        if (byte_off >= avail) return 0;
        switch (type) {
            case Bit:
                return (data[byte_off] >> bit_lo) & 1u;
            case HammingWeight: {
                int cnt = std::min(byte_cnt, avail - byte_off);
                uint32_t hw = 0;
                for (int i = 0; i < cnt; i++)
                    hw += static_cast<uint32_t>(__builtin_popcount(data[byte_off + i]));
                return hw;
            }
            case BitRange: {
                uint64_t val  = readInt(data, byte_off, byte_cnt, big_end);
                int      width = bit_hi - bit_lo + 1;
                uint64_t mask  = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
                return (val >> bit_lo) & mask;
            }
            case RawValue:
                return readInt(data, byte_off, byte_cnt, big_end);
        }
        return 0;
    }

    QString summary() const {
        switch (type) {
            case RawValue:
                return QString("[raw %1B %2] %3  @  byte %4")
                    .arg(byte_cnt).arg(big_end?"BE":"LE").arg(name).arg(byte_off);
            case HammingWeight:
                return QString("[HW %1B] %2  @  byte %3").arg(byte_cnt).arg(name).arg(byte_off);
            case Bit:
                return QString("[bit%1] %2  @  byte %3").arg(bit_lo).arg(name).arg(byte_off);
            case BitRange:
                return QString("[bits %1:%2, %3B %4] %5  @  byte %6")
                    .arg(bit_hi).arg(bit_lo).arg(byte_cnt)
                    .arg(big_end?"BE":"LE").arg(name).arg(byte_off);
        }
        return name;
    }
};

static bool runAddLabelDialog(QWidget* parent, int max_byte, LabelDef& def)
{
    QDialog d(parent);
    d.setWindowTitle("Define Label");
    d.setMinimumWidth(340);
    auto* fl = new QFormLayout(&d);

    auto* le_name  = new QLineEdit(def.name);
    auto* cmb_type = new QComboBox;
    cmb_type->addItems({
        "Raw integer  (N bytes, LE/BE)",
        "Hamming weight  (popcount of N bytes)",
        "Single bit  (0 or 1)",
        "Bit range  [hi:lo] extracted from N bytes",
    });
    cmb_type->setCurrentIndex(static_cast<int>(def.type));

    auto* sp_off = new QSpinBox;
    sp_off->setRange(0, std::max(0, max_byte - 1));
    sp_off->setValue(def.byte_off);

    // byte count (Raw / HW / BitRange)
    auto* lbl_cnt = new QLabel("Byte count:");
    auto* sp_cnt  = new QSpinBox; sp_cnt->setRange(1, 8); sp_cnt->setValue(std::max(1, def.byte_cnt));

    // endianness (Raw / BitRange)
    auto* lbl_end = new QLabel("Endianness:");
    auto* cmb_end = new QComboBox; cmb_end->addItems({"little-endian", "big-endian"});
    cmb_end->setCurrentIndex(def.big_end ? 1 : 0);

    // Single bit
    auto* lbl_bit = new QLabel("Bit index (0=LSB):");
    auto* sp_bit  = new QSpinBox; sp_bit->setRange(0, 7); sp_bit->setValue(def.bit_lo);

    // Bit range lo/hi
    auto* lbl_blo = new QLabel("Low bit (0=LSB, inclusive):");
    auto* sp_blo  = new QSpinBox; sp_blo->setRange(0, 63); sp_blo->setValue(def.bit_lo);
    auto* lbl_bhi = new QLabel("High bit (inclusive):");
    auto* sp_bhi  = new QSpinBox; sp_bhi->setRange(0, 63); sp_bhi->setValue(std::max(def.bit_hi, def.bit_lo));
    auto* lbl_bw  = new QLabel;  // width preview

    auto update_bw = [=]() {
        int w = sp_bhi->value() - sp_blo->value() + 1;
        if (w > 0)
            lbl_bw->setText(QString("→ %1-bit value, 0–%2").arg(w).arg((1LL << std::min(w, 30)) - 1));
        else
            lbl_bw->setText("<font color='red'>hi must be ≥ lo</font>");
        lbl_bw->setTextFormat(Qt::RichText);
    };
    update_bw();
    QObject::connect(sp_blo, QOverload<int>::of(&QSpinBox::valueChanged), &d, [=](int){ update_bw(); });
    QObject::connect(sp_bhi, QOverload<int>::of(&QSpinBox::valueChanged), &d, [=](int){ update_bw(); });

    fl->addRow("Label name:", le_name);
    fl->addRow("Type:", cmb_type);
    fl->addRow("Byte offset:", sp_off);
    fl->addRow(lbl_cnt, sp_cnt);
    fl->addRow(lbl_end, cmb_end);
    fl->addRow(lbl_bit, sp_bit);
    fl->addRow(lbl_blo, sp_blo);
    fl->addRow(lbl_bhi, sp_bhi);
    fl->addRow(new QLabel(""), lbl_bw);

    auto set_vis = [](QWidget* a, QWidget* b, bool v){ a->setVisible(v); b->setVisible(v); };
    auto update_vis = [&](int idx) {
        bool is_raw   = (idx == int(LabelDef::RawValue));
        bool is_hw    = (idx == int(LabelDef::HammingWeight));
        bool is_bit   = (idx == int(LabelDef::Bit));
        bool is_range = (idx == int(LabelDef::BitRange));
        set_vis(lbl_cnt, sp_cnt,  is_raw || is_hw || is_range);
        set_vis(lbl_end, cmb_end, is_raw || is_range);
        set_vis(lbl_bit, sp_bit,  is_bit);
        set_vis(lbl_blo, sp_blo,  is_range);
        set_vis(lbl_bhi, sp_bhi,  is_range);
        lbl_bw->setVisible(is_range);
        d.adjustSize();
    };
    update_vis(cmb_type->currentIndex());
    QObject::connect(cmb_type, QOverload<int>::of(&QComboBox::currentIndexChanged), &d, update_vis);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QObject::connect(bb, &QDialogButtonBox::accepted, &d, [&]() {
        if (le_name->text().trimmed().isEmpty()) {
            QMessageBox::warning(&d, "Label", "Please enter a label name."); return;
        }
        if (cmb_type->currentIndex() == int(LabelDef::BitRange) &&
                sp_bhi->value() < sp_blo->value()) {
            QMessageBox::warning(&d, "Label", "High bit must be ≥ low bit."); return;
        }
        d.accept();
    });
    QObject::connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
    fl->addRow(bb);
    d.adjustSize();

    if (d.exec() != QDialog::Accepted) return false;

    def.name     = le_name->text().trimmed();
    def.type     = static_cast<LabelDef::Type>(cmb_type->currentIndex());
    def.byte_off = sp_off->value();
    def.byte_cnt = sp_cnt->value();
    def.big_end  = (cmb_end->currentIndex() == 1);
    def.bit_lo   = (def.type == LabelDef::Bit) ? sp_bit->value() : sp_blo->value();
    def.bit_hi   = sp_bhi->value();
    return true;
}

void MainWindow::onExportDataset() {
    if (!hasActiveDs() || activeDs().is_result) {
        QMessageBox::information(this, "Export Dataset", "No file loaded.");
        return;
    }
    const TrsHeader& h = activeDs().file->header();

    // ── Configuration dialog ──────────────────────────────────────────────
    QDialog cfg(this);
    cfg.setWindowTitle("Export Dataset — configuration");
    auto* vl_cfg = new QVBoxLayout(&cfg);

    // Range
    auto* grp_range  = new QGroupBox("Range");
    auto* fl_range   = new QFormLayout(grp_range);
    auto* sp_first   = new QSpinBox; sp_first->setRange(0, std::max(0, h.num_traces-1)); sp_first->setValue(0);
    auto* sp_count   = new QSpinBox; sp_count->setRange(1, h.num_traces);                sp_count->setValue(h.num_traces);
    auto* sp_s_first = new QSpinBox; sp_s_first->setRange(0, std::max(0,(int)h.num_samples-1)); sp_s_first->setValue(0);
    auto* sp_s_count = new QSpinBox; sp_s_count->setRange(0, (int)h.num_samples);        sp_s_count->setValue(0);
    sp_s_count->setSpecialValueText("All");
    fl_range->addRow("First trace:",          sp_first);
    fl_range->addRow("Trace count:",          sp_count);
    fl_range->addRow("First sample:",         sp_s_first);
    fl_range->addRow("Sample count (0=all):", sp_s_count);
    vl_cfg->addWidget(grp_range);

    // Label definitions
    auto* grp_lbl = new QGroupBox("Labels  (each becomes a named array in the NPZ)");
    auto* vl_lbl  = new QVBoxLayout(grp_lbl);
    auto* lw      = new QListWidget;
    lw->setToolTip("dtype is chosen automatically:\n"
                   "  1 byte  →  uint8\n  2 bytes →  uint16\n"
                   "  3-4 bytes → uint32\n  5-8 bytes → uint64\n"
                   "  HW / Bit  → uint8");

    QVector<LabelDef> labels;

    // Pre-populate from the file's param_map
    for (const auto& [pname, param] : h.param_map) {
        LabelDef d;
        d.name     = QString::fromStdString(pname);
        d.type     = LabelDef::RawValue;
        d.byte_off = param.offset;
        d.byte_cnt = std::min<int>(param.length, 8);
        labels.append(d);
        new QListWidgetItem(d.summary(), lw);
    }

    auto* hl_btns    = new QHBoxLayout;
    auto* btn_add    = new QPushButton("Add label…");
    auto* btn_edit   = new QPushButton("Edit…");
    auto* btn_remove = new QPushButton("Remove");
    btn_edit->setEnabled(false);
    btn_remove->setEnabled(false);
    hl_btns->addWidget(btn_add);
    hl_btns->addWidget(btn_edit);
    hl_btns->addWidget(btn_remove);
    hl_btns->addStretch();
    vl_lbl->addWidget(lw);
    vl_lbl->addLayout(hl_btns);
    vl_cfg->addWidget(grp_lbl);

    connect(lw, &QListWidget::currentRowChanged, &cfg, [&](int row) {
        btn_edit->setEnabled(row >= 0);
        btn_remove->setEnabled(row >= 0);
    });
    connect(btn_add, &QPushButton::clicked, &cfg, [&]() {
        LabelDef def;
        if (!runAddLabelDialog(&cfg, h.data_length, def)) return;
        labels.append(def);
        new QListWidgetItem(def.summary(), lw);
    });
    connect(btn_edit, &QPushButton::clicked, &cfg, [&]() {
        int row = lw->currentRow();
        if (row < 0 || row >= labels.size()) return;
        LabelDef def = labels[row];
        if (!runAddLabelDialog(&cfg, h.data_length, def)) return;
        labels[row] = def;
        lw->item(row)->setText(def.summary());
    });
    connect(btn_remove, &QPushButton::clicked, &cfg, [&]() {
        int row = lw->currentRow();
        if (row < 0 || row >= labels.size()) return;
        labels.removeAt(row);
        delete lw->takeItem(row);
    });

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, [&]() {
        if (labels.isEmpty()) {
            if (QMessageBox::question(&cfg, "No labels",
                    "No labels defined. Export traces only?",
                    QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
                return;
        }
        cfg.accept();
    });
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    vl_cfg->addWidget(cfg_bb);

    if (cfg.exec() != QDialog::Accepted) return;

    // ── Resolve parameters ────────────────────────────────────────────────
    const int32_t first   = sp_first->value();
    const int32_t count   = sp_count->value();
    const int64_t s_first = sp_s_first->value();
    const int64_t s_count = sp_s_count->value();
    const int64_t raw_ns  = (s_count == 0)
        ? (h.num_samples - s_first)
        : std::min<int64_t>(s_count, h.num_samples - s_first);

    int64_t eff_ns = raw_ns;
    for (const auto& t : activeDs().pipeline)
        eff_ns = t->transformedCount(eff_ns);

    // Memory estimate
    int64_t mem_est = static_cast<int64_t>(count) * eff_ns * sizeof(float);
    if (mem_est > 4LL * 1024 * 1024 * 1024) {
        if (QMessageBox::warning(this, "Memory warning",
                QString("Trace matrix will require ~%1 GB.\nContinue?")
                    .arg(double(mem_est)/(1024.0*1024*1024), 0,'f',1),
                QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
            return;
    }

    // ── Pick output path ──────────────────────────────────────────────────
    QString out_path = QFileDialog::getSaveFileName(
        this, "Export Dataset", recentDir("npy"), "NumPy archive (*.npz)");
    if (out_path.isEmpty()) return;
    updateRecentDir("npy", out_path);

    // ── Accumulate ────────────────────────────────────────────────────────
    std::vector<float> traces_flat(static_cast<size_t>(count) * static_cast<size_t>(eff_ns), 0.0f);
    std::vector<std::vector<uint64_t>> label_vals(
        labels.size(), std::vector<uint64_t>(static_cast<size_t>(count), 0));

    QProgressDialog prog("Exporting traces…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    std::vector<float> trace_buf(static_cast<size_t>(std::max(raw_ns, eff_ns)));
    int32_t written = 0;

    for (int32_t ti = 0; ti < count; ti++) {
        if (prog.wasCanceled()) break;
        prog.setValue(ti);
        QApplication::processEvents();

        int32_t src_idx = first + ti;

        // Labels
        if (!labels.isEmpty()) {
            auto data_bytes = activeDs().file->readData(src_idx);
            for (int li = 0; li < labels.size(); li++)
                label_vals[li][ti] = labels[li].compute(data_bytes);
        }

        // Samples
        std::fill(trace_buf.begin(), trace_buf.end(), 0.0f);
        if (s_first < h.num_samples && s_first + raw_ns > 0) {
            int64_t src_start = std::max<int64_t>(0, s_first);
            int64_t src_end   = std::min<int64_t>(h.num_samples, s_first + raw_ns);
            int64_t dst_off   = src_start - s_first;
            activeDs().file->readSamples(src_idx, src_start, src_end - src_start,
                                   trace_buf.data() + dst_off);
        }
        for (const auto& t : activeDs().pipeline) t->reset();
        int64_t n_out = raw_ns;
        for (const auto& t : activeDs().pipeline)
            n_out = t->apply(trace_buf.data(), n_out, 0);

        float* row = traces_flat.data() + static_cast<size_t>(ti) * static_cast<size_t>(eff_ns);
        std::copy(trace_buf.begin(), trace_buf.begin() + n_out, row);
        written++;
    }
    prog.setValue(count);
    if (written == 0) return;

    // ── Build NPZ ─────────────────────────────────────────────────────────
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;

    entries.push_back({"traces.npy",
        buildNpyBytes("<f4", written, static_cast<int32_t>(eff_ns),
                      traces_flat.data(),
                      static_cast<size_t>(written) * static_cast<size_t>(eff_ns) * sizeof(float))});

    for (int li = 0; li < labels.size(); li++) {
        const LabelDef& ld = labels[li];
        size_t es = ld.elem_size();
        std::vector<uint8_t> buf(static_cast<size_t>(written) * es);
        for (int32_t ti = 0; ti < written; ti++) {
            uint64_t v = label_vals[li][ti];
            std::memcpy(buf.data() + static_cast<size_t>(ti) * es, &v, es);
        }
        entries.push_back({ld.name.toStdString() + ".npy",
            buildNpy1DBytes(ld.dtype(), buf.data(), written, es)});
    }

    QString err;
    if (!saveNpz(out_path, entries, err))
        QMessageBox::critical(this, "Export failed", err);
    else
        QMessageBox::information(this, "Export complete",
            QString("Saved %1 traces × %2 samples, %3 label(s)\n→ %4")
                .arg(written).arg(eff_ns).arg(labels.size()).arg(out_path));
}
