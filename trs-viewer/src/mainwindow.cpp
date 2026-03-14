#include "mainwindow.h"
#include "heatmap_widget.h"
#include "ttest.h"
#include "xcorr.h"

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
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLineEdit>
#include <QLabel>
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
#include <QSpinBox>
#include <QSplitter>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Statistical helpers for t-test threshold calculation
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
    connect(btn_apply_, &QPushButton::clicked, this, &MainWindow::onApplyTraces);
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

    side_l->addWidget(grp_file);
    side_l->addWidget(grp_trace);
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
    btn_mode_pan_->setCheckable(true);
    btn_mode_measure_->setCheckable(true);
    btn_mode_box_zoom_->setCheckable(true);
    btn_mode_pan_->setChecked(true);
    btn_mode_pan_->setToolTip("Drag to pan, scroll wheel to zoom");
    btn_mode_measure_->setToolTip("Click two points to measure distance (P)");
    btn_mode_box_zoom_->setToolTip("Drag to select a region and zoom into it (Z)");

    mode_group_ = new QButtonGroup(this);
    mode_group_->addButton(btn_mode_pan_,      0);
    mode_group_->addButton(btn_mode_measure_,  1);
    mode_group_->addButton(btn_mode_box_zoom_, 2);
    mode_group_->setExclusive(true);

    connect(mode_group_, &QButtonGroup::idClicked, this, [this](int id) {
        InteractionMode m = id == 0 ? InteractionMode::Pan
                          : id == 1 ? InteractionMode::Measure
                                    : InteractionMode::BoxZoom;
        plot_widget_->setMode(m);
        if (id == 0 || id == 2) lbl_measure_->setText("–");
    });

    // Separator
    auto* sep1 = new QFrame; sep1->setFrameShape(QFrame::VLine);
    auto* sep2 = new QFrame; sep2->setFrameShape(QFrame::VLine);

    // Zoom buttons
    btn_zoom_in_  = new QPushButton("＋ Zoom In");
    btn_zoom_out_ = new QPushButton("－ Zoom Out");
    btn_reset_    = new QPushButton("⟳ Reset  [R]");
    btn_zoom_in_->setToolTip("Zoom in X (also: scroll wheel up)");
    btn_zoom_out_->setToolTip("Zoom out X (also: scroll wheel down)");

    auto* btn_yzoom_in  = new QPushButton("↑ Amp");
    auto* btn_yzoom_out = new QPushButton("↓ Amp");
    btn_yzoom_in ->setToolTip("Zoom in Y / taller traces (also: Ctrl/Shift+scroll up)");
    btn_yzoom_out->setToolTip("Zoom out Y / shorter traces (also: Ctrl/Shift+scroll down)");

    // plot_widget_ hasn't been constructed yet at this point, so use lambdas
    // that capture `this` — by the time the button is clicked, plot_widget_ is valid.
    connect(btn_zoom_in_,  &QPushButton::clicked, this, [this](){ plot_widget_->zoomIn(); });
    connect(btn_zoom_out_, &QPushButton::clicked, this, [this](){ plot_widget_->zoomOut(); });
    connect(btn_reset_,    &QPushButton::clicked, this, &MainWindow::onResetView);
    connect(btn_yzoom_in,  &QPushButton::clicked, this, [this](){ plot_widget_->zoomInY(); });
    connect(btn_yzoom_out, &QPushButton::clicked, this, [this](){ plot_widget_->zoomOutY(); });

    // Theme selector
    combo_theme_ = new QComboBox;
    combo_theme_->addItems({"Dark", "Light"});
    connect(combo_theme_, &QComboBox::currentIndexChanged,
            this, &MainWindow::onThemeChanged);

    toolbar_l->addWidget(btn_mode_pan_);
    toolbar_l->addWidget(btn_mode_measure_);
    toolbar_l->addWidget(btn_mode_box_zoom_);
    toolbar_l->addWidget(sep1);
    toolbar_l->addWidget(btn_zoom_in_);
    toolbar_l->addWidget(btn_zoom_out_);
    toolbar_l->addWidget(btn_reset_);
    toolbar_l->addWidget(btn_yzoom_in);
    toolbar_l->addWidget(btn_yzoom_out);
    toolbar_l->addWidget(sep2);
    toolbar_l->addWidget(new QLabel("Theme:"));
    toolbar_l->addWidget(combo_theme_);
    toolbar_l->addStretch();

    // Plot widget
    plot_widget_ = new PlotWidget;
    connect(plot_widget_, &PlotWidget::viewChanged,
            this, &MainWindow::onViewChanged);
    connect(plot_widget_, &PlotWidget::measurementUpdated,
            this, &MainWindow::onMeasurementUpdated);

    right_l->addWidget(toolbar);
    right_l->addWidget(plot_widget_, 1);

    splitter->addWidget(side);
    splitter->addWidget(right_pane);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);

    // Keyboard shortcuts
    auto* sc_reset = new QShortcut(QKeySequence("R"), this);
    connect(sc_reset, &QShortcut::activated, this, &MainWindow::onResetView);
    auto* sc_plus  = new QShortcut(QKeySequence("+"), this);
    connect(sc_plus,  &QShortcut::activated, plot_widget_, &PlotWidget::zoomIn);
    auto* sc_minus = new QShortcut(QKeySequence("-"), this);
    connect(sc_minus, &QShortcut::activated, plot_widget_, &PlotWidget::zoomOut);
    auto* sc_p = new QShortcut(QKeySequence("P"), this);
    connect(sc_p, &QShortcut::activated, this, [this]() {
        bool measure = (plot_widget_->mode() == InteractionMode::Pan);
        plot_widget_->setMode(measure ? InteractionMode::Measure
                                       : InteractionMode::Pan);
        btn_mode_pan_->setChecked(!measure);
        btn_mode_measure_->setChecked(measure);
        btn_mode_box_zoom_->setChecked(false);
        if (!measure) lbl_measure_->setText("–");
    });
    auto* sc_z = new QShortcut(QKeySequence("Z"), this);
    connect(sc_z, &QShortcut::activated, this, [this]() {
        bool box_zoom = (plot_widget_->mode() == InteractionMode::BoxZoom);
        plot_widget_->setMode(box_zoom ? InteractionMode::Pan
                                       : InteractionMode::BoxZoom);
        btn_mode_pan_->setChecked(box_zoom);
        btn_mode_measure_->setChecked(false);
        btn_mode_box_zoom_->setChecked(!box_zoom);
        if (box_zoom) lbl_measure_->setText("–");
    });
}

MainWindow::~MainWindow() = default;

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
}

void MainWindow::onOpenFile() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open TRS file", {}, "TRS files (*.trs);;All files (*)");
    if (!path.isEmpty()) openFile(path);
}

void MainWindow::openFile(const QString& path) {
    auto f = std::make_unique<TrsFile>();
    std::string err;
    if (!f->open(path.toStdString(), err)) {
        QMessageBox::critical(this, "Error opening TRS file",
                              QString::fromStdString(err));
        return;
    }
    trs_file_ = std::move(f);

    // Clear pipeline whenever a new file is opened.
    pipeline_.clear();
    rebuildTransformList();
    plot_widget_->setTransforms(pipeline_);

    int n = trs_file_->header().num_traces;
    spin_first_->setMaximum(std::max(0, n - 1));
    spin_count_->setMaximum(n);
    spin_count_->setValue(1);

    updateFileInfo();
    onApplyTraces();
}

void MainWindow::updateFileInfo() {
    if (!trs_file_) { lbl_file_->setText("No file"); lbl_info_->clear(); return; }

    const auto& h = trs_file_->header();
    lbl_file_->setText(QString::fromStdString(trs_file_->path()).section('/', -1));

    const char* type_str = "?";
    switch (h.sample_type) {
    case SampleType::INT8:    type_str = "int8";    break;
    case SampleType::INT16:   type_str = "int16";   break;
    case SampleType::INT32:   type_str = "int32";   break;
    case SampleType::FLOAT32: type_str = "float32"; break;
    }

    // Compute effective sample count after pipeline (decimating transforms shrink it).
    int64_t effective_samples = h.num_samples;
    for (const auto& t : pipeline_)
        effective_samples = t->transformedCount(effective_samples);

    QString info = QString("Traces:  %1\nSamples: %2\nType:    %3\nData:    %4 B/trace")
        .arg(h.num_traces).arg(h.num_samples)
        .arg(type_str).arg(h.data_length);
    if (effective_samples != h.num_samples)
        info += QString("\nAfter pipeline: %1").arg(effective_samples);
    lbl_info_->setText(info);
}

void MainWindow::onApplyTraces() {
    if (!trs_file_) return;

    plot_widget_->clearTraces();

    int first = spin_first_->value();
    int count = spin_count_->value();
    int max   = trs_file_->header().num_traces;

    for (int i = 0; i < count && (first + i) < max; i++) {
        QColor  col   = TRACE_COLORS[(first + i) % NUM_COLORS];
        QString label = QString("Trace %1").arg(first + i);
        plot_widget_->addTrace(trs_file_.get(), first + i, col, label);
    }
    plot_widget_->setTransforms(pipeline_);
    plot_widget_->resetView();
}

void MainWindow::onAddTransform() {
    auto tx = createTransform(combo_transform_->currentIndex());
    if (!tx) return;
    pipeline_.push_back(tx);
    rebuildTransformList();
    updateFileInfo();
    plot_widget_->setTransforms(pipeline_);
    plot_widget_->update();
}

void MainWindow::onRemoveTransform() {
    int row = list_transforms_->currentRow();
    if (row < 0 || row >= static_cast<int>(pipeline_.size())) return;
    pipeline_.erase(pipeline_.begin() + row);
    rebuildTransformList();
    updateFileInfo();
    plot_widget_->setTransforms(pipeline_);
    plot_widget_->update();
}

void MainWindow::onMoveTransformUp() {
    int row = list_transforms_->currentRow();
    if (row <= 0 || row >= static_cast<int>(pipeline_.size())) return;
    std::swap(pipeline_[row], pipeline_[row - 1]);
    rebuildTransformList();
    list_transforms_->setCurrentRow(row - 1);
    updateFileInfo();
    plot_widget_->setTransforms(pipeline_);
    plot_widget_->update();
}

void MainWindow::onMoveTransformDown() {
    int row = list_transforms_->currentRow();
    if (row < 0 || row + 1 >= static_cast<int>(pipeline_.size())) return;
    std::swap(pipeline_[row], pipeline_[row + 1]);
    rebuildTransformList();
    list_transforms_->setCurrentRow(row + 1);
    updateFileInfo();
    plot_widget_->setTransforms(pipeline_);
    plot_widget_->update();
}

void MainWindow::onResetView() {
    plot_widget_->resetView();
    plot_widget_->clearCropRanges();
}

void MainWindow::onViewChanged(int64_t start, int64_t end, int64_t /*total*/) {
    int64_t raw_span = end - start;
    int64_t eff_span = raw_span;
    for (const auto& t : pipeline_)
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
    plot_widget_->setTheme(index == 0 ? PlotTheme::dark() : PlotTheme::light());
}

void MainWindow::rebuildTransformList() {
    list_transforms_->clear();
    for (int i = 0; i < static_cast<int>(pipeline_.size()); i++) {
        list_transforms_->addItem(
            QString("%1. %2").arg(i + 1)
                             .arg(QString::fromStdString(pipeline_[i]->name())));
    }
}

// ---------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------

// Write a TRS file (always FLOAT32 output) with the given pipeline applied.
// Streams in 256 K-sample chunks to keep RAM usage low.
// Returns false and sets err_out on failure.
static bool exportTracesToTrs(
    const QString& out_path,
    TrsFile* src,
    int32_t first_trace, int32_t count,
    const std::vector<std::shared_ptr<ITransform>>& pipeline,
    QProgressDialog* progress,
    QString& err_out)
{
    const TrsHeader& h = src->header();
    int32_t n_traces  = std::min(count, h.num_traces - first_trace);
    if (n_traces <= 0) { err_out = "No traces to export."; return false; }
    int32_t n_samples = h.num_samples;

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

    // TRS header TLVs -------------------------------------------------------
    write_tlv_hdr(0x41, 4); write_le32(n_traces);                         // NUMBER_TRACES
    write_tlv_hdr(0x42, 4); write_le32(static_cast<int32_t>(out_samples)); // NUMBER_SAMPLES
    write_tlv_hdr(0x43, 1); std::fputc(0x14, fp);                         // SAMPLE_CODING: float32
    if (h.data_length > 0) {
        write_tlv_hdr(0x44, 2); write_le16(h.data_length); // DATA_LENGTH
    }
    std::fputc(0x5F, fp); std::fputc(0x00, fp);      // TRACE_BLOCK

    // Trace data -------------------------------------------------------------
    constexpr int64_t CHUNK = 256 * 1024;
    std::vector<float> buf(CHUNK);

    for (int32_t ti = 0; ti < n_traces; ti++) {
        if (progress) {
            if (progress->wasCanceled()) {
                std::fclose(fp);
                QFile::remove(out_path);
                err_out = "Export cancelled.";
                return false;
            }
            progress->setValue(ti);
            QApplication::processEvents();
        }

        int32_t src_idx = first_trace + ti;

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
            int64_t read  = src->readSamples(src_idx, written, chunk, buf.data());
            if (read <= 0) break;
            int64_t out_count = read;
            for (auto& t : pipeline) out_count = t->apply(buf.data(), out_count, written);
            std::fwrite(buf.data(), sizeof(float), static_cast<size_t>(out_count), fp);
            written += read;
        }
    }

    std::fclose(fp);
    return true;
}

void MainWindow::onExportTrs() {
    if (!trs_file_) {
        QMessageBox::information(this, "Export TRS", "No file loaded.");
        return;
    }

    // Ask which traces to export (default: all).
    int n = trs_file_->header().num_traces;

    QDialog range_dlg(this);
    range_dlg.setWindowTitle("Export TRS — select range");
    auto* fl       = new QFormLayout(&range_dlg);
    auto* sp_first = new QSpinBox;
    auto* sp_count = new QSpinBox;
    sp_first->setRange(0, std::max(0, n - 1));
    sp_first->setValue(0);
    sp_count->setRange(1, n);
    sp_count->setValue(n);
    auto* bb = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);
    fl->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &range_dlg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &range_dlg, &QDialog::reject);
    if (range_dlg.exec() != QDialog::Accepted) return;

    int32_t first = static_cast<int32_t>(sp_first->value());
    int32_t count = static_cast<int32_t>(sp_count->value());

    QString path = QFileDialog::getSaveFileName(
        this, "Export processed TRS", {}, "TRS files (*.trs)");
    if (path.isEmpty()) return;

    QProgressDialog progress("Exporting traces…", "Cancel",
                             0, count, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(500);

    QString err;
    bool ok = exportTracesToTrs(path, trs_file_.get(), first, count,
                                pipeline_, &progress, err);
    progress.setValue(count);

    if (!ok)
        QMessageBox::critical(this, "Export failed", err);
    else
        QMessageBox::information(this, "Export complete",
            QString("Saved %1 trace(s) to:\n%2").arg(count).arg(path));
}

void MainWindow::onExportPng() {
    if (!trs_file_) {
        QMessageBox::information(this, "Export PNG", "No file loaded.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Export PNG", {}, "PNG images (*.png)");
    if (path.isEmpty()) return;

    QPixmap px = plot_widget_->grab();
    QImage img = px.toImage();
    constexpr int kMaxPx = 2400;
    if (img.width() > kMaxPx || img.height() > kMaxPx)
        img = img.scaled(kMaxPx, kMaxPx, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    if (!img.save(path, "PNG"))
        QMessageBox::critical(this, "Export failed",
                              "Could not write PNG to:\n" + path);
}

void MainWindow::onExportPdf() {
    if (!trs_file_) {
        QMessageBox::information(this, "Export PDF", "No file loaded.");
        return;
    }

    QString path = QFileDialog::getSaveFileName(
        this, "Export PDF", {}, "PDF files (*.pdf)");
    if (path.isEmpty()) return;

    // Grab the plot at current widget resolution, then paint into PDF.
    QPixmap px = plot_widget_->grab();

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
// Only STORE (method 0) compression is supported.
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

        if (method != 0) {
            err = QString("Entry '%1' uses compression method %2; only STORE (0) is supported.")
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
        if (std::fread(data.data(), 1, uncomp_sz, fp) != uncomp_sz) {
            err = QString("Truncated data for entry '%1'.").arg(QString::fromStdString(fname));
            std::fclose(fp); return false;
        }
        (void)comp_sz;
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

    // dtype must be float32 or uint8
    bool is_f32 = (hdr.find("'<f4'") != std::string::npos || hdr.find("\"<f4\"") != std::string::npos);
    bool is_u8  = (hdr.find("'|u1'") != std::string::npos || hdr.find("\"u1\""  ) != std::string::npos ||
                   hdr.find("'uint8'") != std::string::npos);
    if (!is_f32 && !is_u8) {
        err = "Only float32 ('<f4') and uint8 ('|u1') dtypes are supported."; return false;
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
    if (is_f32) {
        if (data_offset + static_cast<size_t>(n_elements) * 4 > buf.size()) {
            err = "NPY data truncated."; return false;
        }
        data.resize(static_cast<size_t>(n_elements));
        std::memcpy(data.data(), buf.data() + data_offset,
                    static_cast<size_t>(n_elements) * sizeof(float));
    } else {
        // uint8 → cast to float
        if (data_offset + static_cast<size_t>(n_elements) > buf.size()) {
            err = "NPY data truncated."; return false;
        }
        data.resize(static_cast<size_t>(n_elements));
        for (int64_t i = 0; i < n_elements; i++)
            data[static_cast<size_t>(i)] = static_cast<float>(buf[data_offset + static_cast<size_t>(i)]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Load NPY helpers
// ---------------------------------------------------------------------------

void MainWindow::onLoadNpyTTest() {
    QString path = QFileDialog::getOpenFileName(
        this, "Load t-test NPY", {}, "NumPy files (*.npy);;All files (*)");
    if (path.isEmpty()) return;

    std::vector<float> data;
    std::vector<int64_t> shape;
    QString err;
    if (!loadNpy(path, data, shape, err)) {
        QMessageBox::critical(this, "Load failed", err); return;
    }
    if (shape.size() != 1) {
        QMessageBox::critical(this, "Load failed",
            QString("Expected a 1-D array, got %1-D.").arg(shape.size())); return;
    }

    auto tstat_ptr = std::make_shared<std::vector<float>>(std::move(data));

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(QString("T-test — %1 samples — %2")
                            .arg(shape[0])
                            .arg(QFileInfo(path).fileName()));
    dlg->resize(1100, 520);

    auto* pw = new PlotWidget(dlg);
    pw->addTrace(tstat_ptr, QColor("#4fc3f7"), "t-statistic");
    pw->setThresholds(true, 4.5, -4.5);
    pw->resetView();

    auto* lbl_thr  = new QLabel("Threshold ±:");
    auto* spin_thr = new QDoubleSpinBox;
    spin_thr->setRange(0.1, 1000.0); spin_thr->setValue(4.5);
    spin_thr->setDecimals(2); spin_thr->setSingleStep(0.5);
    connect(spin_thr, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [pw](double v) { pw->setThresholds(true, v, -v); });

    auto* chk_onesided_npy = new QCheckBox("One-sided (+)");
    chk_onesided_npy->setToolTip("Show only positive threshold (use after abs() preprocessing)");
    connect(chk_onesided_npy, &QCheckBox::toggled, dlg, [pw, lbl_thr](bool on) {
        pw->setThresholdOneSided(on);
        lbl_thr->setText(on ? "Threshold +:" : "Threshold ±:");
    });

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, dlg, [dlg, tstat_ptr]() {
        QString p = QFileDialog::getSaveFileName(dlg, "Export t-test as NumPy",
                                                 {}, "NumPy files (*.npy)");
        if (p.isEmpty()) return;
        QString e;
        if (!saveNpy(p, tstat_ptr->data(), static_cast<int64_t>(tstat_ptr->size()), e))
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
        sp_alpha->setValue(0.05);      sp_alpha->setSingleStep(0.01);

        int64_t n_L = static_cast<int64_t>(tstat_ptr->size());
        auto* lbl_nL = new QLabel(QString::number(n_L));

        // No accumulator available — let user enter group sizes
        auto* sp_nA = new QSpinBox; sp_nA->setRange(2, 10000000); sp_nA->setValue(100);
        auto* sp_nB = new QSpinBox; sp_nB->setRange(2, 10000000); sp_nB->setValue(100);

        auto* lbl_ath = new QLabel;
        auto* lbl_nu  = new QLabel;
        auto* lbl_th  = new QLabel;
        lbl_th->setTextFormat(Qt::RichText);

        // Equal-variance Welch df from n_A, n_B
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
        recalc();

        auto* bb = new QDialogButtonBox(QDialogButtonBox::Apply | QDialogButtonBox::Close);
        connect(bb->button(QDialogButtonBox::Apply), &QPushButton::clicked, cd, [=]() {
            double a    = sp_alpha->value();
            double a_th = 1.0 - std::pow(1.0 - a, 1.0 / static_cast<double>(n_L));
            double nu   = calc_nu(sp_nA->value(), sp_nB->value());
            spin_thr->setValue(invTCdf(1.0 - a_th / 2.0, nu));
        });
        connect(bb, &QDialogButtonBox::rejected, cd, &QDialog::close);

        fl->addRow("Significance level α:", sp_alpha);
        fl->addRow("Trace length n_L:",      lbl_nL);
        fl->addRow("Group A  n_A:",           sp_nA);
        fl->addRow("Group B  n_B:",           sp_nB);
        fl->addRow(new QLabel);
        fl->addRow("Bonferroni α_TH:",        lbl_ath);
        fl->addRow("Approx. Welch ν̂:",        lbl_nu);
        fl->addRow("Threshold TH:",           lbl_th);
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
        QString path = QFileDialog::getSaveFileName(dlg, "Export t-test as PDF", {}, "PDF files (*.pdf)");
        if (path.isEmpty()) return;
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

    auto* ctrl   = new QWidget(dlg);
    auto* ctrl_l = new QHBoxLayout(ctrl);
    ctrl_l->setContentsMargins(4, 2, 4, 2);
    ctrl_l->addWidget(new QLabel(QString("Samples: <b>%1</b>").arg(shape[0])));
    ctrl_l->setSpacing(6);
    auto* lbl_f = qobject_cast<QLabel*>(ctrl_l->itemAt(0)->widget());
    if (lbl_f) lbl_f->setTextFormat(Qt::RichText);
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

    auto* vl = new QVBoxLayout(dlg);
    vl->setContentsMargins(4, 4, 4, 4); vl->setSpacing(4);
    vl->addWidget(ctrl);
    vl->addWidget(pw, 1);
    dlg->show();
}

void MainWindow::onLoadNpyHeatmap() {
    QString path = QFileDialog::getOpenFileName(
        this, "Load heatmap NPY", {}, "NumPy files (*.npy);;All files (*)");
    if (path.isEmpty()) return;

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
        QString p = QFileDialog::getSaveFileName(dlg, "Export as PNG", {}, "PNG images (*.png)");
        if (p.isEmpty()) return;
        if (!heatmap->exportPng(p))
            QMessageBox::critical(dlg, "Export failed", "Could not write:\n" + p);
        else
            QMessageBox::information(dlg, "Saved", "Saved: " + p);
    });

    auto* btn_exp_npy = new QPushButton("Export .npy…");
    connect(btn_exp_npy, &QPushButton::clicked, dlg, [=]() {
        QString p = QFileDialog::getSaveFileName(dlg, "Export as NumPy", {}, "NumPy files (*.npy)");
        if (p.isEmpty()) return;
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
        {}, "NumPy files (*.npy *.npz);;All files (*)");
    if (path.isEmpty()) return;

    std::vector<float> traces_flat;
    std::vector<float> data_flat;
    int64_t n_traces = 0, n_samples = 0;
    int64_t data_cols = 0;
    QString err;

    if (path.endsWith(".npz", Qt::CaseInsensitive)) {
        std::map<std::string, std::vector<uint8_t>> entries;
        if (!loadNpz(path, entries, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }
        // Look for 'traces', 'traces.npy', or the first 2-D float32 entry
        auto it = entries.find("traces.npy");
        if (it == entries.end()) it = entries.find("traces");
        if (it == entries.end()) {
            // pick the first 2-D entry
            for (auto& kv : entries) {
                std::vector<int64_t> sh;
                std::vector<float> tmp;
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
                QString("Traces entry is %1-D; expected 2-D (n_traces × n_samples).").arg(sh.size()));
            return;
        }
        n_traces = sh[0]; n_samples = sh[1];

        // Optional 'data' or 'data.npy' entry (n_traces × data_length)
        for (const char* dname : {"data.npy", "data", "labels.npy", "labels"}) {
            auto di = entries.find(dname);
            if (di == entries.end()) continue;
            std::vector<int64_t> dsh;
            if (parseNpyBytes(di->second, data_flat, dsh, err) && dsh.size() == 2 && dsh[0] == n_traces) {
                data_cols = dsh[1];
                break;
            }
        }
    } else {
        // Plain .npy
        std::vector<int64_t> shape;
        if (!loadNpy(path, traces_flat, shape, err)) {
            QMessageBox::critical(this, "Load failed", err); return;
        }
        if (shape.size() != 2) {
            QMessageBox::critical(this, "Load failed",
                QString("Expected a 2-D array (n_traces × n_samples), got %1-D.").arg(shape.size()));
            return;
        }
        n_traces = shape[0]; n_samples = shape[1];
    }

    if (n_traces < 1 || n_samples < 1) {
        QMessageBox::critical(this, "Load failed", "Empty array."); return;
    }

    // Build the in-memory TrsFile
    auto f = std::make_unique<TrsFile>();
    std::vector<uint8_t> data_bytes;
    int16_t data_length = 0;
    if (data_cols > 0) {
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

    trs_file_ = std::move(f);
    pipeline_.clear();
    rebuildTransformList();
    plot_widget_->setTransforms(pipeline_);

    int n = trs_file_->header().num_traces;
    spin_first_->setMaximum(std::max(0, n - 1));
    spin_count_->setMaximum(n);
    spin_count_->setValue(1);
    setWindowTitle(QString("TRS Viewer — %1").arg(QFileInfo(path).fileName()));

    updateFileInfo();
    onApplyTraces();
}

// ---------------------------------------------------------------------------
// Export traces → 2-D NPY (n_traces × n_samples, pipeline applied)
// ---------------------------------------------------------------------------
void MainWindow::onExportNpy() {
    if (!trs_file_) {
        QMessageBox::information(this, "Export NPY", "No file loaded."); return;
    }
    const TrsHeader& h = trs_file_->header();

    // Config dialog: same range picker as TRS export
    QDialog cfg(this);
    cfg.setWindowTitle("Export as NPY — configuration");
    auto* fl     = new QFormLayout(&cfg);
    auto* sp_first = new QSpinBox; sp_first->setRange(0, std::max(0, h.num_traces-1)); sp_first->setValue(0);
    auto* sp_count = new QSpinBox; sp_count->setRange(1, h.num_traces); sp_count->setValue(h.num_traces);
    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);
    fl->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first = static_cast<int32_t>(sp_first->value());
    int32_t count = static_cast<int32_t>(sp_count->value());
    count = std::min(count, h.num_traces - first);

    int64_t out_samples = h.num_samples;
    for (const auto& t : pipeline_) out_samples = t->transformedCount(out_samples);

    QString path = QFileDialog::getSaveFileName(
        this, "Export traces as NPY", {}, "NumPy files (*.npy)");
    if (path.isEmpty()) return;

    // Allocate full matrix (may be large)
    const size_t total = static_cast<size_t>(count) * static_cast<size_t>(out_samples);
    double mem_mb = total * sizeof(float) / (1024.0 * 1024.0);
    if (mem_mb > 2048.0) {
        if (QMessageBox::warning(this, "Memory warning",
                QString("Output matrix will require ~%1 GB.\nContinue?").arg(mem_mb/1024.0, 0,'f',1),
                QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes) return;
    }

    std::vector<float> matrix(total);
    std::vector<float> buf(static_cast<size_t>(h.num_samples));

    QProgressDialog prog("Exporting…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    for (int32_t ti = 0; ti < count; ti++) {
        if (prog.wasCanceled()) return;
        prog.setValue(ti);
        QApplication::processEvents();

        int32_t src = first + ti;
        int64_t got = trs_file_->readSamples(src, 0, h.num_samples, buf.data());
        if (got < h.num_samples)
            std::fill(buf.begin() + static_cast<size_t>(got), buf.end(), 0.0f);
        for (const auto& t : pipeline_) t->reset();
        int64_t n_out = got;
        for (const auto& t : pipeline_) n_out = t->apply(buf.data(), n_out, 0);
        std::copy(buf.begin(), buf.begin() + static_cast<size_t>(n_out),
                  matrix.begin() + static_cast<ptrdiff_t>(ti) * static_cast<ptrdiff_t>(out_samples));
    }
    prog.setValue(count);

    QString err;
    if (!saveNpy2D(path, matrix.data(), count, out_samples, err))
        QMessageBox::critical(this, "Export failed", err);
    else
        QMessageBox::information(this, "Exported",
            QString("Saved %1 × %2 matrix to:\n%3").arg(count).arg(out_samples).arg(path));
}

// ---------------------------------------------------------------------------
// Export traces → NPZ (traces.npy + data.npy if TRS has data bytes)
// ---------------------------------------------------------------------------
void MainWindow::onExportNpz() {
    if (!trs_file_) {
        QMessageBox::information(this, "Export NPZ", "No file loaded."); return;
    }
    const TrsHeader& h = trs_file_->header();

    QDialog cfg(this);
    cfg.setWindowTitle("Export as NPZ — configuration");
    auto* fl       = new QFormLayout(&cfg);
    auto* sp_first = new QSpinBox; sp_first->setRange(0, std::max(0, h.num_traces-1)); sp_first->setValue(0);
    auto* sp_count = new QSpinBox; sp_count->setRange(1, h.num_traces); sp_count->setValue(h.num_traces);
    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);
    if (h.data_length > 0)
        fl->addRow(new QLabel(QString("<i>Will also export data.npy (%1 bytes/trace)</i>")
                                  .arg(h.data_length)));
    fl->addRow(bb);
    connect(bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first = static_cast<int32_t>(sp_first->value());
    int32_t count = static_cast<int32_t>(sp_count->value());
    count = std::min(count, h.num_traces - first);

    int64_t out_samples = h.num_samples;
    for (const auto& t : pipeline_) out_samples = t->transformedCount(out_samples);

    QString path = QFileDialog::getSaveFileName(
        this, "Export traces as NPZ", {}, "NumPy archives (*.npz)");
    if (path.isEmpty()) return;

    // Build trace matrix
    std::vector<float> traces(static_cast<size_t>(count) * static_cast<size_t>(out_samples));
    std::vector<float> buf(static_cast<size_t>(h.num_samples));

    QProgressDialog prog("Exporting…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    for (int32_t ti = 0; ti < count; ti++) {
        if (prog.wasCanceled()) return;
        prog.setValue(ti);
        QApplication::processEvents();
        int32_t src = first + ti;
        int64_t got = trs_file_->readSamples(src, 0, h.num_samples, buf.data());
        if (got < h.num_samples)
            std::fill(buf.begin() + static_cast<size_t>(got), buf.end(), 0.0f);
        for (const auto& t : pipeline_) t->reset();
        int64_t n_out = got;
        for (const auto& t : pipeline_) n_out = t->apply(buf.data(), n_out, 0);
        std::copy(buf.begin(), buf.begin() + static_cast<size_t>(n_out),
                  traces.begin() + static_cast<ptrdiff_t>(ti) * static_cast<ptrdiff_t>(out_samples));
    }
    prog.setValue(count);

    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries;
    entries.push_back({"traces.npy",
        buildNpyBytes("<f4", count, out_samples, traces.data(),
                      static_cast<size_t>(count) * static_cast<size_t>(out_samples) * sizeof(float))});

    if (h.data_length > 0) {
        // Build data matrix (n_traces × data_length, uint8)
        std::vector<uint8_t> data_mat(static_cast<size_t>(count) * static_cast<size_t>(h.data_length));
        for (int32_t ti = 0; ti < count; ti++) {
            auto db = trs_file_->readData(first + ti);
            size_t copy_n = std::min(static_cast<size_t>(h.data_length), db.size());
            std::copy(db.begin(), db.begin() + static_cast<ptrdiff_t>(copy_n),
                      data_mat.begin() + static_cast<ptrdiff_t>(ti) * h.data_length);
        }
        entries.push_back({"data.npy",
            buildNpyBytes("|u1", count, h.data_length, data_mat.data(), data_mat.size())});
    }

    QString err;
    if (!saveNpz(path, entries, err))
        QMessageBox::critical(this, "Export failed", err);
    else
        QMessageBox::information(this, "Exported",
            QString("Saved %1 × %2 traces%3 to:\n%4")
                .arg(count).arg(out_samples)
                .arg(h.data_length > 0 ? " + data" : "")
                .arg(path));
}

void MainWindow::onRunTTest() {
    if (!trs_file_) {
        QMessageBox::information(this, "T-test", "No file loaded.");
        return;
    }
    const TrsHeader& h = trs_file_->header();
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
    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    fl->addRow("First trace:", sp_first);
    fl->addRow("Count:",       sp_count);

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

    fl->addRow(cfg_bb);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);
    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first    = static_cast<int32_t>(sp_first->value());
    int32_t count    = static_cast<int32_t>(sp_count->value());
    int32_t byte_idx = have_ttest_param ? auto_byte_idx
                                        : static_cast<int32_t>(sp_byte->value());

    // Effective sample count after pipeline
    int64_t effective_samples = h.num_samples;
    for (const auto& t : pipeline_)
        effective_samples = t->transformedCount(effective_samples);

    // Memory estimate warning
    int64_t mem_bytes = effective_samples * 4LL * static_cast<int64_t>(sizeof(double));
    if (mem_bytes > 2LL * 1024 * 1024 * 1024) {
        if (QMessageBox::warning(this, "Memory warning",
                QString("Accumulators will require ~%1 GB.\nContinue?")
                    .arg(double(mem_bytes) / (1024.0*1024*1024), 0, 'f', 1),
                QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
            return;
    }

    // --- Accumulation ---
    auto acc_ptr = std::make_shared<TTestAccumulator>(static_cast<int32_t>(effective_samples));
    TTestAccumulator& acc = *acc_ptr;

    QProgressDialog prog("Accumulating traces…", "Cancel", 0, count, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(400);

    // Full-trace buffer for pipeline application
    std::vector<float> trace_buf(static_cast<size_t>(h.num_samples));
    int32_t skipped = 0;

    for (int32_t ti = 0; ti < count; ti++) {
        if (prog.wasCanceled()) return;
        prog.setLabelText(QString("Accumulating trace %1 / %2…").arg(ti + 1).arg(count));
        prog.setValue(ti);
        QApplication::processEvents();

        int32_t src_idx = first + ti;
        auto data_bytes = trs_file_->readData(src_idx);
        if (byte_idx >= static_cast<int32_t>(data_bytes.size())) { skipped++; continue; }
        int group = (data_bytes[byte_idx] != 0) ? 1 : 0;

        // Read full trace, apply pipeline, accumulate once
        int64_t got = trs_file_->readSamples(src_idx, 0, h.num_samples, trace_buf.data());
        if (got <= 0) { skipped++; continue; }
        if (got < h.num_samples)
            std::fill(trace_buf.begin() + static_cast<size_t>(got), trace_buf.end(), 0.0f);
        for (const auto& t : pipeline_) t->reset();
        int64_t n_out = got;
        for (const auto& t : pipeline_)
            n_out = t->apply(trace_buf.data(), n_out, 0);
        acc.addTrace(group, trace_buf.data(), static_cast<int32_t>(n_out));
    }
    prog.setValue(count);

    if (skipped > 0)
        QMessageBox::warning(this, "T-test",
            QString("%1 traces skipped (data byte out of range).").arg(skipped));

    // --- Compute ---
    std::vector<float> tstat;
    std::string err;
    if (!acc.compute(tstat, err)) {
        QMessageBox::critical(this, "T-test failed", QString::fromStdString(err));
        return;
    }

    int64_t n0 = acc.countGroup(0), n1 = acc.countGroup(1);

    // --- Result window ---
    auto tstat_ptr = std::make_shared<std::vector<float>>(std::move(tstat));

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle(QString("Welch t-test — %1 traces (G0:%2  G1:%3)")
                            .arg(count).arg(n0).arg(n1));
    dlg->resize(1100, 520);

    auto* pw = new PlotWidget(dlg);
    pw->addTrace(tstat_ptr, QColor("#4fc3f7"), "t-statistic");
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
    connect(chk_onesided, &QCheckBox::toggled, dlg, [pw, lbl_thr](bool on) {
        pw->setThresholdOneSided(on);
        lbl_thr->setText(on ? "Threshold +:" : "Threshold ±:");
    });

    auto* btn_exp_trs = new QPushButton("Export TRS…");
    auto* btn_exp_npy = new QPushButton("Export .npy…");

    connect(spin_thr, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [pw](double v) { pw->setThresholds(true, v, -v); });

    connect(btn_exp_trs, &QPushButton::clicked, dlg, [dlg, tstat_ptr]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export t-test as TRS",
                                                    {}, "TRS files (*.trs)");
        if (path.isEmpty()) return;
        // Write single-trace float32 TRS
        FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
        if (!fp) { QMessageBox::critical(dlg, "Export failed", "Cannot create file."); return; }
        int32_t ns = static_cast<int32_t>(tstat_ptr->size());
        // Header TLVs
        auto wle32 = [&](int32_t v) {
            uint8_t b[4] = {uint8_t(v),uint8_t(v>>8),uint8_t(v>>16),uint8_t(v>>24)};
            std::fwrite(b, 1, 4, fp);
        };
        auto wtlv = [&](uint8_t tag, uint8_t len) { std::fputc(tag,fp); std::fputc(len,fp); };
        wtlv(0x41, 4); wle32(1);     // NUMBER_TRACES = 1
        wtlv(0x42, 4); wle32(ns);    // NUMBER_SAMPLES
        wtlv(0x43, 1); std::fputc(0x14, fp);  // SAMPLE_CODING: float32
        std::fputc(0x5F, fp); std::fputc(0x00, fp); // TRACE_BLOCK
        std::fwrite(tstat_ptr->data(), sizeof(float), static_cast<size_t>(ns), fp);
        std::fclose(fp);
        QMessageBox::information(dlg, "Export complete", "Saved: " + path);
    });

    connect(btn_exp_npy, &QPushButton::clicked, dlg, [dlg, tstat_ptr]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export t-test as NumPy",
                                                    {}, "NumPy files (*.npy)");
        if (path.isEmpty()) return;
        QString err;
        if (!saveNpy(path, tstat_ptr->data(), static_cast<int64_t>(tstat_ptr->size()), err))
            QMessageBox::critical(dlg, "Export failed", err);
        else
            QMessageBox::information(dlg, "Export complete", "Saved: " + path);
    });

    auto* btn_calc_th = new QPushButton("Calc TH…");
    connect(btn_calc_th, &QPushButton::clicked, dlg, [=]() {
        auto* cd = new QDialog(dlg);
        cd->setWindowTitle("Threshold Calculator");
        cd->setWindowModality(Qt::WindowModal);
        auto* fl = new QFormLayout(cd);

        auto* sp_alpha = new QDoubleSpinBox;
        sp_alpha->setRange(1e-6, 0.5); sp_alpha->setDecimals(6);
        sp_alpha->setValue(0.05);      sp_alpha->setSingleStep(0.01);

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
        fl->addRow("Bonferroni α_TH:",        lbl_ath);
        fl->addRow("Median Welch ν̂:",         lbl_nu);
        fl->addRow("Threshold TH:",           lbl_th);
        fl->addRow(bb);
        cd->show();
    });

    // Style dialog button
    auto* btn_style = new QPushButton("Style…");
    connect(btn_style, &QPushButton::clicked, dlg, [=]() {
        auto* sd = new QDialog(dlg);
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

    // PDF export button
    auto* btn_exp_pdf = new QPushButton("Export PDF…");
    connect(btn_exp_pdf, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export t-test as PDF", {}, "PDF files (*.pdf)");
        if (path.isEmpty()) return;
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

    auto* ctrl = new QWidget(dlg);
    auto* ctrl_l = new QHBoxLayout(ctrl);
    ctrl_l->setContentsMargins(4, 2, 4, 2);
    ctrl_l->addWidget(lbl_groups);
    ctrl_l->addStretch();
    auto* btn_yzi = new QPushButton("↑ Amp");
    auto* btn_yzo = new QPushButton("↓ Amp");
    btn_yzi->setToolTip("Zoom in Y (Ctrl/Shift+scroll up)");
    btn_yzo->setToolTip("Zoom out Y / shorter traces (Ctrl/Shift+scroll down)");
    connect(btn_yzi, &QPushButton::clicked, dlg, [pw](){ pw->zoomInY(); });
    connect(btn_yzo, &QPushButton::clicked, dlg, [pw](){ pw->zoomOutY(); });

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

    auto* vl = new QVBoxLayout(dlg);
    vl->setContentsMargins(4, 4, 4, 4);
    vl->setSpacing(4);
    vl->addWidget(ctrl);
    vl->addWidget(pw, 1);

    dlg->show();
}

// ---------------------------------------------------------------------------
// Crop & Merge
// ---------------------------------------------------------------------------

void MainWindow::onCropEdit() {
    if (!trs_file_) {
        QMessageBox::information(this, "Crop & Merge", "No file loaded.");
        return;
    }

    // Restore Pan mode when dialog is closed
    InteractionMode prev_mode = plot_widget_->mode();

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setWindowTitle("Crop & Merge — Range Editor");
    dlg->resize(500, 380);

    // ---- widgets ----
    auto* lbl_hint = new QLabel(
        "Drag on the plot (in drag-select mode) or click "
        "<b>Add current view</b> to add sample ranges.\n"
        "The selected ranges are concatenated per trace on export.");
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

    // ---- layout ----
    auto* vl = new QVBoxLayout(dlg);
    vl->addWidget(lbl_hint);
    vl->addWidget(btn_drag);
    vl->addWidget(list, 1);
    vl->addWidget(lbl_total);

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
        const auto& ranges = plot_widget_->cropRanges();
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
    connect(plot_widget_, &PlotWidget::cropRangesChanged, dlg, rebuildList);

    connect(btn_drag, &QPushButton::toggled, dlg, [this](bool on) {
        plot_widget_->setMode(on ? InteractionMode::CropSelect : InteractionMode::Pan);
    });

    connect(btn_add_view, &QPushButton::clicked, dlg, [this]() {
        plot_widget_->addCropRange(plot_widget_->viewStart(), plot_widget_->viewEnd());
    });

    connect(btn_remove, &QPushButton::clicked, dlg, [=]() {
        int row = list->currentRow();
        if (row >= 0) plot_widget_->removeCropRangeAt(row);
    });

    connect(btn_clear, &QPushButton::clicked, dlg, [this]() {
        plot_widget_->clearCropRanges();
    });

    connect(btn_close, &QPushButton::clicked, dlg, &QDialog::close);

    connect(dlg, &QDialog::finished, this, [this, prev_mode]() {
        // Restore previous interaction mode and clear any pending rubber-band
        plot_widget_->setMode(prev_mode);
    });

    // ---- export ----
    connect(btn_export, &QPushButton::clicked, dlg, [this, dlg]() {
        const auto& ranges = plot_widget_->cropRanges();
        if (ranges.empty()) return;

        // Compute total samples per output trace
        int64_t total_samples = 0;
        for (const auto& r : ranges) total_samples += r.second - r.first;
        if (total_samples <= 0) return;

        const TrsHeader& h = trs_file_->header();

        QString path = QFileDialog::getSaveFileName(
            dlg, "Export cropped TRS", {}, "TRS files (*.trs)");
        if (path.isEmpty()) return;

        int n_traces = h.num_traces;
        QProgressDialog prog("Exporting traces…", "Cancel", 0, n_traces, dlg);
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
        tlv(0x41, 4); fputle32(n_traces);                       // NUMBER_TRACES
        tlv(0x42, 4); fputle32(static_cast<int32_t>(total_samples)); // NUMBER_SAMPLES
        tlv(0x43, 1); std::fputc(0x14, fp);                     // float32
        if (h.data_length > 0) {
            tlv(0x44, 2); fputle16(h.data_length);              // DATA_LENGTH
        }
        std::fputc(0x5F, fp); std::fputc(0x00, fp);             // TRACE_BLOCK

        constexpr int64_t CHUNK = 256 * 1024;
        std::vector<float> buf(CHUNK);
        bool cancelled = false;

        for (int ti = 0; ti < n_traces && !cancelled; ti++) {
            if (prog.wasCanceled()) { cancelled = true; break; }
            prog.setLabelText(QString("Exporting trace %1 / %2…").arg(ti + 1).arg(n_traces));
            prog.setValue(ti);
            QApplication::processEvents();

            // Auxiliary data bytes
            if (h.data_length > 0) {
                auto data = trs_file_->readData(ti);
                std::fwrite(data.data(), 1, data.size(), fp);
            }

            // Concatenate each range
            for (const auto& r : ranges) {
                int64_t s   = r.first;
                int64_t end = r.second;
                while (s < end) {
                    int64_t chunk = std::min(CHUNK, end - s);
                    int64_t read  = trs_file_->readSamples(ti, s, chunk, buf.data());
                    if (read <= 0) {
                        // Fill remainder with zeros if read fails
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
            }
        }

        prog.setValue(n_traces);
        std::fclose(fp);

        if (cancelled) {
            QFile::remove(path);
            QMessageBox::information(dlg, "Cancelled", "Export was cancelled.");
        } else {
            QMessageBox::information(dlg, "Export complete",
                QString("Saved %1 trace(s) with %2 samples each to:\n%3")
                    .arg(n_traces).arg(total_samples).arg(path));
        }
    });

    dlg->show();
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-Correlation SCA
// ---------------------------------------------------------------------------

void MainWindow::onRunXCorr() {
    if (!trs_file_) {
        QMessageBox::information(this, "Cross-Correlation", "No file loaded.");
        return;
    }
    const TrsHeader& h = trs_file_->header();
    int n_total   = h.num_traces;
    int n_samples = h.num_samples;

    // ---- Config dialog ----
    QDialog cfg(this);
    cfg.setWindowTitle("Cross-Correlation — configuration");
    auto* vl_cfg = new QVBoxLayout(&cfg);

    // Trace range
    auto* grp_traces = new QGroupBox("Traces");
    auto* fl_traces  = new QFormLayout(grp_traces);
    auto* sp_first   = new QSpinBox; sp_first->setRange(0, std::max(0, n_total - 1)); sp_first->setValue(0);
    auto* sp_count   = new QSpinBox; sp_count->setRange(2, n_total); sp_count->setValue(n_total);
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

    // Update M / memory estimate labels (accounts for pipeline decimation)
    auto updateEstimate = [&]() {
        int64_t ns = sp_s_count->value() == 0 ? n_samples : sp_s_count->value();
        for (const auto& t : pipeline_) ns = t->transformedCount(ns);
        int     st = sp_stride->value();
        int64_t M  = (ns + st - 1) / st;
        lbl_M->setText(QString::number(M));
        double mem_mb = static_cast<double>(M) * M * 4.0 / (1024.0 * 1024.0);
        if (mem_mb >= 1024.0)
            lbl_mem->setText(QString("%1 GB").arg(mem_mb / 1024.0, 0, 'f', 2));
        else
            lbl_mem->setText(QString("%1 MB").arg(mem_mb, 0, 'f', 1));
    };
    connect(sp_stride,  QOverload<int>::of(&QSpinBox::valueChanged), [&](int) { updateEstimate(); });
    connect(sp_s_count, QOverload<int>::of(&QSpinBox::valueChanged), [&](int) { updateEstimate(); });
    updateEstimate();

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
    vl_method->addWidget(rb_baseline);
    vl_method->addWidget(rb_dual);
    vl_method->addWidget(rb_mp);
    vl_method->addWidget(rb_twowin);
    vl_method->addWidget(lbl_dual_warn);

    connect(rb_dual, &QRadioButton::toggled, lbl_dual_warn, &QLabel::setEnabled);
    connect(rb_mp,   &QRadioButton::toggled, [lbl_dual_warn, rb_dual, rb_mp]() {
        lbl_dual_warn->setEnabled(rb_dual->isChecked() || rb_mp->isChecked());
    });

    // Reference window (Two-Window mode only)
    auto* grp_ref   = new QGroupBox("Reference Window (Two-Window mode)");
    auto* fl_ref    = new QFormLayout(grp_ref);
    auto* sp_r_first = new QSpinBox; sp_r_first->setRange(0, std::max(0, n_samples - 1)); sp_r_first->setValue(0);
    auto* sp_r_count = new QSpinBox; sp_r_count->setRange(1, n_samples); sp_r_count->setValue(std::min(512, n_samples));
    fl_ref->addRow("Ref first sample:", sp_r_first);
    fl_ref->addRow("Ref count:",        sp_r_count);
    grp_ref->setVisible(false);
    connect(rb_twowin, &QRadioButton::toggled, grp_ref, &QWidget::setVisible);

    auto* cfg_bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(cfg_bb, &QDialogButtonBox::accepted, &cfg, &QDialog::accept);
    connect(cfg_bb, &QDialogButtonBox::rejected, &cfg, &QDialog::reject);

    vl_cfg->addWidget(grp_traces);
    vl_cfg->addWidget(grp_samples);
    vl_cfg->addWidget(grp_ref);
    vl_cfg->addWidget(grp_ds);
    vl_cfg->addWidget(grp_method);
    vl_cfg->addWidget(cfg_bb);

    if (cfg.exec() != QDialog::Accepted) return;

    int32_t first_trace  = static_cast<int32_t>(sp_first->value());
    int32_t num_traces   = static_cast<int32_t>(sp_count->value());
    int64_t first_sample = static_cast<int64_t>(sp_s_first->value());
    int64_t num_samples_req = static_cast<int64_t>(sp_s_count->value()); // 0 = all
    int32_t stride       = static_cast<int32_t>(sp_stride->value());
    bool    is_twowin    = rb_twowin->isChecked();
    XCorrMethod method   = rb_mp->isChecked()     ? XCorrMethod::MPCleaned
                         : rb_dual->isChecked()    ? XCorrMethod::DualMatrix
                         : is_twowin               ? XCorrMethod::TwoWindow
                                                   : XCorrMethod::Baseline;

    // Memory warning for large matrices (effective M accounts for pipeline)
    {
        int64_t ns = num_samples_req == 0 ? (n_samples - first_sample) : num_samples_req;
        for (const auto& t : pipeline_) ns = t->transformedCount(ns);
        int64_t M  = (ns + stride - 1) / stride;
        double  mem_mb = static_cast<double>(M) * M * 4.0 / (1024.0 * 1024.0);
        if (mem_mb > 2048.0) {
            if (QMessageBox::warning(this, "Memory warning",
                    QString("The output matrix will require ~%1 GB.\nContinue?")
                        .arg(mem_mb / 1024.0, 0, 'f', 1),
                    QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
                return;
        }
    }

    // ---- Progress dialog + computation ----
    QProgressDialog prog("Initialising…", "Cancel", 0, 100, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setMinimumDuration(0);
    prog.setValue(0);
    QApplication::processEvents();

    XCorrResult result;
    std::string err;

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

    bool ok;
    if (is_twowin) {
        int64_t ref_first = static_cast<int64_t>(sp_r_first->value());
        int64_t ref_count = static_cast<int64_t>(sp_r_count->value());
        int64_t ns = (num_samples_req == 0) ? (h.num_samples - first_sample) : num_samples_req;
        ok = computeTwoWindowCorr(
            trs_file_.get(), first_trace, num_traces,
            ref_first, ref_count,
            first_sample, ns,
            stride, pipeline_, result, progCb, err);
    } else {
        ok = computeXCorr(
            trs_file_.get(),
            first_trace, num_traces,
            first_sample, num_samples_req,
            stride, method, pipeline_, result, progCb, err);
    }

    prog.setValue(prog.maximum());

    if (!ok) {
        if (!err.empty())
            QMessageBox::critical(this, "Cross-Correlation failed",
                                  QString::fromStdString(err));
        return;  // cancelled
    }

    // ---- Result window ----
    auto result_ptr = std::make_shared<XCorrResult>(std::move(result));

    auto* dlg = new QDialog(this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    QString method_str = result_ptr->method == XCorrMethod::MPCleaned  ? "MP-Cleaned"
                       : result_ptr->method == XCorrMethod::DualMatrix  ? "Dual Matrix"
                       : result_ptr->method == XCorrMethod::TwoWindow   ? "Two-Window"
                                                                         : "Baseline";
    QString title = (result_ptr->method == XCorrMethod::TwoWindow)
        ? QString("Two-Window Match  search=%1  ref=%2  n=%3")
              .arg(result_ptr->rows).arg(result_ptr->cols).arg(result_ptr->n_traces)
        : QString("Cross-Correlation [%1]  M=%2  n=%3")
              .arg(method_str).arg(result_ptr->M).arg(result_ptr->n_traces);
    if (result_ptr->method == XCorrMethod::MPCleaned)
        title += QString("  λ+=%1  signal=%2")
                     .arg(result_ptr->lambda_plus, 0, 'g', 4)
                     .arg(result_ptr->n_signal);
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

    auto* btn_reset_view = new QPushButton("Reset View");
    auto* btn_exp_png    = new QPushButton("Export PNG…");
    auto* btn_exp_npy    = new QPushButton("Export .npy…");

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

    connect(heatmap, &HeatmapWidget::hoverInfo, dlg, [lbl_hover](int s1, int s2, float val) {
        lbl_hover->setText(
            QString("C[%1, %2] = %3").arg(s1).arg(s2)
                .arg(static_cast<double>(val), 0, 'g', 6));
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

    // Two-window: default to Lukasz colormap + abs value for template-match look
    if (result_ptr->method == XCorrMethod::TwoWindow) {
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
                                                    {}, "PNG images (*.png)");
        if (path.isEmpty()) return;
        if (!heatmap->exportPng(path))
            QMessageBox::critical(dlg, "Export failed", "Could not write:\n" + path);
        else
            QMessageBox::information(dlg, "Saved", "Saved: " + path);
    });

    connect(btn_exp_npy, &QPushButton::clicked, dlg, [=]() {
        QString path = QFileDialog::getSaveFileName(dlg, "Export matrix as NumPy",
                                                    {}, "NumPy files (*.npy)");
        if (path.isEmpty()) return;
        // Write 2-D float32 .npy
        FILE* fp = std::fopen(path.toLocal8Bit().constData(), "wb");
        if (!fp) { QMessageBox::critical(dlg, "Export failed", "Cannot create:\n" + path); return; }
        const uint8_t magic[] = {0x93,'N','U','M','P','Y',0x01,0x00};
        std::fwrite(magic, 1, 8, fp);
        int32_t M = result_ptr->M;
        std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                           std::to_string(M) + ", " + std::to_string(M) + "), }";
        size_t content_len = dict.size() + 1;
        size_t header_len  = ((content_len + 10 + 63) / 64) * 64 - 10;
        dict.resize(header_len - 1, ' ');
        dict += '\n';
        uint16_t hl = static_cast<uint16_t>(header_len);
        uint8_t hl_bytes[2] = {uint8_t(hl & 0xFF), uint8_t(hl >> 8)};
        std::fwrite(hl_bytes, 1, 2, fp);
        std::fwrite(dict.c_str(), 1, dict.size(), fp);
        std::fwrite(result_ptr->matrix.data(), sizeof(float),
                    static_cast<size_t>(M) * static_cast<size_t>(M), fp);
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
        il->addWidget(new QLabel(
            QString("λ+ (MP upper edge) = <b>%1</b>    "
                    "Signal eigenvalues above λ+: <b>%2</b>")
                .arg(result_ptr->lambda_plus, 0, 'g', 5)
                .arg(result_ptr->n_signal)));
        auto* l = qobject_cast<QLabel*>(il->itemAt(0)->widget());
        if (l) l->setTextFormat(Qt::RichText);
        il->addStretch();
    }

    auto* vl = new QVBoxLayout(dlg);
    vl->setContentsMargins(4, 4, 4, 4);
    vl->setSpacing(4);
    if (info_bar) vl->addWidget(info_bar);
    vl->addWidget(ctrl);
    vl->addWidget(proc_row);
    vl->addWidget(heatmap, 1);

    dlg->show();
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
        bool ok;
        int w = QInputDialog::getInt(this, "Window Resample",
                                     "Window size (samples per output point):",
                                     64, 2, 1'000'000, 1, &ok);
        if (!ok) return nullptr;
        return std::make_shared<WindowResampleTransform>(w);
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
    default: return nullptr;
    }
}
