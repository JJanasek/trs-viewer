#include "figure_dialog.h"
#include "figure.h"
#include "plot_widget.h"

#include <QApplication>
#include <QBuffer>
#include <QCheckBox>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QSvgGenerator>
#include <QVBoxLayout>

#include <cmath>
#include <memory>

namespace {

// Everything the dialog remembers between uses (QSettings "figure/…").
struct FigureSettings {
    int    preset    = 0;          // index into figurePresets(); -1 = custom
    double width_cm  = 12.2, height_cm = 4.5;   // = the first preset (Springer, page width)
    int    font      = 0;          // FigureFont
    double font_pt   = 8.0, line_pt = 0.5;
    int    legend    = 0;          // FigureStyle::Legend
    bool   grid = false, box = true, title = false, transparent = false, mono = true;
    int    format    = 0;          // 0 PDF, 1 SVG, 2 PNG
    int    dpi       = 600;

    static FigureSettings load() {
        QSettings st("trs-viewer", "trs-viewer");
        FigureSettings f;
        st.beginGroup("figure");
        f.preset    = st.value("preset", f.preset).toInt();
        f.width_cm  = st.value("width_cm", f.width_cm).toDouble();
        f.height_cm = st.value("height_cm", f.height_cm).toDouble();
        f.font      = st.value("font", f.font).toInt();
        f.font_pt   = st.value("font_pt", f.font_pt).toDouble();
        f.line_pt   = st.value("line_pt", f.line_pt).toDouble();
        f.legend    = st.value("legend", f.legend).toInt();
        f.grid      = st.value("grid", f.grid).toBool();
        f.box       = st.value("box", f.box).toBool();
        f.title     = st.value("title", f.title).toBool();
        f.transparent = st.value("transparent", f.transparent).toBool();
        f.mono      = st.value("mono", f.mono).toBool();
        f.format    = st.value("format", f.format).toInt();
        f.dpi       = st.value("dpi", f.dpi).toInt();
        st.endGroup();
        return f;
    }
    void save() const {
        QSettings st("trs-viewer", "trs-viewer");
        st.beginGroup("figure");
        st.setValue("preset", preset);       st.setValue("width_cm", width_cm);
        st.setValue("height_cm", height_cm); st.setValue("font", font);
        st.setValue("font_pt", font_pt);     st.setValue("line_pt", line_pt);
        st.setValue("legend", legend);       st.setValue("grid", grid);
        st.setValue("box", box);             st.setValue("title", title);
        st.setValue("transparent", transparent);
        st.setValue("mono", mono);
        st.setValue("format", format);       st.setValue("dpi", dpi);
        st.endGroup();
    }
    FigureStyle style() const {
        FigureStyle s;
        s.width_cm = width_cm;  s.height_cm = height_cm;
        s.font_family = resolveFontStyle(static_cast<FigureFont>(font));
        s.font_pt = font_pt;    s.line_pt = line_pt;
        s.grid = grid;          s.box = box;
        s.legend = static_cast<FigureStyle::Legend>(legend);
        s.title = title;        s.transparent = transparent;
        s.mono = mono;
        s.png_dpi = dpi;
        return s;
    }
};

QString formatExt(int f) { return f == 1 ? "svg" : f == 2 ? "png" : "pdf"; }

QByteArray figureSvg(const FigureData& d, const FigureStyle& s) {
    QByteArray out;
    QBuffer buf(&out);
    buf.open(QIODevice::WriteOnly);
    const double w = s.width_cm / 2.54 * 72.0, h = s.height_cm / 2.54 * 72.0;
    QSvgGenerator g;
    g.setOutputDevice(&buf);
    g.setResolution(72);
    g.setSize(QSize(static_cast<int>(std::ceil(w)), static_cast<int>(std::ceil(h))));
    g.setViewBox(QRectF(0, 0, w, h));
    QPainter p(&g);
    renderFigure(p, QRectF(0, 0, w, h), d, s);
    p.end();
    return out;
}

}  // namespace

void copyFigureToClipboard(PlotWidget* pw) {
    if (!pw) return;
    const FigureData d = pw->figureData();
    const FigureStyle s = FigureSettings::load().style();
    auto* mime = new QMimeData;
    QImage img = renderFigureImage(d, s, 300.0 / 72.0);
    const int dpm = static_cast<int>(std::lround(300.0 / 0.0254));
    img.setDotsPerMeterX(dpm);
    img.setDotsPerMeterY(dpm);
    mime->setImageData(img);
    mime->setData("image/svg+xml", figureSvg(d, s));
    QApplication::clipboard()->setMimeData(mime);
}

void showFigureExportDialog(PlotWidget* pw, QWidget* parent) {
    if (!pw) return;
    auto data = std::make_shared<FigureData>(pw->figureData());
    if (data->series.empty()) {
        QMessageBox::information(parent, "Export Figure", "There is nothing in this plot to export.");
        return;
    }
    auto cfg = std::make_shared<FigureSettings>(FigureSettings::load());
    // The plot's own data in sample/raw units; `data` is what gets drawn,
    // after the axis-unit choice below.
    const auto base = std::make_shared<FigureData>(*data);

    QDialog dlg(parent);
    dlg.setWindowTitle("Export Figure");
    auto* top = new QHBoxLayout(&dlg);

    // ---- settings ----
    auto* form_w = new QWidget;
    form_w->setMaximumWidth(360);
    auto* fl = new QFormLayout(form_w);
    // Presets under their group headings (not selectable); item data = the
    // preset's index, -1 for Custom.
    auto* cmb_preset = new QComboBox;
    cmb_preset->setToolTip("What the figure is for. Sets the size and fonts so the figure goes "
                           "into the document without scaling.");
    {
        QString group;
        for (size_t i = 0; i < figurePresets().size(); i++) {
            const FigurePreset& pr = figurePresets()[i];
            if (pr.group != group) {
                group = pr.group;
                cmb_preset->addItem(group, -2);
                if (auto* m = qobject_cast<QStandardItemModel*>(cmb_preset->model())) {
                    QStandardItem* it = m->item(cmb_preset->count() - 1);
                    it->setFlags(it->flags() & ~(Qt::ItemIsSelectable | Qt::ItemIsEnabled));
                    QFont f = it->font(); f.setBold(true); it->setFont(f);
                }
            }
            cmb_preset->addItem(pr.name, static_cast<int>(i));
            cmb_preset->setItemData(cmb_preset->count() - 1, pr.note, Qt::ToolTipRole);
        }
    }
    cmb_preset->addItem("Custom size", -1);
    auto* lbl_preset = new QLabel;
    lbl_preset->setWordWrap(true);
    auto* sp_w = new QDoubleSpinBox; sp_w->setRange(2, 60); sp_w->setDecimals(1); sp_w->setSuffix(" cm");
    auto* sp_h = new QDoubleSpinBox; sp_h->setRange(1.5, 40); sp_h->setDecimals(1); sp_h->setSuffix(" cm");
    auto* cmb_font = new QComboBox;
    for (int f = 0; f < 3; f++) cmb_font->addItem(fontStyleName(static_cast<FigureFont>(f)));
    auto* lbl_font = new QLabel;
    lbl_font->setWordWrap(true);
    auto* sp_fpt = new QDoubleSpinBox; sp_fpt->setRange(5, 24); sp_fpt->setDecimals(1); sp_fpt->setSuffix(" pt");
    sp_fpt->setToolTip("Text size in the figure. Placed at its exported width, it appears at exactly "
                       "this size — usually 1–2 pt below the document's body text.");
    auto* sp_lw = new QDoubleSpinBox; sp_lw->setRange(0.1, 5); sp_lw->setDecimals(2);
    sp_lw->setSingleStep(0.1); sp_lw->setSuffix(" pt");
    auto* cmb_leg = new QComboBox;
    cmb_leg->addItems({"Automatic", "Inside, top right", "Below the plot", "Off"});
    // Axis units: sample index or time (from the file's sample interval, or a
    // rate typed here), raw values or volts (from the file's value scale).
    auto* cmb_x = new QComboBox;
    cmb_x->addItems({"Sample index", "Time"});
    auto* sp_rate = new QDoubleSpinBox;
    sp_rate->setRange(0.0, 1e6);
    sp_rate->setDecimals(3);
    sp_rate->setSuffix(" MS/s");
    sp_rate->setSpecialValueText("unknown");
    sp_rate->setValue(base->dt > 0 ? 1e-6 / base->dt : 0.0);
    sp_rate->setToolTip(base->dt > 0 ? QString("From the file: %1 per sample").arg(
                                           base->dt >= 1e-6 ? QString("%1 µs").arg(base->dt * 1e6)
                                                            : QString("%1 ns").arg(base->dt * 1e9))
                                     : QString("The file has no sample interval — enter the "
                                               "sampling rate to show time."));
    auto* cmb_y = new QComboBox;
    cmb_y->addItems({"Raw values", base->y_unit.isEmpty() || base->y_unit == "V"
                                       ? QString("Voltage (from the file)") : QString("Physical units (from the file)")});
    auto* ed_xl = new QLineEdit(data->x_label);
    auto* ed_yl = new QLineEdit(data->y_label);
    ed_xl->setPlaceholderText("e.g. Time samples");
    ed_yl->setPlaceholderText("e.g. t-value");
    auto* chk_mono  = new QCheckBox("Data in black (print)");
    chk_mono->setToolTip("Published SCA figures usually draw the data thin and black, keeping "
                         "colour for the threshold. Off: keep the plot's own colours (slides).");
    auto* chk_grid  = new QCheckBox("Light grid");
    auto* chk_box   = new QCheckBox("Frame on all four sides");
    auto* chk_title = new QCheckBox("Title (usually left to the LaTeX caption)");
    auto* chk_trans = new QCheckBox("Transparent background (slides)");
    auto* cmb_fmt = new QComboBox;
    cmb_fmt->addItems({"PDF — vector, for LaTeX", "SVG — vector, for slides / Inkscape", "PNG — image"});
    auto* sp_dpi = new QSpinBox; sp_dpi->setRange(72, 2400); sp_dpi->setSingleStep(150);
    sp_dpi->setSuffix(" dpi");
    auto* lbl_hint = new QLabel;
    lbl_hint->setWordWrap(true);
    lbl_hint->setTextInteractionFlags(Qt::TextSelectableByMouse);

    fl->addRow("For:", cmb_preset);
    fl->addRow("", lbl_preset);
    fl->addRow("Size:", [&]{ auto* w = new QWidget; auto* l = new QHBoxLayout(w);
                             l->setContentsMargins(0,0,0,0); l->addWidget(sp_w);
                             l->addWidget(new QLabel("×")); l->addWidget(sp_h); return w; }());
    fl->addRow("Font:", cmb_font);
    fl->addRow("", lbl_font);
    fl->addRow("Font size:", sp_fpt);
    fl->addRow("Line width:", sp_lw);
    if (base->x_samples) {
        fl->addRow("X axis:", [&]{ auto* w = new QWidget; auto* l = new QHBoxLayout(w);
                                   l->setContentsMargins(0,0,0,0); l->addWidget(cmb_x, 1);
                                   l->addWidget(sp_rate, 1); return w; }());
    }
    if (base->y_scale > 0) fl->addRow("Y axis:", cmb_y);
    fl->addRow("X label:", ed_xl);
    fl->addRow("Y label:", ed_yl);
    fl->addRow("Legend:", cmb_leg);
    fl->addRow(chk_mono);
    fl->addRow(chk_grid);
    fl->addRow(chk_box);
    fl->addRow(chk_title);
    fl->addRow(chk_trans);
    fl->addRow("Format:", cmb_fmt);
    fl->addRow("Resolution:", sp_dpi);
    fl->addRow(lbl_hint);
    top->addWidget(form_w);

    // ---- preview ----
    auto* right = new QVBoxLayout;
    auto* preview = new QLabel;
    preview->setMinimumSize(560, 260);
    preview->setAlignment(Qt::AlignCenter);
    preview->setStyleSheet("QLabel { background: #9aa3ad; }");   // shows the page edge
    auto* lbl_size = new QLabel;
    right->addWidget(preview, 1);
    right->addWidget(lbl_size);
    auto* bb = new QDialogButtonBox;
    auto* btn_copy = bb->addButton("Copy to Clipboard", QDialogButtonBox::ActionRole);
    bb->addButton(QDialogButtonBox::Cancel);
    auto* btn_export = bb->addButton("Export…", QDialogButtonBox::AcceptRole);
    btn_export->setDefault(true);
    right->addWidget(bb);
    top->addLayout(right, 1);

    // ---- wiring ----
    bool loading = true;
    auto toUi = [&]() {
        loading = true;
        const int ci = cmb_preset->findData(cfg->preset >= 0 && cfg->preset < static_cast<int>(figurePresets().size())
                                            ? cfg->preset : -1);
        cmb_preset->setCurrentIndex(ci >= 0 ? ci : cmb_preset->count() - 1);
        sp_w->setValue(cfg->width_cm);   sp_h->setValue(cfg->height_cm);
        cmb_font->setCurrentIndex(cfg->font);
        sp_fpt->setValue(cfg->font_pt);  sp_lw->setValue(cfg->line_pt);
        cmb_leg->setCurrentIndex(cfg->legend);
        chk_grid->setChecked(cfg->grid); chk_box->setChecked(cfg->box);
        chk_title->setChecked(cfg->title); chk_trans->setChecked(cfg->transparent);
        chk_mono->setChecked(cfg->mono);
        cmb_fmt->setCurrentIndex(cfg->format);
        sp_dpi->setValue(cfg->dpi);
        loading = false;
    };
    auto refresh = [&]() {
        const FigureStyle st = cfg->style();
        const auto f = static_cast<FigureFont>(cfg->font);
        lbl_font->setText(fontStyleIsExact(f)
            ? QString("<small>Using %1.</small>").arg(st.font_family)
            : QString("<small>Using %1. Install Latin Modern for an exact LaTeX match.</small>")
                  .arg(st.font_family));
        sp_dpi->setEnabled(cfg->format == 2);
        chk_trans->setEnabled(cfg->format != 0);
        // Full-width presets are exactly \textwidth / \columnwidth.
        QString width_tex = QString("%1cm").arg(cfg->width_cm);
        const bool is_preset = cfg->preset >= 0 && cfg->preset < static_cast<int>(figurePresets().size());
        if (is_preset && !figurePresets()[static_cast<size_t>(cfg->preset)].tex_width.isEmpty())
            width_tex = figurePresets()[static_cast<size_t>(cfg->preset)].tex_width;
        lbl_preset->setText(is_preset
            ? QString("<small>%1</small>").arg(figurePresets()[static_cast<size_t>(cfg->preset)].note.toHtmlEscaped())
            : QString("<small>Your own size and fonts.</small>"));
        lbl_hint->setText(QString("<small>LaTeX: <tt>\\includegraphics[width=%1]{figure}</tt><br>"
                                  "Already that wide, so the text stays at %2 pt.</small>")
                              .arg(width_tex).arg(cfg->font_pt));
        // Preview at the dialog's resolution, with the true proportions.
        const double w_pt = cfg->width_cm / 2.54 * 72.0, h_pt = cfg->height_cm / 2.54 * 72.0;
        const QSize avail = preview->size() - QSize(16, 16);
        const double dpr = preview->devicePixelRatioF();
        const double k = std::min(avail.width() / w_pt, avail.height() / h_pt) * dpr;
        QImage img = renderFigureImage(*data, st, k);
        img.setDevicePixelRatio(dpr);
        preview->setPixmap(QPixmap::fromImage(img));
        lbl_size->setText(QString("%1 × %2 cm  (%3 × %4 in)")
                              .arg(cfg->width_cm, 0, 'f', 1).arg(cfg->height_cm, 0, 'f', 1)
                              .arg(cfg->width_cm / 2.54, 0, 'f', 2).arg(cfg->height_cm / 2.54, 0, 'f', 2));
    };
    auto fromUi = [&]() {
        if (loading) return;
        cfg->width_cm = sp_w->value();   cfg->height_cm = sp_h->value();
        cfg->font = cmb_font->currentIndex();
        cfg->font_pt = sp_fpt->value();  cfg->line_pt = sp_lw->value();
        cfg->legend = cmb_leg->currentIndex();
        cfg->grid = chk_grid->isChecked(); cfg->box = chk_box->isChecked();
        cfg->title = chk_title->isChecked(); cfg->transparent = chk_trans->isChecked();
        cfg->mono = chk_mono->isChecked();
        AxisUnits u;
        u.time = cmb_x->currentIndex() == 1;
        u.dt   = sp_rate->value() > 0 ? 1e-6 / sp_rate->value() : 0.0;
        u.physical_y = cmb_y->currentIndex() == 1;
        *data = applyAxisUnits(*base, u, nullptr, nullptr);
        data->x_label = ed_xl->text();
        data->y_label = ed_yl->text();
        cfg->format = cmb_fmt->currentIndex();
        cfg->dpi = sp_dpi->value();
        refresh();
    };
    // Editing size/font by hand turns the preset into "Custom".
    auto edited = [&]() {
        if (loading) return;
        loading = true;
        cmb_preset->setCurrentIndex(cmb_preset->count() - 1);
        cfg->preset = -1;
        loading = false;
        fromUi();
    };
    QObject::connect(cmb_preset, QOverload<int>::of(&QComboBox::currentIndexChanged), &dlg, [&](int ci) {
        if (loading) return;
        const int i = cmb_preset->itemData(ci).toInt();
        if (i < 0 || i >= static_cast<int>(figurePresets().size())) { cfg->preset = -1; refresh(); return; }
        const FigurePreset& pr = figurePresets()[static_cast<size_t>(i)];
        cfg->preset = i;
        cfg->width_cm = pr.width_cm; cfg->height_cm = pr.height_cm;
        cfg->font = static_cast<int>(pr.font);
        cfg->font_pt = pr.font_pt;   cfg->line_pt = pr.line_pt;
        cfg->mono = pr.mono;
        toUi();
        refresh();
    });
    for (QDoubleSpinBox* sp : {sp_w, sp_h, sp_fpt, sp_lw})
        QObject::connect(sp, QOverload<double>::of(&QDoubleSpinBox::valueChanged), &dlg, edited);
    QObject::connect(cmb_font, QOverload<int>::of(&QComboBox::currentIndexChanged), &dlg, edited);
    for (QComboBox* c : {cmb_leg, cmb_fmt})
        QObject::connect(c, QOverload<int>::of(&QComboBox::currentIndexChanged), &dlg, fromUi);
    for (QLineEdit* e : {ed_xl, ed_yl})
        QObject::connect(e, &QLineEdit::textChanged, &dlg, fromUi);
    for (QCheckBox* c : {chk_grid, chk_box, chk_title, chk_trans, chk_mono})
        QObject::connect(c, &QCheckBox::toggled, &dlg, fromUi);
    QObject::connect(sp_dpi, QOverload<int>::of(&QSpinBox::valueChanged), &dlg, fromUi);

    // Changing the axis units writes the matching label (still editable).
    auto unitsChanged = [&]() {
        AxisUnits u;
        u.time = cmb_x->currentIndex() == 1;
        u.dt   = sp_rate->value() > 0 ? 1e-6 / sp_rate->value() : 0.0;
        u.physical_y = cmb_y->currentIndex() == 1;
        sp_rate->setEnabled(u.time);
        QString xl = ed_xl->text(), yl = ed_yl->text();
        applyAxisUnits(*base, u, base->x_samples ? &xl : nullptr, base->y_scale > 0 ? &yl : nullptr);
        if (base->y_scale > 0 && !u.physical_y) yl = base->y_label.isEmpty() ? QString("ADC value") : base->y_label;
        { QSignalBlocker b1(ed_xl), b2(ed_yl); ed_xl->setText(xl); ed_yl->setText(yl); }
        fromUi();
    };
    QObject::connect(cmb_x, QOverload<int>::of(&QComboBox::currentIndexChanged), &dlg, unitsChanged);
    QObject::connect(cmb_y, QOverload<int>::of(&QComboBox::currentIndexChanged), &dlg, unitsChanged);
    QObject::connect(sp_rate, QOverload<double>::of(&QDoubleSpinBox::valueChanged), &dlg, unitsChanged);

    QObject::connect(btn_copy, &QPushButton::clicked, &dlg, [&]() {
        cfg->save();
        copyFigureToClipboard(pw);
    });
    QObject::connect(bb, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    QObject::connect(btn_export, &QPushButton::clicked, &dlg, [&]() {
        QSettings st("trs-viewer", "trs-viewer");
        const QString ext = formatExt(cfg->format);
        QString dir = st.value("recentDir/figure").toString();
        QString base = data->title.isEmpty() ? QString("figure") : data->title;
        base.replace(QRegularExpression("[^A-Za-z0-9_-]+"), "_");
        const QString path = QFileDialog::getSaveFileName(&dlg, "Export Figure",
            QDir(dir).filePath(base.left(60) + "." + ext),
            QString("%1 (*.%2)").arg(cmb_fmt->currentText().section(" —", 0, 0)).arg(ext));
        if (path.isEmpty()) return;
        st.setValue("recentDir/figure", QFileInfo(path).absolutePath());
        QString err;
        QString out = path;
        if (QFileInfo(out).suffix().toLower() != ext) out += "." + ext;
        if (!exportFigure(out, *data, cfg->style(), err)) {
            QMessageBox::critical(&dlg, "Export failed", err);
            return;
        }
        cfg->save();
        dlg.accept();
    });

    toUi();
    {
        QSignalBlocker b1(cmb_x), b2(cmb_y);
        cmb_x->setCurrentIndex(base->x_samples && base->dt > 0 ? 1 : 0);
        cmb_y->setCurrentIndex(base->y_scale > 0 ? 1 : 0);
    }
    unitsChanged();
    dlg.resize(1080, 720);
    dlg.show();
    QApplication::processEvents();   // let the preview label get its real size
    refresh();
    dlg.exec();
}
