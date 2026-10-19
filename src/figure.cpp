#include "figure.h"

#include <QFileInfo>
#include <QFont>
#include <QFontDatabase>
#include <QFontMetricsF>
#include <QImage>
#include <QPageSize>
#include <QPainter>
#include <QPainterPath>
#include <QPdfWriter>
#include <QSvgGenerator>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

const std::vector<FigurePreset>& figurePresets() {
    // Widths are the LaTeX text/column widths of each class; heights give the
    // wide, short strips trace plots usually are. Fonts: serif to match the
    // body text in papers and theses, sans for slides.
    // Widths are the LaTeX \textwidth / \columnwidth of each class (checked
    // against the class files); heights give the wide, short strips trace
    // plots usually are (≈3:1 full width, ≈2:1 half width). Serif to match
    // the body text in papers and theses, sans for slides; data in black for
    // print, as published SCA figures do.
    static const std::vector<FigurePreset> presets = {
        {"LNCS — full width",          12.2, 4.5, FigureFont::LatexSerif, 8.0, 0.5, true,
         "Springer LNCS (CARDIS, COSADE, CT-RSA, older CHES): \\textwidth = 12.2 cm"},
        {"LNCS — half width",           5.9, 3.0, FigureFont::LatexSerif, 7.0, 0.4, true,
         "Two subfigures side by side in LNCS"},
        {"TCHES — full width",         13.6, 4.8, FigureFont::LatexSerif, 8.0, 0.5, true,
         "IACR Transactions (iacrtrans / iacrj): \\textwidth = 13.65 cm"},
        {"TCHES — half width",          6.6, 3.2, FigureFont::LatexSerif, 7.0, 0.4, true,
         "Two subfigures side by side in TCHES"},
        {"IEEE — one column",           8.8, 4.0, FigureFont::Times, 8.0, 0.5, true,
         "IEEEtran two-column: \\columnwidth = 8.85 cm (3.49 in)"},
        {"IEEE — two columns",         18.1, 5.0, FigureFont::Times, 8.0, 0.5, true,
         "IEEEtran figure* across both columns: \\textwidth = 18.1 cm (7.14 in)"},
        {"Thesis (A4) — full width",   15.0, 5.5, FigureFont::LatexSerif, 9.0, 0.6, true,
         "Typical A4 thesis text width (KOMA DIV 10–12: 14.7–15.8 cm)"},
        {"Beamer 16:9 — full width",   14.0, 6.0, FigureFont::Sans, 9.0, 0.8, false,
         "Beamer aspectratio=169: 16 × 9 cm frame, \\textwidth = 14 cm"},
        {"Beamer 16:9 — half width",    6.8, 5.0, FigureFont::Sans, 8.0, 0.8, false,
         "One of two columns on a 16:9 Beamer slide"},
        {"Beamer 4:3 — full width",    10.8, 5.5, FigureFont::Sans, 9.0, 0.8, false,
         "Default Beamer: 12.8 × 9.6 cm frame, \\textwidth = 10.8 cm"},
        {"PowerPoint 16:9 — full",     30.0, 12.0, FigureFont::Sans, 16.0, 1.5, false,
         "33.9 × 19.1 cm slide; export SVG (or PNG at 300 dpi)"},
        {"PowerPoint 16:9 — half",     15.0, 10.0, FigureFont::Sans, 16.0, 1.5, false,
         "One of two columns on a PowerPoint slide"},
    };
    return presets;
}

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

namespace {
const QStringList& fontCandidates(FigureFont f) {
    static const QStringList latex = {"Latin Modern Roman", "LM Roman 10", "CMU Serif",
                                      "Computer Modern", "CMU Serif Roman",
                                      "TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                      "Times New Roman", "DejaVu Serif"};
    static const QStringList times = {"TeX Gyre Termes", "Nimbus Roman", "Liberation Serif",
                                      "Times New Roman", "Times", "DejaVu Serif"};
    static const QStringList sans  = {"DejaVu Sans", "Liberation Sans", "Arial", "Helvetica",
                                      "Source Sans 3", "Noto Sans"};
    return f == FigureFont::Times ? times : f == FigureFont::Sans ? sans : latex;
}
}  // namespace

QString fontStyleName(FigureFont f) {
    switch (f) {
    case FigureFont::LatexSerif: return "Serif — LaTeX look";
    case FigureFont::Times:      return "Times — IEEE / ACM";
    case FigureFont::Sans:       return "Sans — slides";
    }
    return {};
}

QString resolveFontStyle(FigureFont f) {
    const QStringList installed = QFontDatabase::families();
    for (const QString& c : fontCandidates(f))
        if (installed.contains(c, Qt::CaseInsensitive)) return c;
    return f == FigureFont::Sans ? QString("Sans Serif") : QString("Serif");
}

bool fontStyleIsExact(FigureFont f) {
    // "Exact" = one of the families the style is named after, not a fallback.
    const int exact = f == FigureFont::LatexSerif ? 5 : f == FigureFont::Times ? 4 : 4;
    const QStringList installed = QFontDatabase::families();
    const QStringList& c = fontCandidates(f);
    for (int i = 0; i < std::min(exact, static_cast<int>(c.size())); i++)
        if (installed.contains(c[i], Qt::CaseInsensitive)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Ticks
// ---------------------------------------------------------------------------

std::vector<double> niceTicks(double lo, double hi, int target) {
    std::vector<double> out;
    if (!(hi > lo) || target < 1) return out;
    const double raw  = (hi - lo) / target;
    const double mag  = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    const double step = (norm < 1.5 ? 1.0 : norm < 3.0 ? 2.0 : norm < 7.0 ? 5.0 : 10.0) * mag;
    const double first = std::ceil(lo / step - 1e-9) * step;
    for (double v = first; v <= hi + step * 1e-9; v += step)
        out.push_back(std::fabs(v) < step * 1e-9 ? 0.0 : v);
    return out;
}

namespace {

// Tick labels: plain numbers; a common power of ten moves to the axis end
// (×10⁴), as in matplotlib/pgfplots, instead of the screen's k/M suffixes.
struct TickFormat { int exponent = 0; int decimals = 0; };

TickFormat tickFormat(const std::vector<double>& ticks) {
    TickFormat f;
    double maxabs = 0;
    for (double t : ticks) maxabs = std::max(maxabs, std::fabs(t));
    if (maxabs >= 1e5 || (maxabs > 0 && maxabs < 1e-3))
        f.exponent = static_cast<int>(std::floor(std::log10(maxabs)));
    if (ticks.size() >= 2) {
        double step = std::fabs(ticks.back() - ticks.front());
        for (size_t i = 1; i < ticks.size(); i++)
            step = std::min(step, std::fabs(ticks[i] - ticks[i - 1]));
        step /= std::pow(10.0, f.exponent);
        f.decimals = std::clamp(static_cast<int>(std::ceil(-std::log10(step) - 1e-9)), 0, 6);
        // Enough digits for every tick (a 4.5 threshold among integer ticks).
        for (double t : ticks) {
            const double v = std::fabs(t) / std::pow(10.0, f.exponent);
            while (f.decimals < 6 && std::fabs(v * std::pow(10.0, f.decimals)
                                               - std::round(v * std::pow(10.0, f.decimals))) > 1e-6)
                f.decimals++;
        }
    }
    return f;
}

QString tickText(double v, const TickFormat& f) {
    const double s = v / std::pow(10.0, f.exponent);
    QString t = QString::number(s, 'f', f.decimals);
    // Only the decimals each tick needs (20, 4.5, 0 — not 20.0, 4.5, 0.0).
    if (t.contains('.')) {
        while (t.endsWith('0')) t.chop(1);
        if (t.endsWith('.')) t.chop(1);
    }
    if (t.startsWith('-')) t.replace(0, 1, QChar(char16_t(0x2212)));   // real minus sign
    if (t == QString(QChar(char16_t(0x2212))) + "0") t = "0";
    return t;
}

QString superscript(int n) {
    static const char16_t sup[] = {0x2070, 0x00B9, 0x00B2, 0x00B3, 0x2074,
                                   0x2075, 0x2076, 0x2077, 0x2078, 0x2079};
    QString out;
    if (n < 0) { out += QChar(char16_t(0x207B)); n = -n; }
    for (QChar c : QString::number(n)) out += QChar(sup[c.digitValue()]);
    return out;
}

QFont figureFont(const FigureStyle& s, double scale = 1.0) {
    QFont f(s.font_family);
    f.setPointSizeF(s.font_pt * scale);
    f.setStyleHint(s.font_family.contains("Sans") ? QFont::SansSerif : QFont::Serif);
    f.setHintingPreference(QFont::PreferNoHinting);   // exact glyph geometry in PDFs
    return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void renderFigure(QPainter& p, const QRectF& page, const FigureData& d, const FigureStyle& s) {
    p.save();
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setRenderHint(QPainter::TextAntialiasing, true);
    if (!s.transparent) p.fillRect(page, Qt::white);

    const QColor ink(0x22, 0x22, 0x22);
    const QColor grid_col(0xDD, 0xDD, 0xDD);
    const QFont  font  = figureFont(s);
    const QFont  small = figureFont(s, 0.85);
    const QFontMetricsF fm(font, p.device());
    const double fh   = fm.height();
    const double tick = 2.5;                       // pt, drawn inward (as in pgfplots/MATLAB figures)
    const double axis_w = 0.5;                     // pt

    const std::vector<double> xt = niceTicks(d.x0, d.x1, std::max(3, static_cast<int>(page.width() / 55)));
    std::vector<double> yt = niceTicks(d.y0, d.y1, std::max(3, static_cast<int>(page.height() / 28)));
    // Papers label the TVLA threshold on the y axis (−4.5 / 4.5); add it as
    // a tick unless an existing one is already close.
    if (d.thresholds && yt.size() >= 2) {
        const double step = std::fabs(yt[1] - yt[0]);
        // The threshold wins over a regular tick that would crowd it.
        auto addTick = [&](double v) {
            if (v <= d.y0 || v >= d.y1) return;
            yt.erase(std::remove_if(yt.begin(), yt.end(),
                         [&](double t) { return std::fabs(t - v) < step * 0.35; }), yt.end());
            yt.push_back(v);
        };
        addTick(d.thr_pos);
        if (!d.one_sided) addTick(d.thr_neg);
        std::sort(yt.begin(), yt.end());
    }
    const TickFormat xf = tickFormat(xt), yf = tickFormat(yt);

    // Print style: data in black, secondary curves grey (colour only for
    // the threshold), as in most published SCA figures.
    auto dataColor = [&](const FigureSeries& se) {
        if (!s.mono) return se.color;
        return se.overlay ? QColor(0x99, 0x99, 0x99) : QColor(0x11, 0x11, 0x11);
    };

    // ---- legend entries ----
    struct Entry { QString label; QColor color; Qt::PenStyle style; double width; bool filled; };
    std::vector<Entry> entries;
    for (const auto& se : d.series)
        if (!se.label.isEmpty())
            entries.push_back({se.label, dataColor(se), Qt::SolidLine, s.line_pt, se.filled && !s.mono});
    const QColor thr_col(0xD6, 0x27, 0x28);
    const bool thr_visible = d.thresholds &&
        ((d.thr_pos > d.y0 && d.thr_pos < d.y1) || (!d.one_sided && d.thr_neg > d.y0 && d.thr_neg < d.y1));
    if (thr_visible)   // only list it when the line is actually drawn
        entries.push_back({d.one_sided ? QString("threshold %1").arg(d.thr_pos)
                                       : QString("threshold ±%1").arg(d.thr_pos),
                           thr_col, Qt::DashLine, 0.8, false});
    FigureStyle::Legend legend = s.legend;
    if (legend == FigureStyle::Legend::Auto) {
        // In print style several primary curves are all black — a legend
        // can't tell them apart, so leave it to the caption.
        int primaries = 0;
        for (const auto& se : d.series) primaries += (!se.overlay && !se.label.isEmpty()) ? 1 : 0;
        const bool indistinct = s.mono && primaries > 1;
        legend = (entries.size() >= 2 && entries.size() <= 6 && !indistinct)
                     ? FigureStyle::Legend::Inside : FigureStyle::Legend::Off;
    }
    if (entries.empty()) legend = FigureStyle::Legend::Off;

    // ---- margins from the text that has to fit ----
    double ytick_w = 0;
    for (double v : yt) ytick_w = std::max(ytick_w, fm.horizontalAdvance(tickText(v, yf)));
    const double pad = 2.0;
    double left   = pad + (d.y_label.isEmpty() ? 0 : fh + 3) + ytick_w + 3;
    double bottom = pad + (d.x_label.isEmpty() ? 0 : fh + 1) + fh + 2;
    double top    = pad + fh * 0.5 + (s.title && !d.title.isEmpty() ? fh * 1.4 : 0)
                  + (yf.exponent ? fh : 0);
    double right  = pad + fm.horizontalAdvance(tickText(xt.empty() ? 0 : xt.back(), xf)) * 0.5;
    if (xf.exponent) right = std::max(right, fm.horizontalAdvance("×10" + superscript(xf.exponent)) + pad);
    double legend_h = 0;
    if (legend == FigureStyle::Legend::Below) {
        legend_h = fh * 1.3;
        bottom += legend_h;
    }
    const QRectF plot(page.left() + left, page.top() + top,
                      std::max(10.0, page.width() - left - right),
                      std::max(10.0, page.height() - top - bottom));

    auto X = [&](double v) { return plot.left() + (v - d.x0) / (d.x1 - d.x0) * plot.width(); };
    auto Y = [&](double v) { return plot.bottom() - (v - d.y0) / (d.y1 - d.y0) * plot.height(); };

    // ---- title ----
    if (s.title && !d.title.isEmpty()) {
        p.setFont(font);
        p.setPen(ink);
        p.drawText(QRectF(page.left(), page.top() + pad, page.width(), fh * 1.2),
                   Qt::AlignHCenter | Qt::AlignVCenter, d.title);
    }

    // ---- grid ----
    if (s.grid) {
        p.setPen(QPen(grid_col, 0.4));
        for (double v : xt) if (v > d.x0 && v < d.x1) p.drawLine(QPointF(X(v), plot.top()), QPointF(X(v), plot.bottom()));
        for (double v : yt) if (v > d.y0 && v < d.y1) p.drawLine(QPointF(plot.left(), Y(v)), QPointF(plot.right(), Y(v)));
    }

    // ---- data (clipped to the axes) ----
    p.save();
    p.setClipRect(plot);
    for (const auto& b : d.bands) {
        QColor c = b.color; c.setAlphaF(0.10);
        p.fillRect(QRectF(QPointF(X(b.x0), plot.top()), QPointF(X(b.x1), plot.bottom())), c);
    }
    // Overlays first, so the primary curves sit on top.
    for (int pass = 0; pass < 2; pass++) {
        for (const auto& se : d.series) {
            if (se.overlay != (pass == 0)) continue;
            const double lw = se.overlay ? s.line_pt * 0.8 : s.line_pt;
            const QColor col = dataColor(se);
            if (!se.ymin.empty()) {
                // Envelope: the area between per-column min and max.
                QPainterPath path;
                for (size_t i = 0; i < se.x.size(); i++) {
                    const QPointF pt(X(se.x[i]), Y(se.ymax[i]));
                    if (i == 0) path.moveTo(pt); else path.lineTo(pt);
                }
                for (size_t i = se.x.size(); i-- > 0;)
                    path.lineTo(QPointF(X(se.x[i]), Y(se.ymin[i])));
                path.closeSubpath();
                p.setPen(QPen(col, lw * 0.5));
                p.setBrush(col);
                p.drawPath(path);
                p.setBrush(Qt::NoBrush);
                continue;
            }
            if (se.y.empty()) continue;
            QPainterPath path;
            for (size_t i = 0; i < se.x.size() && i < se.y.size(); i++) {
                const QPointF pt(X(se.x[i]), Y(se.y[i]));
                if (i == 0) path.moveTo(pt); else path.lineTo(pt);
            }
            if (se.filled && !s.mono) {   // screen nicety; print figures don't fill
                QPainterPath area = path;
                const double base = Y(std::clamp(0.0, d.y0, d.y1));
                area.lineTo(QPointF(X(se.x.back()), base));
                area.lineTo(QPointF(X(se.x.front()), base));
                area.closeSubpath();
                QColor fc = col; fc.setAlphaF(s.mono ? 0.15 : 0.25);
                p.fillPath(area, fc);
            }
            QPen pen(col, lw);
            pen.setJoinStyle(Qt::RoundJoin);
            pen.setCapStyle(Qt::RoundCap);
            p.setPen(pen);
            p.drawPath(path);
        }
    }
    if (d.thresholds) {
        QPen tp(thr_col, 0.8, Qt::CustomDashLine);
        tp.setDashPattern({5, 3});   // in pen widths
        p.setPen(tp);
        auto hline = [&](double v) {
            if (v > d.y0 && v < d.y1) p.drawLine(QPointF(plot.left(), Y(v)), QPointF(plot.right(), Y(v)));
        };
        hline(d.thr_pos);
        if (!d.one_sided) hline(d.thr_neg);
    }
    p.setFont(small);
    for (const auto& b : d.bands) {
        if (b.label.isEmpty()) continue;
        QColor c = b.color.darker(130);
        p.setPen(c);
        p.drawText(QRectF(X(b.x0) + 2, plot.top() + 1, std::max(0.0, X(b.x1) - X(b.x0) - 4), fh),
                   Qt::AlignLeft | Qt::AlignTop, b.label);
    }
    p.restore();

    // ---- axes ----
    p.setPen(QPen(ink, axis_w));
    p.drawLine(plot.bottomLeft(), plot.bottomRight());
    p.drawLine(plot.bottomLeft(), plot.topLeft());
    if (s.box) {
        p.drawLine(plot.topLeft(), plot.topRight());
        p.drawLine(plot.topRight(), plot.bottomRight());
    }
    p.setFont(font);
    for (double v : xt) {
        if (v < d.x0 - 1e-12 * std::fabs(d.x1 - d.x0) || v > d.x1 + 1e-12 * std::fabs(d.x1 - d.x0)) continue;
        const double x = X(v);
        p.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() - tick));
        if (s.box) p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.top() + tick));
        p.drawText(QRectF(x - 40, plot.bottom() + 2, 80, fh),
                   Qt::AlignHCenter | Qt::AlignTop, tickText(v, xf));
    }
    for (double v : yt) {
        if (v < d.y0 - 1e-12 * std::fabs(d.y1 - d.y0) || v > d.y1 + 1e-12 * std::fabs(d.y1 - d.y0)) continue;
        const double y = Y(v);
        p.drawLine(QPointF(plot.left(), y), QPointF(plot.left() + tick, y));
        if (s.box) p.drawLine(QPointF(plot.right(), y), QPointF(plot.right() - tick, y));
        p.drawText(QRectF(plot.left() - 3 - ytick_w - 20, y - fh / 2, ytick_w + 20, fh),
                   Qt::AlignRight | Qt::AlignVCenter, tickText(v, yf));
    }
    if (xf.exponent)
        p.drawText(QRectF(plot.right() - 60, plot.bottom() + 2 + fh, 60 + right, fh),
                   Qt::AlignRight | Qt::AlignTop, "×10" + superscript(xf.exponent));
    if (yf.exponent)
        p.drawText(QRectF(plot.left(), plot.top() - fh * 1.1, 80, fh),
                   Qt::AlignLeft | Qt::AlignBottom, "×10" + superscript(yf.exponent));

    // Axis labels
    if (!d.x_label.isEmpty())
        p.drawText(QRectF(plot.left(), plot.bottom() + 2 + fh + 1, plot.width(), fh),
                   Qt::AlignHCenter | Qt::AlignTop, d.x_label);
    if (!d.y_label.isEmpty()) {
        p.save();
        p.translate(page.left() + pad + fh * 0.5, plot.center().y());
        p.rotate(-90);
        p.drawText(QRectF(-plot.height() / 2 - 40, -fh / 2, plot.height() + 80, fh),
                   Qt::AlignCenter, d.y_label);
        p.restore();
    }

    // ---- legend ----
    if (legend != FigureStyle::Legend::Off) {
        p.setFont(small);
        const QFontMetricsF sm(small, p.device());
        const double sw = 12, gap = 3, row = sm.height();
        if (legend == FigureStyle::Legend::Inside) {
            double w = 0;
            for (const auto& e : entries) w = std::max(w, sm.horizontalAdvance(e.label));
            const QRectF box(plot.right() - (w + sw + gap + 8) - 3, plot.top() + 3,
                             w + sw + gap + 8, entries.size() * row + 4);
            p.setPen(QPen(QColor(0xBB, 0xBB, 0xBB), 0.4));
            p.setBrush(QColor(255, 255, 255, 230));
            p.drawRect(box);
            p.setBrush(Qt::NoBrush);
            double y = box.top() + 2;
            for (const auto& e : entries) {
                const double cy = y + row / 2;
                if (e.filled) {
                    QColor fc = e.color; fc.setAlphaF(0.35);
                    p.fillRect(QRectF(box.left() + 4, cy - 2.5, sw, 5), fc);
                }
                QPen pen(e.color, std::max(0.8, e.width), e.style);
                p.setPen(pen);
                p.drawLine(QPointF(box.left() + 4, cy), QPointF(box.left() + 4 + sw, cy));
                p.setPen(ink);
                p.drawText(QRectF(box.left() + 4 + sw + gap, y, w + 2, row),
                           Qt::AlignLeft | Qt::AlignVCenter, e.label);
                y += row;
            }
        } else {   // Below: one row, centred under the x label
            double total = 0;
            for (const auto& e : entries) total += sw + gap + sm.horizontalAdvance(e.label) + 10;
            double x = plot.center().x() - total / 2;
            const double cy = page.bottom() - pad - legend_h / 2;
            for (const auto& e : entries) {
                p.setPen(QPen(e.color, std::max(0.8, e.width), e.style));
                p.drawLine(QPointF(x, cy), QPointF(x + sw, cy));
                p.setPen(ink);
                const double tw = sm.horizontalAdvance(e.label);
                p.drawText(QRectF(x + sw + gap, cy - row / 2, tw + 2, row),
                           Qt::AlignLeft | Qt::AlignVCenter, e.label);
                x += sw + gap + tw + 10;
            }
        }
    }
    p.restore();
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

QImage renderFigureImage(const FigureData& d, const FigureStyle& s, double scale) {
    const double w_pt = s.width_cm  / 2.54 * 72.0;
    const double h_pt = s.height_cm / 2.54 * 72.0;
    QImage img(std::max(1, static_cast<int>(std::ceil(w_pt * scale))),
               std::max(1, static_cast<int>(std::ceil(h_pt * scale))),
               QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);
    // Lay text out at 72 dpi (1 pt = 1 unit) and scale the painter, so the
    // image is the vector figure at this resolution.
    const int dpm72 = static_cast<int>(std::lround(72.0 / 0.0254));
    img.setDotsPerMeterX(dpm72);
    img.setDotsPerMeterY(dpm72);
    QPainter p(&img);
    p.scale(scale, scale);
    renderFigure(p, QRectF(0, 0, w_pt, h_pt), d, s);
    p.end();
    return img;
}

bool exportFigure(const QString& path, const FigureData& d, const FigureStyle& s, QString& error) {
    const double w_pt = s.width_cm  / 2.54 * 72.0;
    const double h_pt = s.height_cm / 2.54 * 72.0;
    const QString ext = QFileInfo(path).suffix().toLower();
    const QRectF page(0, 0, w_pt, h_pt);

    if (ext == "pdf") {
        QPdfWriter w(path);
        w.setResolution(72);   // 1 painter unit = 1 pt
        w.setPageSize(QPageSize(QSizeF(s.width_cm * 10.0, s.height_cm * 10.0), QPageSize::Millimeter));
        w.setPageMargins(QMarginsF(0, 0, 0, 0));
        w.setCreator("trs-viewer");
        w.setTitle(d.title);
        QPainter p;
        if (!p.begin(&w)) { error = "Cannot write " + path; return false; }
        renderFigure(p, page, d, s);
        p.end();
        return true;
    }
    if (ext == "svg") {
        QSvgGenerator g;
        g.setFileName(path);
        g.setResolution(72);
        g.setSize(QSize(static_cast<int>(std::ceil(w_pt)), static_cast<int>(std::ceil(h_pt))));
        g.setViewBox(page);
        g.setTitle(d.title);
        QPainter p;
        if (!p.begin(&g)) { error = "Cannot write " + path; return false; }
        renderFigure(p, page, d, s);
        p.end();
        return true;
    }
    if (ext == "png") {
        QImage img = renderFigureImage(d, s, s.png_dpi / 72.0);
        // Record the real resolution so LaTeX/Word place it at the right size.
        const int dpm = static_cast<int>(std::lround(s.png_dpi / 0.0254));
        img.setDotsPerMeterX(dpm);
        img.setDotsPerMeterY(dpm);
        if (!img.save(path, "PNG")) { error = "Cannot write " + path; return false; }
        return true;
    }
    error = "Unknown format \"" + ext + "\" — use .pdf, .svg or .png.";
    return false;
}

// ---------------------------------------------------------------------------
// Physical axis units
// ---------------------------------------------------------------------------

namespace {
// Picks an SI prefix so the largest value lands in [1, 1000).
std::pair<double, QString> siPrefix(double maxabs, const QString& unit) {
    struct P { double f; const char* p; };
    static const P ps[] = {{1, ""}, {1e3, "m"}, {1e6, "µ"}, {1e9, "n"}, {1e12, "p"}};
    for (const P& p : ps)
        if (maxabs * p.f >= 1.0 || p.f == 1e12) return {p.f, QString(p.p) + unit};
    return {1.0, unit};
}
}  // namespace

FigureData applyAxisUnits(const FigureData& d, const AxisUnits& u, QString* x_label, QString* y_label) {
    FigureData o = d;
    if (u.time && u.dt > 0 && d.x_samples) {
        const double maxabs = std::max(std::fabs(d.x0), std::fabs(d.x1)) * u.dt;
        const auto [f, unit] = siPrefix(maxabs, "s");
        const double k = u.dt * f;
        auto conv = [&](double& v) { v *= k; };
        conv(o.x0); conv(o.x1);
        for (auto& se : o.series) for (double& x : se.x) conv(x);
        for (auto& b : o.bands) { conv(b.x0); conv(b.x1); }
        if (x_label) *x_label = QString("Time [%1]").arg(unit);
    } else if (x_label && d.x_samples) {
        *x_label = "Sample index";
    }
    if (u.physical_y && d.y_scale > 0) {
        const double maxabs = std::max(std::fabs(d.y0), std::fabs(d.y1)) * d.y_scale;
        const auto [f, unit] = siPrefix(maxabs, d.y_unit.isEmpty() ? QString("V") : d.y_unit);
        const double k = d.y_scale * f;
        auto conv = [&](auto& v) { v = static_cast<std::remove_reference_t<decltype(v)>>(v * k); };
        conv(o.y0); conv(o.y1);
        for (auto& se : o.series) {
            for (float& v : se.y) conv(v);
            for (float& v : se.ymin) conv(v);
            for (float& v : se.ymax) conv(v);
        }
        const QString what = (d.y_unit.isEmpty() || d.y_unit == "V") ? QString("Voltage") : QString("Value");
        if (y_label) *y_label = QString("%1 [%2]").arg(what, unit);
    }
    return o;
}
