#include "plot_widget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

static constexpr int64_t RENDER_CHUNK    = 1 << 20;  // 1 M samples = 4 MB
static constexpr int64_t STRIDE_THRESHOLD = 8;
static constexpr int64_t YRANGE_SAMPLES  = 2048;

// ---------------------------------------------------------------------------
// PlotTheme built-ins
// ---------------------------------------------------------------------------
PlotTheme PlotTheme::dark() {
    return {
        QColor(22,  22,  28),   // bg_outer
        QColor(10,  10,  15),   // bg_plot
        QColor(45,  45,  55),   // grid
        QColor(80,  80, 100),   // border
        QColor(180, 180, 200),  // axis_text
        QColor(200, 200, 220),  // legend_text
    };
}

PlotTheme PlotTheme::light() {
    return {
        QColor(220, 220, 228),  // bg_outer
        QColor(255, 255, 255),  // bg_plot
        QColor(210, 210, 218),  // grid
        QColor(140, 140, 160),  // border
        QColor( 50,  50,  70),  // axis_text
        QColor( 30,  30,  50),  // legend_text
    };
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
PlotWidget::PlotWidget(QWidget* parent)
    : QWidget(parent)
{
    setMouseTracking(true);
    setMinimumSize(400, 200);
    setTheme(PlotTheme::dark());
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void PlotWidget::setTheme(const PlotTheme& theme) {
    theme_ = theme;
    QPalette pal = palette();
    pal.setColor(QPalette::Window, theme_.bg_outer);
    setPalette(pal);
    setAutoFillBackground(true);
    update();
}

void PlotWidget::setMode(InteractionMode mode) {
    mode_ = mode;
    if (mode_ == InteractionMode::Measure ||
        mode_ == InteractionMode::BoxZoom  ||
        mode_ == InteractionMode::CropSelect)
        setCursor(Qt::CrossCursor);
    else
        setCursor(Qt::ArrowCursor);
    dragging_           = false;
    rubber_band_active_ = false;
    update();
}

void PlotWidget::addCropRange(int64_t start, int64_t end) {
    if (end <= start) return;
    start = std::clamp(start, INT64_C(0), total_samples_);
    end   = std::clamp(end,   INT64_C(0), total_samples_);
    if (end <= start) return;
    crop_ranges_.push_back({start, end});
    emit cropRangesChanged();
    update();
}

void PlotWidget::removeCropRangeAt(int idx) {
    if (idx < 0 || idx >= static_cast<int>(crop_ranges_.size())) return;
    crop_ranges_.erase(crop_ranges_.begin() + idx);
    emit cropRangesChanged();
    update();
}

void PlotWidget::clearCropRanges() {
    if (crop_ranges_.empty()) return;
    crop_ranges_.clear();
    emit cropRangesChanged();
    update();
}

void PlotWidget::setThresholds(bool show, double pos, double neg) {
    show_thresholds_ = show;
    threshold_pos_   = pos;
    threshold_neg_   = neg;
    update();
}

void PlotWidget::addTrace(TrsFile* file, int32_t trace_idx,
                           QColor color, const QString& label)
{
    TraceEntry te;
    te.file       = file;
    te.trace_idx  = trace_idx;
    te.color      = color;
    te.label      = label.isEmpty() ? QString("Trace %1").arg(trace_idx) : label;
    te.transforms = transforms_;
    traces_.push_back(std::move(te));

    if (file && total_samples_ == 0) {
        total_samples_ = file->header().num_samples;
        view_start_    = 0;
        view_end_      = total_samples_;
    }
    update();
}

void PlotWidget::addTrace(std::shared_ptr<std::vector<float>> data,
                           QColor color, const QString& label)
{
    TraceEntry te;
    te.mem_data  = std::move(data);
    te.color     = color;
    te.label     = label.isEmpty() ? QString("T-statistic") : label;
    te.transforms = transforms_;
    traces_.push_back(std::move(te));

    if (total_samples_ == 0 && !traces_.back().mem_data->empty()) {
        total_samples_ = static_cast<int64_t>(traces_.back().mem_data->size());
        view_start_    = 0;
        view_end_      = total_samples_;
    }
    update();
}

void PlotWidget::clearTraces() {
    traces_.clear();
    total_samples_ = view_start_ = view_end_ = 0;
    meas_count_ = 0;
    update();
}

void PlotWidget::setTransforms(const std::vector<std::shared_ptr<ITransform>>& tx) {
    transforms_ = tx;
    for (auto& te : traces_) te.transforms = tx;
    update();
}

void PlotWidget::resetView() {
    int64_t startup = 0;
    for (const auto& t : transforms_)
        startup = std::max(startup, t->startupSamples());
    view_start_ = std::min(startup, total_samples_);
    view_end_   = total_samples_;
    clearMeasurement();
    emit viewChanged(view_start_, view_end_, total_samples_);
    update();
}

void PlotWidget::clearMeasurement() {
    meas_count_ = 0;
    emit measurementUpdated(0, 0, 0, 0, false);
    update();
}

void PlotWidget::zoomIn() {
    int64_t c = (view_start_ + view_end_) / 2;
    int64_t h = std::max(INT64_C(5), (view_end_ - view_start_) / 4);
    view_start_ = std::max(INT64_C(0), c - h);
    view_end_   = std::min(total_samples_, c + h);
    emit viewChanged(view_start_, view_end_, total_samples_);
    update();
}

void PlotWidget::zoomOut() {
    int64_t c = (view_start_ + view_end_) / 2;
    int64_t h = view_end_ - view_start_;
    view_start_ = std::max(INT64_C(0), c - h);
    view_end_   = std::min(total_samples_, c + h);
    emit viewChanged(view_start_, view_end_, total_samples_);
    update();
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

QRect PlotWidget::plotRect() const {
    return QRect(ML, MT, width() - ML - MR, height() - MT - MB);
}

int64_t PlotWidget::pixelToSample(int px, const QRect& pr) const {
    int64_t len = view_end_ - view_start_;
    return view_start_ + static_cast<int64_t>(px - pr.left()) * len / pr.width();
}

int PlotWidget::sampleToPixel(int64_t s, const QRect& pr) const {
    int64_t len = view_end_ - view_start_;
    if (len <= 0) return pr.left();
    return pr.left() + static_cast<int>((s - view_start_) * pr.width() / len);
}

int PlotWidget::valueToPixel(float v, const QRect& pr, float ymin, float ymax) const {
    if (ymax == ymin) return pr.center().y();
    float t = (v - ymin) / (ymax - ymin);
    return pr.bottom() - static_cast<int>(t * pr.height());
}

double PlotWidget::pixelToValue(int py, const QRect& pr, float ymin, float ymax) const {
    if (pr.height() <= 0) return 0.0;
    double t = 1.0 - static_cast<double>(py - pr.top()) / pr.height();
    return ymin + t * (ymax - ymin);
}

// ---------------------------------------------------------------------------
// readTraceSamples helper
// ---------------------------------------------------------------------------

int64_t PlotWidget::readTraceSamples(const TraceEntry& te,
                                      int64_t sample_offset, int64_t count,
                                      float* buf) const {
    if (te.mem_data) {
        const auto& v = *te.mem_data;
        if (sample_offset >= static_cast<int64_t>(v.size())) return 0;
        int64_t avail = static_cast<int64_t>(v.size()) - sample_offset;
        int64_t n = std::min(count, avail);
        if (n > 0) std::memcpy(buf, v.data() + static_cast<size_t>(sample_offset),
                               static_cast<size_t>(n) * sizeof(float));
        return n;
    }
    if (!te.file) return 0;
    return te.file->readSamples(te.trace_idx, sample_offset, count, buf);
}

// ---------------------------------------------------------------------------
// Y-range estimation
// ---------------------------------------------------------------------------

void PlotWidget::computeYRange(float& ymin, float& ymax) {
    ymin = std::numeric_limits<float>::max();
    ymax = std::numeric_limits<float>::lowest();

    int64_t view_len = view_end_ - view_start_;
    if (view_len <= 0) { ymin = -1; ymax = 1; return; }

    // The render stride path samples at view_len/(4*W). Ensure the Y-range
    // estimate is never coarser, so peaks the renderer draws are not outside
    // the computed range (which would cause visual clipping on the plot border).
    int64_t W = std::max(1, plotRect().width());
    int64_t n_points = std::max(static_cast<int64_t>(YRANGE_SAMPLES), W * INT64_C(4));
    int64_t stride = std::max(INT64_C(1), view_len / n_points);

    for (const auto& te : traces_) {
        if (!te.visible || (!te.file && !te.mem_data)) continue;
        float v;
        for (int64_t s = view_start_; s < view_end_; s += stride) {
            readTraceSamples(te, s, 1, &v);
            for (const auto& t : te.transforms)
                if (!t->requiresSequential())
                    t->apply(&v, 1, s);
            ymin = std::min(ymin, v);
            ymax = std::max(ymax, v);
        }
    }

    if (ymin == std::numeric_limits<float>::max()) { ymin = -1; ymax = 1; return; }
    if (ymin == ymax) { ymin -= 1; ymax += 1; return; }

    float pad = (ymax - ymin) * 0.05f;
    ymin -= pad;
    ymax += pad;
}

// ---------------------------------------------------------------------------
// Trace rendering
// ---------------------------------------------------------------------------

void PlotWidget::renderTrace(TraceEntry& te, QPainter& p,
                              const QRect& pr, float ymin, float ymax)
{
    if (!te.visible || (!te.file && !te.mem_data)) return;
    int W = pr.width();
    if (W <= 0) return;

    int64_t view_len = view_end_ - view_start_;
    if (view_len <= 0) return;

    std::vector<float> pixMin(W,  std::numeric_limits<float>::max());
    std::vector<float> pixMax(W,  std::numeric_limits<float>::lowest());

    bool has_sequential = false;
    for (const auto& t : te.transforms)
        if (t->requiresSequential()) { has_sequential = true; break; }

    for (auto& t : te.transforms) t->reset();

    int64_t spp = (view_len + W - 1) / W;

    // ------------------------------------------------------------------
    // ZOOMED-IN PATH: fewer samples than pixels — draw connected polyline.
    // At most `view_len` samples are read (≤ widget width), so memory
    // stays tiny. Dots are drawn when there is enough space per sample.
    // ------------------------------------------------------------------
    if (view_len <= W) {
        std::vector<float> buf(view_len);
        int64_t raw_read = readTraceSamples(te, view_start_, view_len, buf.data());
        if (raw_read <= 0) return;
        int64_t out_count = raw_read;
        for (auto& t : te.transforms)
            out_count = t->apply(buf.data(), out_count, view_start_);
        if (out_count <= 0) return;

        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(te.color, 1.5));

        // Map output sample i to x pixel via its position in raw coordinates.
        // Output sample i covers raw[i * raw_read/out_count .. (i+1) * raw_read/out_count).
        auto sx = [&](int64_t i) -> int {
            int64_t raw_s = (out_count > 1)
                ? view_start_ + i * raw_read / out_count
                : view_start_;
            return pr.left() + static_cast<int>(
                static_cast<double>(raw_s - view_start_) * W / view_len);
        };

        // Line segments
        for (int64_t i = 1; i < out_count; i++)
            p.drawLine(sx(i - 1), valueToPixel(buf[i-1], pr, ymin, ymax),
                       sx(i),     valueToPixel(buf[i],   pr, ymin, ymax));

        // Sample dots when >= 4 pixels per output sample
        if (out_count > 0 && W / out_count >= 4) {
            p.setBrush(te.color);
            p.setPen(Qt::NoPen);
            for (int64_t i = 0; i < out_count; i++)
                p.drawEllipse(QPoint(sx(i), valueToPixel(buf[i], pr, ymin, ymax)), 3, 3);
            p.setBrush(Qt::NoBrush);
        }

        p.setRenderHint(QPainter::Antialiasing, false);
        return;
    }

    if (!has_sequential && spp > STRIDE_THRESHOLD) {
        int64_t stride = std::max(INT64_C(1), spp / 4);
        float   sbuf;
        for (int64_t s = view_start_; s < view_end_; s += stride) {
            readTraceSamples(te, s, 1, &sbuf);
            for (auto& t : te.transforms) t->apply(&sbuf, 1, s);
            int px = static_cast<int>((s - view_start_) * W / view_len);
            if (px >= 0 && px < W) {
                pixMin[px] = std::min(pixMin[px], sbuf);
                pixMax[px] = std::max(pixMax[px], sbuf);
            }
        }
    } else {
        std::vector<float> buf;
        buf.reserve(RENDER_CHUNK);
        int64_t chunk_start = view_start_;
        while (chunk_start < view_end_) {
            int64_t chunk_end = std::min(chunk_start + RENDER_CHUNK, view_end_);
            int64_t chunk_len = chunk_end - chunk_start;
            buf.resize(chunk_len);
            int64_t raw_read = readTraceSamples(te, chunk_start, chunk_len, buf.data());
            if (raw_read <= 0) break;
            int64_t out_count = raw_read;
            for (auto& t : te.transforms)
                out_count = t->apply(buf.data(), out_count, chunk_start);
            // Map each output sample back to a raw-coordinate pixel.
            // Output sample i covers raw[chunk_start + i*raw_read/out_count .. ].
            for (int64_t i = 0; i < out_count; i++) {
                int64_t s = (out_count > 0)
                    ? chunk_start + i * raw_read / out_count
                    : chunk_start;
                int px = static_cast<int>((s - view_start_) * W / view_len);
                if (px >= 0 && px < W) {
                    pixMin[px] = std::min(pixMin[px], buf[i]);
                    pixMax[px] = std::max(pixMax[px], buf[i]);
                }
            }
            chunk_start = chunk_end;
        }
    }

    // Draw per-pixel min-max segments with gap bridging.
    // valueToPixel maps larger values to smaller y (top of screen), so:
    //   bot = valueToPixel(pixMin) — large y, bottom of bar on screen
    //   top = valueToPixel(pixMax) — small y, top of bar on screen
    // When the signal jumps between consecutive columns the bars may not
    // overlap; we extend the current bar to close the gap so the trace
    // appears continuous.
    p.setPen(QPen(te.color, 1));
    int prev_top = INT_MAX, prev_bot = INT_MAX;
    for (int px = 0; px < W; px++) {
        if (pixMin[px] == std::numeric_limits<float>::max()) {
            prev_top = prev_bot = INT_MAX;
            continue;
        }
        int top = valueToPixel(pixMax[px], pr, ymin, ymax);
        int bot = valueToPixel(pixMin[px], pr, ymin, ymax);

        if (prev_top != INT_MAX) {
            // Signal jumped up  → current bar is above the previous bar → extend down
            if (bot < prev_top)  bot = prev_top;
            // Signal jumped down → current bar is below the previous bar → extend up
            else if (top > prev_bot) top = prev_bot;
        }
        // Keep original unextended values for next iteration
        prev_top = valueToPixel(pixMax[px], pr, ymin, ymax);
        prev_bot = valueToPixel(pixMin[px], pr, ymin, ymax);

        if (top > bot) std::swap(top, bot);
        if (top == bot) p.drawPoint(pr.left() + px, top);
        else            p.drawLine(pr.left() + px, top, pr.left() + px, bot);
    }
}

// ---------------------------------------------------------------------------
// Measurement overlay
// ---------------------------------------------------------------------------

void PlotWidget::drawMeasurement(QPainter& p, const QRect& pr,
                                  float ymin, float ymax)
{
    if (meas_count_ == 0) return;

    // Colors: P1 = yellow, P2 = cyan-green
    const QColor col1(255, 200,   0);
    const QColor col2(  0, 220, 140);

    auto drawMarker = [&](int64_t s, double v, QColor col, const QString& tag) {
        int x = sampleToPixel(s, pr);
        int y = valueToPixel(static_cast<float>(v), pr, ymin, ymax);
        y = std::clamp(y, pr.top(), pr.bottom());

        // Vertical dashed line
        QPen dpen(col, 1, Qt::DashLine);
        p.setPen(dpen);
        p.drawLine(x, pr.top(), x, pr.bottom());

        // Horizontal tick at the value
        p.setPen(QPen(col, 2));
        p.drawLine(x - 6, y, x + 6, y);

        // Label
        p.setPen(col);
        QFont f = font(); f.setPointSize(8); f.setBold(true); p.setFont(f);
        p.drawText(x + 4, pr.top() + 14, tag);
    };

    drawMarker(meas_s1_, meas_v1_, col1, "P1");

    if (meas_count_ == 2) {
        drawMarker(meas_s2_, meas_v2_, col2, "P2");

        // Horizontal distance line between the two marker x positions
        int x1 = sampleToPixel(meas_s1_, pr);
        int x2 = sampleToPixel(meas_s2_, pr);
        int ym = pr.top() + 24;
        p.setPen(QPen(QColor(180, 180, 60), 1, Qt::DotLine));
        p.drawLine(x1, ym, x2, ym);
    }
}

// ---------------------------------------------------------------------------
// paintEvent
// ---------------------------------------------------------------------------

void PlotWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);

    QRect pr = plotRect();

    p.fillRect(rect(), theme_.bg_outer);
    p.fillRect(pr,     theme_.bg_plot);

    if (traces_.empty() || view_end_ <= view_start_) {
        QColor tc = theme_.axis_text; tc.setAlpha(160);
        p.setPen(tc);
        p.drawText(pr, Qt::AlignCenter, "No traces loaded.\nUse File › Open to load a .trs file.");
        return;
    }

    float ymin, ymax;
    computeYRange(ymin, ymax);
    last_ymin_ = ymin;
    last_ymax_ = ymax;

    // Grid
    p.setPen(QPen(theme_.grid, 1));
    for (int i = 1; i < 5; i++) {
        int y = pr.top() + pr.height() * i / 5;
        p.drawLine(pr.left(), y, pr.right(), y);
    }
    for (int i = 1; i < 10; i++) {
        int x = pr.left() + pr.width() * i / 10;
        p.drawLine(x, pr.top(), x, pr.bottom());
    }

    // Traces
    for (auto& te : traces_)
        renderTrace(te, p, pr, ymin, ymax);

    // Threshold lines
    if (show_thresholds_) {
        const QColor thr_color(255, 160, 0, 220);  // orange
        QPen thr_pen(thr_color, 1, Qt::DashLine);
        p.setPen(thr_pen);
        QFont tf = font(); tf.setPointSize(8); tf.setBold(true); p.setFont(tf);
        for (double tv : {threshold_pos_, threshold_neg_}) {
            int ty = valueToPixel(static_cast<float>(tv), pr, ymin, ymax);
            if (ty >= pr.top() && ty <= pr.bottom()) {
                p.setPen(thr_pen);
                p.drawLine(pr.left(), ty, pr.right(), ty);
                p.setPen(thr_color);
                p.drawText(pr.left() + 4, ty - 3,
                           QString("%1").arg(tv, 0, 'f', 1));
            }
        }
    }

    // Crop range overlays (drawn on top of traces)
    if (!crop_ranges_.empty()) {
        // Two alternating green hues for adjacent ranges
        const QColor fill_colors[2] = { QColor(60, 200, 80, 45),  QColor(60, 200, 160, 45)  };
        const QColor line_colors[2] = { QColor(80, 220, 80, 200), QColor(80, 220, 160, 200) };
        QFont cf = font(); cf.setPointSize(8); cf.setBold(true);
        for (int i = 0; i < static_cast<int>(crop_ranges_.size()); i++) {
            int x1 = sampleToPixel(crop_ranges_[i].first,  pr);
            int x2 = sampleToPixel(crop_ranges_[i].second, pr);
            x1 = std::clamp(x1, pr.left(), pr.right());
            x2 = std::clamp(x2, pr.left(), pr.right());
            if (x1 >= x2) continue;
            const QColor& fc = fill_colors[i & 1];
            const QColor& lc = line_colors[i & 1];
            p.fillRect(QRect(x1, pr.top(), x2 - x1, pr.height()), fc);
            p.setPen(QPen(lc, 1));
            p.drawLine(x1, pr.top(), x1, pr.bottom());
            p.drawLine(x2, pr.top(), x2, pr.bottom());
            p.setFont(cf);
            p.setPen(lc);
            p.drawText(x1 + 3, pr.top() + 13, QString("#%1").arg(i + 1));
        }
    }

    // Measurement overlay (drawn on top of traces)
    drawMeasurement(p, pr, ymin, ymax);

    // Border
    p.setPen(theme_.border);
    p.drawRect(pr);

    // X-axis labels
    p.setPen(theme_.axis_text);
    QFont f = font(); f.setPointSize(8); p.setFont(f);

    for (int i = 0; i <= 6; i++) {
        int64_t s = view_start_ + (view_end_ - view_start_) * i / 6;
        int     x = pr.left() + pr.width() * i / 6;
        p.drawLine(x, pr.bottom(), x, pr.bottom() + 4);

        QString lbl;
        if      (s >= 1'000'000) lbl = QString("%1M").arg(s / 1'000'000.0, 0, 'f', 2);
        else if (s >= 1'000)     lbl = QString("%1k").arg(s / 1'000.0,     0, 'f', 1);
        else                     lbl = QString::number(s);
        p.drawText(x - 28, pr.bottom() + 6, 56, 20, Qt::AlignCenter, lbl);
    }

    // Y-axis labels
    for (int i = 0; i <= 5; i++) {
        float v  = ymin + (ymax - ymin) * i / 5;
        int   py = pr.bottom() - pr.height() * i / 5;
        p.drawLine(pr.left() - 4, py, pr.left(), py);
        QString lbl = QString::number(static_cast<double>(v), 'g', 4);
        p.drawText(0, py - 9, ML - 6, 18, Qt::AlignRight | Qt::AlignVCenter, lbl);
    }

    // Legend
    {
        int lx = pr.right() - 6;
        int ly = pr.top() + 8;
        for (const auto& te : traces_) {
            if (!te.visible) continue;
            p.setPen(te.color);
            p.drawLine(lx - 18, ly, lx, ly);
            p.setPen(theme_.legend_text);
            p.drawText(lx - 130, ly - 8, 108, 16, Qt::AlignRight, te.label);
            ly += 18;
        }
    }

    // BoxZoom / CropSelect rubber-band overlay
    if (rubber_band_active_) {
        int x1 = std::clamp(rubber_band_start_.x(),   pr.left(), pr.right());
        int x2 = std::clamp(rubber_band_current_.x(), pr.left(), pr.right());
        if (x1 > x2) std::swap(x1, x2);
        QRect band(x1, pr.top(), x2 - x1, pr.height());

        bool is_crop = (mode_ == InteractionMode::CropSelect);
        QColor fill_c = is_crop ? QColor( 60, 200,  80, 50) : QColor(100, 160, 255, 50);
        QColor edge_c = is_crop ? QColor( 80, 220,  80, 200) : QColor(120, 180, 255, 200);

        p.fillRect(band, fill_c);
        p.setPen(QPen(edge_c, 1, Qt::DashLine));
        p.drawRect(band);
        p.setPen(QPen(edge_c, 1));
        p.drawLine(x1, pr.top(), x1, pr.bottom());
        p.drawLine(x2, pr.top(), x2, pr.bottom());
    }
}

// ---------------------------------------------------------------------------
// Mouse / wheel interaction
// ---------------------------------------------------------------------------

void PlotWidget::mousePressEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) return;

    QRect pr = plotRect();

    if (mode_ == InteractionMode::Measure) {
        if (!pr.contains(e->pos())) return;
        int64_t s = pixelToSample(e->pos().x(), pr);
        double  v = pixelToValue(e->pos().y(), pr, last_ymin_, last_ymax_);
        s = std::clamp(s, view_start_, view_end_);

        if (meas_count_ < 2) {
            if (meas_count_ == 0) { meas_s1_ = s; meas_v1_ = v; meas_count_ = 1; }
            else                  { meas_s2_ = s; meas_v2_ = v; meas_count_ = 2; }
        } else {
            // Third click: start a new pair
            meas_s1_ = s; meas_v1_ = v; meas_count_ = 1;
        }

        emit measurementUpdated(meas_s1_, meas_v1_, meas_s2_, meas_v2_,
                                 meas_count_ == 2);
        update();
        return;
    }

    if (mode_ == InteractionMode::BoxZoom ||
        mode_ == InteractionMode::CropSelect) {
        if (!pr.contains(e->pos())) return;
        rubber_band_active_  = true;
        rubber_band_start_   = e->pos();
        rubber_band_current_ = e->pos();
        update();
        return;
    }

    // Pan mode
    dragging_            = true;
    drag_origin_         = e->pos();
    drag_start_at_press_ = view_start_;
    drag_end_at_press_   = view_end_;
    setCursor(Qt::ClosedHandCursor);
}

void PlotWidget::mouseMoveEvent(QMouseEvent* e) {
    if (mode_ == InteractionMode::BoxZoom ||
        mode_ == InteractionMode::CropSelect) {
        if (rubber_band_active_) {
            rubber_band_current_ = e->pos();
            update();
        }
        return;
    }

    if (!dragging_ || mode_ != InteractionMode::Pan) return;

    QRect   pr    = plotRect();
    int64_t vlen  = drag_end_at_press_ - drag_start_at_press_;
    int     dx    = e->pos().x() - drag_origin_.x();
    int64_t delta = -static_cast<int64_t>(dx) * vlen / pr.width();

    view_start_ = drag_start_at_press_ + delta;
    view_end_   = drag_end_at_press_   + delta;

    if (view_start_ < 0) { view_end_ -= view_start_; view_start_ = 0; }
    if (view_end_ > total_samples_) {
        view_start_ -= (view_end_ - total_samples_);
        view_end_    = total_samples_;
    }
    view_start_ = std::max(INT64_C(0), view_start_);

    emit viewChanged(view_start_, view_end_, total_samples_);
    update();
}

void PlotWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() != Qt::LeftButton) return;

    if ((mode_ == InteractionMode::BoxZoom ||
         mode_ == InteractionMode::CropSelect) && rubber_band_active_) {
        rubber_band_active_ = false;

        QRect pr = plotRect();
        int x1 = std::clamp(rubber_band_start_.x(),   pr.left(), pr.right());
        int x2 = std::clamp(rubber_band_current_.x(), pr.left(), pr.right());
        if (x1 > x2) std::swap(x1, x2);

        // Require a minimum drag width (5 px) to avoid accidental clicks
        if (x2 - x1 >= 5) {
            int64_t s1 = pixelToSample(x1, pr);
            int64_t s2 = pixelToSample(x2, pr);
            s1 = std::clamp(s1, INT64_C(0), total_samples_);
            s2 = std::clamp(s2, INT64_C(0), total_samples_);
            if (s2 > s1) {
                if (mode_ == InteractionMode::BoxZoom) {
                    view_start_ = s1;
                    view_end_   = s2;
                    emit viewChanged(view_start_, view_end_, total_samples_);
                } else {
                    // CropSelect: append to crop list
                    crop_ranges_.push_back({s1, s2});
                    emit cropRangesChanged();
                }
            }
        }
        update();
        return;
    }

    if (dragging_) {
        dragging_ = false;
        setCursor(mode_ == InteractionMode::BoxZoom ? Qt::CrossCursor : Qt::ArrowCursor);
    }
}

void PlotWidget::wheelEvent(QWheelEvent* e) {
    QRect   pr   = plotRect();
    int64_t vlen = view_end_ - view_start_;

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    double frac = (e->position().x() - pr.left()) / pr.width();
#else
    double frac = (e->posF().x() - pr.left()) / pr.width();
#endif
    frac = std::clamp(frac, 0.0, 1.0);

    int64_t pivot   = view_start_ + static_cast<int64_t>(frac * vlen);
    double  factor  = (e->angleDelta().y() > 0) ? 0.65 : 1.0 / 0.65;
    int64_t new_len = std::max(INT64_C(10), static_cast<int64_t>(vlen * factor));

    view_start_ = pivot - static_cast<int64_t>(frac * new_len);
    view_end_   = view_start_ + new_len;

    if (view_start_ < 0) { view_end_ -= view_start_; view_start_ = 0; }
    if (view_end_ > total_samples_) {
        view_start_ -= (view_end_ - total_samples_);
        view_end_    = total_samples_;
    }
    view_start_ = std::max(INT64_C(0), view_start_);
    view_end_   = std::min(total_samples_, view_end_);

    emit viewChanged(view_start_, view_end_, total_samples_);
    update();
}

void PlotWidget::resizeEvent(QResizeEvent*) {
    update();
}
