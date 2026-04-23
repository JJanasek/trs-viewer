#include "plot_widget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

static constexpr int64_t RENDER_CHUNK = 1 << 20;  // 1 M samples = 4 MB

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

// ---------------------------------------------------------------------------
// Per-trace shifts
// ---------------------------------------------------------------------------

int32_t PlotWidget::traceShift(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(traces_.size())) return 0;
    return traces_[static_cast<size_t>(idx)].shift;
}

std::vector<int32_t> PlotWidget::traceShifts() const {
    std::vector<int32_t> out;
    out.reserve(traces_.size());
    for (const auto& te : traces_) out.push_back(te.shift);
    return out;
}

void PlotWidget::setTraceShift(int idx, int32_t shift) {
    if (idx < 0 || idx >= static_cast<int>(traces_.size())) return;
    auto& te = traces_[static_cast<size_t>(idx)];
    if (te.shift == shift) return;
    te.shift       = shift;
    te.cache.valid = false;
    emit traceShiftsChanged();
    update();
}

void PlotWidget::clearTraceShifts() {
    bool changed = false;
    for (auto& te : traces_) {
        if (te.shift != 0) { te.shift = 0; te.cache.valid = false; changed = true; }
    }
    if (changed) { emit traceShiftsChanged(); update(); }
}

// ---------------------------------------------------------------------------
// AlignDrag helpers
// ---------------------------------------------------------------------------

double PlotWidget::traceValueAt(const TraceEntry& te, int64_t s) const {
    if (!te.cache.valid) return std::numeric_limits<double>::quiet_NaN();
    if (!te.cache.samples.empty()) {
        int64_t out_count = static_cast<int64_t>(te.cache.samples.size());
        int64_t raw_read  = te.cache.raw_read;
        if (raw_read <= 0 || out_count <= 0) return std::numeric_limits<double>::quiet_NaN();
        int64_t i = (s - te.cache.view_start) * out_count / raw_read;
        i = std::clamp(i, INT64_C(0), out_count - 1);
        return te.cache.samples[static_cast<size_t>(i)];
    }
    if (!te.cache.pix_min.empty()) {
        int64_t view_len = te.cache.view_end - te.cache.view_start;
        int     W        = te.cache.W;
        if (view_len <= 0 || W <= 0) return std::numeric_limits<double>::quiet_NaN();
        int px = static_cast<int>((s - te.cache.view_start) * W / view_len);
        px = std::clamp(px, 0, W - 1);
        float mn = te.cache.pix_min[static_cast<size_t>(px)];
        if (mn == std::numeric_limits<float>::max()) return std::numeric_limits<double>::quiet_NaN();
        return (mn + te.cache.pix_max[static_cast<size_t>(px)]) * 0.5;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

int PlotWidget::nearestTrace(int px, int py, const QRect& pr) const {
    if (traces_.empty()) return -1;
    int64_t s       = pixelToSample(px, pr);
    double  v_click = pixelToValue(py, pr, last_ymin_, last_ymax_);
    int     best    = -1;
    double  best_d  = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(traces_.size()); i++) {
        double v = traceValueAt(traces_[static_cast<size_t>(i)], s);
        if (std::isnan(v)) continue;
        double d = std::abs(v - v_click);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

void PlotWidget::setThresholds(bool show, double pos, double neg) {
    show_thresholds_ = show;
    threshold_pos_   = pos;
    threshold_neg_   = neg;
    update();
}

void PlotWidget::setThresholdOneSided(bool one_sided) {
    threshold_one_sided_ = one_sided;
    update();
}

void PlotWidget::setTitle(const QString& title) {
    plot_title_ = title;
    update();
}

void PlotWidget::setTraceWidth(float w) {
    trace_width_ = std::clamp(w, 0.5f, 8.0f);
    update();
}

void PlotWidget::setTraceColor(int idx, const QColor& c) {
    if (idx < 0 || idx >= static_cast<int>(traces_.size())) return;
    traces_[static_cast<size_t>(idx)].color = c;
    update();
}

void PlotWidget::setTraceFilled(int idx, bool filled) {
    if (idx < 0 || idx >= static_cast<int>(traces_.size())) return;
    traces_[static_cast<size_t>(idx)].filled = filled;
    update();
}

void PlotWidget::setAxisLabels(const QString& x_label, const QString& y_label) {
    axis_label_x_ = x_label;
    axis_label_y_ = y_label;
    // invalidate caches — plot rect size changed
    for (auto& te : traces_) te.cache.valid = false;
    update();
}

void PlotWidget::setXScale(double scale, double offset) {
    x_label_scale_  = (scale != 0.0) ? scale : 1.0;
    x_label_offset_ = offset;
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
    for (const auto& t : transforms_) te.transforms.push_back(t->clone());
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
    for (const auto& t : transforms_) te.transforms.push_back(t->clone());
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
    sticky_ymin_ =  std::numeric_limits<float>::infinity();
    sticky_ymax_ = -std::numeric_limits<float>::infinity();
    update();
}

void PlotWidget::setTransforms(const std::vector<std::shared_ptr<ITransform>>& tx) {
    transforms_ = tx;
    for (auto& te : traces_) {
        te.transforms.clear();
        for (const auto& t : tx) te.transforms.push_back(t->clone());
        te.cache.valid = false;
    }
    update();
}

void PlotWidget::resetView() {
    int64_t startup = 0;
    for (const auto& t : transforms_)
        startup = std::max(startup, t->startupSamples());
    view_start_ = std::min(startup, total_samples_);
    view_end_   = total_samples_;
    y_scale_    = 1.0f;
    sticky_ymin_ =  std::numeric_limits<float>::infinity();
    sticky_ymax_ = -std::numeric_limits<float>::infinity();
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

void PlotWidget::zoomInY() {
    y_scale_ = std::clamp(y_scale_ * 0.75f, 0.05f, 200.0f);
    update();
}

void PlotWidget::zoomOutY() {
    y_scale_ = std::clamp(y_scale_ / 0.75f, 0.05f, 200.0f);
    update();
}

void PlotWidget::resetYZoom() {
    y_scale_ = 1.0f;
    sticky_ymin_ =  std::numeric_limits<float>::infinity();
    sticky_ymax_ = -std::numeric_limits<float>::infinity();
    update();
}

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

QRect PlotWidget::plotRect() const {
    int mt = plot_title_.isEmpty() ? MT : MT_TITLE;
    int ml = axis_label_y_.isEmpty() ? ML : ML_YLBL;
    int mb = axis_label_x_.isEmpty() ? MB : MB_XLBL;
    return QRect(ml, mt, width() - ml - MR, height() - mt - mb);
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
    // Apply per-trace shift: positive shift → read from later raw samples
    //   (trace content moves left); negative → read from earlier (moves right).
    const int64_t adj = sample_offset + te.shift;

    // Zero-fill the output first so padding regions are silent.
    std::fill(buf, buf + static_cast<size_t>(count), 0.0f);

    if (count <= 0) return count;

    // Clamp the readable window into [0, total)
    int64_t total = 0;
    if (te.mem_data)
        total = static_cast<int64_t>(te.mem_data->size());
    else if (te.file)
        total = te.file->header().num_samples;
    else
        return count;   // all zeros

    int64_t src_start = std::max<int64_t>(0, adj);
    int64_t src_end   = std::min<int64_t>(total, adj + count);
    if (src_start >= src_end) return count; // entirely outside → all zeros

    int64_t dst_off = src_start - adj;      // leading zeros already filled
    int64_t n       = src_end - src_start;

    if (te.mem_data) {
        const auto& v = *te.mem_data;
        std::memcpy(buf + static_cast<size_t>(dst_off),
                    v.data() + static_cast<size_t>(src_start),
                    static_cast<size_t>(n) * sizeof(float));
    } else {
        te.file->readSamples(te.trace_idx, src_start, n,
                             buf + static_cast<size_t>(dst_off));
    }
    return count;   // always return the requested count (zeros pad the rest)
}

// ---------------------------------------------------------------------------
// Trace cache builder — called once per visible trace per frame.
// Re-uses the existing cache if (view_start, view_end, W) have not changed.
// ---------------------------------------------------------------------------

void PlotWidget::buildTraceCache(TraceEntry& te, int W)
{
    if (!te.visible || (!te.file && !te.mem_data)) return;

    if (te.cache.valid &&
        te.cache.view_start == view_start_ &&
        te.cache.view_end   == view_end_   &&
        te.cache.W          == W)
        return;   // cache is fresh

    const int64_t view_len = view_end_ - view_start_;
    if (view_len <= 0) { te.cache.valid = false; return; }

    te.cache.view_start = view_start_;
    te.cache.view_end   = view_end_;
    te.cache.W          = W;
    te.cache.valid      = true;
    te.cache.ymin       =  std::numeric_limits<float>::max();
    te.cache.ymax       =  std::numeric_limits<float>::lowest();

    for (auto& t : te.transforms) t->reset();

    // Expected output sample count after the full pipeline.
    int64_t expected_out = view_len;
    for (const auto& t : te.transforms)
        expected_out = t->transformedCount(expected_out);

    // ------------------------------------------------------------------
    // ZOOMED-IN: read all view samples, apply full pipeline, keep array.
    // Use output count (not raw view_len) so decimating pipelines still
    // get the connected-polyline path when their output fits in the widget.
    // ------------------------------------------------------------------
    if (expected_out <= W) {
        te.cache.pix_min.clear();
        te.cache.pix_max.clear();
        // Size the buffer for both the raw read AND the post-pipeline output —
        // expanding transforms (e.g. STFT) can produce more samples than view_len.
        te.cache.samples.resize(static_cast<size_t>(std::max(view_len, expected_out)));
        int64_t raw_read  = readTraceSamples(te, view_start_, view_len,
                                             te.cache.samples.data());
        if (raw_read <= 0) { te.cache.valid = false; return; }
        int64_t out_count = raw_read;
        for (auto& t : te.transforms)
            out_count = t->apply(te.cache.samples.data(), out_count, view_start_);
        te.cache.samples.resize(static_cast<size_t>(std::max(INT64_C(0), out_count)));
        te.cache.raw_read = raw_read;

        for (float v : te.cache.samples) {
            te.cache.ymin = std::min(te.cache.ymin, v);
            te.cache.ymax = std::max(te.cache.ymax, v);
        }
        return;
    }

    // ------------------------------------------------------------------
    // ZOOMED-OUT: chunk-based processing → per-pixel min/max aggregates.
    // Never calls apply() on a single float — always passes full chunks.
    // ------------------------------------------------------------------
    te.cache.samples.clear();
    te.cache.raw_read = 0;
    te.cache.pix_min.assign(static_cast<size_t>(W),  std::numeric_limits<float>::max());
    te.cache.pix_max.assign(static_cast<size_t>(W),  std::numeric_limits<float>::lowest());

    // Pre-compute the maximum buffer size needed for any chunk after transforms.
    // Expanding transforms (e.g. STFT) can produce more output than input.
    int64_t max_chunk_out = RENDER_CHUNK;
    for (const auto& t : te.transforms)
        max_chunk_out = t->transformedCount(max_chunk_out);
    const size_t buf_capacity = static_cast<size_t>(std::max(RENDER_CHUNK, max_chunk_out));

    std::vector<float> buf;
    buf.reserve(buf_capacity);

    int64_t chunk_start = view_start_;
    while (chunk_start < view_end_) {
        int64_t chunk_end = std::min(chunk_start + RENDER_CHUNK, view_end_);
        int64_t chunk_len = chunk_end - chunk_start;
        // Size for both input read and worst-case transformed output.
        int64_t chunk_out = chunk_len;
        for (const auto& t : te.transforms)
            chunk_out = t->transformedCount(chunk_out);
        buf.resize(static_cast<size_t>(std::max(chunk_len, chunk_out)));
        int64_t raw_read  = readTraceSamples(te, chunk_start, chunk_len, buf.data());
        if (raw_read <= 0) break;
        int64_t out_count = raw_read;
        for (auto& t : te.transforms)
            out_count = t->apply(buf.data(), out_count, chunk_start);

        for (int64_t i = 0; i < out_count; i++) {
            int64_t s  = chunk_start + i * raw_read / out_count;
            int     px = static_cast<int>((s - view_start_) * W / view_len);
            if (px >= 0 && px < W) {
                float v = buf[static_cast<size_t>(i)];
                auto& mn = te.cache.pix_min[static_cast<size_t>(px)];
                auto& mx = te.cache.pix_max[static_cast<size_t>(px)];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        chunk_start = chunk_end;
    }

    for (int px = 0; px < W; px++) {
        float mn = te.cache.pix_min[static_cast<size_t>(px)];
        float mx = te.cache.pix_max[static_cast<size_t>(px)];
        if (mn != std::numeric_limits<float>::max()) {
            if (mn < te.cache.ymin) te.cache.ymin = mn;
            if (mx > te.cache.ymax) te.cache.ymax = mx;
        }
    }
}

// ---------------------------------------------------------------------------
// Y-range — trivially derived from the pre-built caches.
// ---------------------------------------------------------------------------

void PlotWidget::computeYRange(float& ymin, float& ymax) {
    ymin =  std::numeric_limits<float>::max();
    ymax =  std::numeric_limits<float>::lowest();

    for (const auto& te : traces_) {
        if (!te.visible || !te.cache.valid) continue;
        if (te.cache.ymin < ymin) ymin = te.cache.ymin;
        if (te.cache.ymax > ymax) ymax = te.cache.ymax;
    }

    if (ymin == std::numeric_limits<float>::max()) { ymin = -1; ymax = 1; return; }
    if (ymin == ymax) { ymin -= 1; ymax += 1; return; }

    float pad = (ymax - ymin) * 0.05f;
    ymin -= pad;
    ymax += pad;
}

// ---------------------------------------------------------------------------
// Trace rendering — reads exclusively from the pre-built cache.
// ---------------------------------------------------------------------------

void PlotWidget::renderTrace(const TraceEntry& te, QPainter& p,
                              const QRect& pr, float ymin, float ymax)
{
    if (!te.visible || !te.cache.valid) return;
    const int     W        = pr.width();
    const int64_t view_len = view_end_ - view_start_;
    if (W <= 0 || view_len <= 0) return;

    // ------------------------------------------------------------------
    // ZOOMED-IN: draw connected polyline from cached sample array.
    // ------------------------------------------------------------------
    if (!te.cache.samples.empty()) {
        const int64_t out_count = static_cast<int64_t>(te.cache.samples.size());
        const int64_t raw_read  = te.cache.raw_read;
        const float*  buf       = te.cache.samples.data();

        p.setRenderHint(QPainter::Antialiasing, true);
        p.setPen(QPen(te.color, trace_width_));

        auto sxf = [&](int64_t i) -> float {
            int64_t raw_s = (out_count > 1)
                ? view_start_ + i * raw_read / out_count
                : view_start_;
            return pr.left() + static_cast<float>(
                static_cast<double>(raw_s - view_start_) * W / view_len);
        };

        {
            std::vector<QPointF> pts;
            pts.reserve(static_cast<size_t>(out_count));
            for (int64_t i = 0; i < out_count; i++)
                pts.emplace_back(sxf(i), valueToPixel(buf[i], pr, ymin, ymax));
            p.drawPolyline(pts.data(), static_cast<int>(pts.size()));
        }

        if (out_count > 0 && W / out_count >= 4) {
            p.setBrush(te.color);
            p.setPen(Qt::NoPen);
            for (int64_t i = 0; i < out_count; i++)
                p.drawEllipse(QPointF(sxf(i), valueToPixel(buf[i], pr, ymin, ymax)), 3.0, 3.0);
            p.setBrush(Qt::NoBrush);
        }
        p.setRenderHint(QPainter::Antialiasing, false);
        return;
    }

    // ------------------------------------------------------------------
    // ZOOMED-OUT filled band: build polygon from upper (max) envelope
    // going L→R, then lower (min) envelope going R→L, and fill it.
    // ------------------------------------------------------------------
    if (te.filled) {
        std::vector<QPointF> upper_pts, lower_pts;
        upper_pts.reserve(static_cast<size_t>(W));
        lower_pts.reserve(static_cast<size_t>(W));
        for (int px = 0; px < W; px++) {
            float mn = te.cache.pix_min[static_cast<size_t>(px)];
            float mx = te.cache.pix_max[static_cast<size_t>(px)];
            if (mn == std::numeric_limits<float>::max()) continue;
            upper_pts.emplace_back(pr.left() + px,
                                   valueToPixel(mx, pr, ymin, ymax));
            lower_pts.emplace_back(pr.left() + px,
                                   valueToPixel(mn, pr, ymin, ymax));
        }
        if (!upper_pts.empty()) {
            std::vector<QPointF> poly = upper_pts;
            poly.insert(poly.end(), lower_pts.rbegin(), lower_pts.rend());
            QColor fill = te.color;
            fill.setAlpha(80);
            p.setRenderHint(QPainter::Antialiasing, false);
            p.setBrush(fill);
            p.setPen(Qt::NoPen);
            p.drawPolygon(poly.data(), static_cast<int>(poly.size()));
            p.setBrush(Qt::NoBrush);
        }
        return;
    }

    // ------------------------------------------------------------------
    // ZOOMED-OUT line: draw per-pixel min-max segments with gap bridging.
    // ------------------------------------------------------------------
    p.setPen(QPen(te.color, 1));
    int prev_top = INT_MAX, prev_bot = INT_MAX;

    for (int px = 0; px < W; px++) {
        float mn = te.cache.pix_min[static_cast<size_t>(px)];
        float mx = te.cache.pix_max[static_cast<size_t>(px)];
        if (mn == std::numeric_limits<float>::max()) {
            prev_top = prev_bot = INT_MAX;
            continue;
        }
        int top = valueToPixel(mx, pr, ymin, ymax);
        int bot = valueToPixel(mn, pr, ymin, ymax);

        if (prev_top != INT_MAX) {
            if      (bot < prev_top) bot = prev_top;
            else if (top > prev_bot) top = prev_bot;
        }
        prev_top = valueToPixel(mx, pr, ymin, ymax);
        prev_bot = valueToPixel(mn, pr, ymin, ymax);

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

    // Title
    if (!plot_title_.isEmpty()) {
        QFont tf = font(); tf.setPointSize(10); tf.setBold(true); p.setFont(tf);
        p.setPen(theme_.legend_text);
        p.drawText(QRect(ML, 0, width() - ML - MR, MT_TITLE),
                   Qt::AlignHCenter | Qt::AlignVCenter, plot_title_);
    }

    if (traces_.empty() || view_end_ <= view_start_) {
        QColor tc = theme_.axis_text; tc.setAlpha(160);
        p.setPen(tc);
        p.drawText(pr, Qt::AlignCenter, "No traces loaded.\nUse File › Open to load a .trs file.");
        return;
    }

    // Build (or reuse) per-pixel min/max caches — one pass per trace per frame.
    const int W = pr.width();
    for (auto& te : traces_)
        buildTraceCache(te, W);

    float ymin, ymax;
    computeYRange(ymin, ymax);
    // Sticky range: expand eagerly, never auto-shrink (prevents flicker while panning).
    // Reset by resetView(), resetYZoom(), or clearTraces().
    if (ymin < sticky_ymin_) sticky_ymin_ = ymin;
    if (ymax > sticky_ymax_) sticky_ymax_ = ymax;
    ymin = sticky_ymin_;
    ymax = sticky_ymax_;
    // Apply Y-axis zoom (Ctrl+scroll): expand/contract range around centre.
    if (y_scale_ != 1.0f) {
        float center = (ymin + ymax) * 0.5f;
        float half   = (ymax - ymin) * 0.5f * y_scale_;
        ymin = center - half;
        ymax = center + half;
    }
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
        const QColor thr_color(220, 120, 0, 230);  // amber-orange
        QPen thr_pen(thr_color, 1.2, Qt::DashLine);
        QFont tf = font(); tf.setPointSize(8); tf.setBold(true);
        auto drawThrLine = [&](double tv) {
            int ty = valueToPixel(static_cast<float>(tv), pr, ymin, ymax);
            if (ty >= pr.top() && ty <= pr.bottom()) {
                p.setPen(thr_pen);
                p.drawLine(pr.left(), ty, pr.right(), ty);
                p.setFont(tf);
                p.setPen(thr_color);
                p.drawText(pr.left() + 4, ty - 3, QString("%1").arg(tv, 0, 'f', 2));
            }
        };
        drawThrLine(threshold_pos_);
        if (!threshold_one_sided_) drawThrLine(threshold_neg_);
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

    // Axis name labels
    if (!axis_label_x_.isEmpty() || !axis_label_y_.isEmpty()) {
        QFont lf = font(); lf.setPointSize(9); lf.setBold(false); p.setFont(lf);
        p.setPen(theme_.axis_text);
        if (!axis_label_x_.isEmpty())
            p.drawText(QRect(pr.left(), height() - 18, pr.width(), 18),
                       Qt::AlignHCenter | Qt::AlignVCenter, axis_label_x_);
        if (!axis_label_y_.isEmpty()) {
            p.save();
            p.translate(12, pr.top() + pr.height() / 2);
            p.rotate(-90);
            p.drawText(QRect(-pr.height()/2, -14, pr.height(), 16),
                       Qt::AlignHCenter | Qt::AlignVCenter, axis_label_y_);
            p.restore();
        }
    }

    // X-axis labels
    p.setPen(theme_.axis_text);
    QFont f = font(); f.setPointSize(8); p.setFont(f);

    for (int i = 0; i <= 6; i++) {
        int64_t s = view_start_ + (view_end_ - view_start_) * i / 6;
        int     x = pr.left() + pr.width() * i / 6;
        p.drawLine(x, pr.bottom(), x, pr.bottom() + 4);

        double sv = static_cast<double>(s) * x_label_scale_ + x_label_offset_;
        QString lbl;
        if (x_label_scale_ == 1.0 && x_label_offset_ == 0.0) {
            // Raw sample/bin index — use integer formatting with k/M suffixes.
            if      (s >= 1'000'000) lbl = QString("%1M").arg(s / 1'000'000.0, 0, 'f', 2);
            else if (s >= 1'000)     lbl = QString("%1k").arg(s / 1'000.0,     0, 'f', 1);
            else                     lbl = QString::number(s);
        } else {
            // Scaled axis (e.g. Hz) — use SI prefixes.
            double av = std::abs(sv);
            if      (av >= 1e9)  lbl = QString("%1G").arg(sv / 1e9,  0, 'f', 2);
            else if (av >= 1e6)  lbl = QString("%1M").arg(sv / 1e6,  0, 'f', 2);
            else if (av >= 1e3)  lbl = QString("%1k").arg(sv / 1e3,  0, 'f', 1);
            else if (av >= 1.0)  lbl = QString("%1").arg(sv,          0, 'f', 1);
            else if (av >= 1e-3) lbl = QString("%1m").arg(sv * 1e3,  0, 'f', 1);
            else                 lbl = QString("%1µ").arg(sv * 1e6,  0, 'f', 1);
        }
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
        QFont lf = font(); lf.setPointSize(8); p.setFont(lf);
        int lx = pr.right() - 6;
        int ly = pr.top() + 8;
        for (const auto& te : traces_) {
            if (!te.visible) continue;
            if (te.filled) {
                QColor swatch = te.color; swatch.setAlpha(80);
                p.fillRect(lx - 18, ly - 5, 18, 10, swatch);
                p.setPen(QPen(te.color, 1));
                p.drawRect(lx - 18, ly - 5, 18, 10);
            } else {
                p.setPen(te.color);
                p.drawLine(lx - 18, ly, lx, ly);
            }
            p.setPen(theme_.legend_text);
            p.drawText(lx - 150, ly - 8, 128, 16, Qt::AlignRight, te.label);
            ly += 18;
        }
        if (show_thresholds_) {
            const QColor thr_color(220, 120, 0, 230);
            p.setPen(QPen(thr_color, 1, Qt::DashLine));
            p.drawLine(lx - 18, ly, lx, ly);
            p.setPen(theme_.legend_text);
            p.drawText(lx - 150, ly - 8, 128, 16, Qt::AlignRight,
                       QString("Threshold (+%1)").arg(threshold_pos_, 0, 'f', 1));
            ly += 18;
            if (!threshold_one_sided_) {
                p.setPen(QPen(thr_color, 1, Qt::DashLine));
                p.drawLine(lx - 18, ly, lx, ly);
                p.setPen(theme_.legend_text);
                p.drawText(lx - 150, ly - 8, 128, 16, Qt::AlignRight,
                           QString("Threshold (-%1)").arg(threshold_pos_, 0, 'f', 1));
            }
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

    if (mode_ == InteractionMode::AlignDrag) {
        if (!pr.contains(e->pos())) return;
        int idx = nearestTrace(e->pos().x(), e->pos().y(), pr);
        if (idx < 0) return;
        align_drag_idx_          = idx;
        align_drag_sample_origin_ = pixelToSample(e->pos().x(), pr);
        align_drag_shift_origin_  = traces_[static_cast<size_t>(idx)].shift;
        setCursor(Qt::SizeHorCursor);
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

    if (mode_ == InteractionMode::AlignDrag && align_drag_idx_ >= 0) {
        QRect   pr      = plotRect();
        int64_t cur_s   = pixelToSample(e->pos().x(), pr);
        int64_t delta   = cur_s - align_drag_sample_origin_;
        auto&   te      = traces_[static_cast<size_t>(align_drag_idx_)];
        int32_t new_shift = align_drag_shift_origin_ + static_cast<int32_t>(delta);
        if (te.shift != new_shift) {
            te.shift       = new_shift;
            te.cache.valid = false;
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

    if (mode_ == InteractionMode::AlignDrag && align_drag_idx_ >= 0) {
        align_drag_idx_ = -1;
        setCursor(Qt::ArrowCursor);
        emit traceShiftsChanged();
        return;
    }

    if (dragging_) {
        dragging_ = false;
        setCursor(mode_ == InteractionMode::BoxZoom ? Qt::CrossCursor : Qt::ArrowCursor);
    }
}

void PlotWidget::wheelEvent(QWheelEvent* e) {
    // Ctrl+scroll or Shift+scroll → Y-axis zoom (expand/contract amplitude range)
    if (e->modifiers() & (Qt::ControlModifier | Qt::ShiftModifier)) {
        float factor = (e->angleDelta().y() > 0) ? 0.75f : 1.0f / 0.75f;
        y_scale_ = std::clamp(y_scale_ * factor, 0.05f, 200.0f);
        update();
        return;
    }

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
