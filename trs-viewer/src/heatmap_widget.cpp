#include "heatmap_widget.h"

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Separable Gaussian blur (clamp-to-edge boundary)
// ---------------------------------------------------------------------------
static void gaussianBlur(const float* src, float* dst, int rows, int cols, float sigma) {
    if (rows <= 0 || cols <= 0) return;
    if (sigma <= 0.0f) {
        std::copy(src, src + static_cast<size_t>(rows) * cols, dst);
        return;
    }
    int r = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    int ksize = 2 * r + 1;
    std::vector<float> kernel(ksize);
    float ksum = 0.0f;
    for (int i = 0; i < ksize; i++) {
        float x = static_cast<float>(i - r);
        kernel[i] = std::exp(-0.5f * x * x / (sigma * sigma));
        ksum += kernel[i];
    }
    for (auto& k : kernel) k /= ksum;

    std::vector<float> tmp(static_cast<size_t>(rows) * cols);

    // Horizontal pass
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float acc = 0.0f;
            for (int ki = 0; ki < ksize; ki++) {
                int c = std::clamp(col + ki - r, 0, cols - 1);
                acc += src[row * cols + c] * kernel[ki];
            }
            tmp[row * cols + col] = acc;
        }
    }

    // Vertical pass
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float acc = 0.0f;
            for (int ki = 0; ki < ksize; ki++) {
                int rr = std::clamp(row + ki - r, 0, rows - 1);
                acc += tmp[rr * cols + col] * kernel[ki];
            }
            dst[row * cols + col] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Colormaps — anchor points, t ∈ [0, 1]  (low → high)
// ---------------------------------------------------------------------------
struct CmapEntry { float t; uint8_t r, g, b; };

static const CmapEntry kRdBu[] = {
    {0.00f,   5, 113, 176}, {0.25f, 146, 197, 222},
    {0.50f, 247, 247, 247}, {0.75f, 244, 165, 130},
    {1.00f, 202,   0,  32},
};
static const CmapEntry kGrayscale[] = {
    {0.0f,   0,   0,   0},
    {1.0f, 255, 255, 255},
};
static const CmapEntry kHot[] = {
    {0.00f,   0,   0,   0}, {0.33f, 255,   0,   0},
    {0.67f, 255, 255,   0}, {1.00f, 255, 255, 255},
};
static const CmapEntry kViridis[] = {
    {0.00f,  68,   1,  84}, {0.25f,  59,  82, 139},
    {0.50f,  33, 145, 140}, {0.75f,  94, 201,  98},
    {1.00f, 253, 231,  37},
};
static const CmapEntry kPlasma[] = {
    {0.00f,  13,   8, 135}, {0.25f, 156,  23, 158},
    {0.50f, 237, 104,  60}, {0.75f, 246, 200,  33},
    {1.00f, 240, 249,  33},
};
static const CmapEntry kLukasz[] = {
    {0.0f,   0,   0,   0},
    {1.0f,   0, 255,   0},
};

static QRgb interpColormap(float t, const CmapEntry* cm, int n) {
    t = std::clamp(t, 0.0f, 1.0f);
    int seg = 0;
    for (int i = 0; i < n - 2; i++)
        if (t >= cm[i + 1].t) seg = i + 1;
    float lo = cm[seg].t, hi = cm[seg + 1].t;
    float f  = (hi > lo) ? (t - lo) / (hi - lo) : 0.0f;
    f = std::clamp(f, 0.0f, 1.0f);
    int r = static_cast<int>(cm[seg].r + f * (cm[seg+1].r - cm[seg].r));
    int g = static_cast<int>(cm[seg].g + f * (cm[seg+1].g - cm[seg].g));
    int b = static_cast<int>(cm[seg].b + f * (cm[seg+1].b - cm[seg].b));
    return qRgb(std::clamp(r,0,255), std::clamp(g,0,255), std::clamp(b,0,255));
}

QRgb HeatmapWidget::colormap(float v) const {
    float range = vmax_ - vmin_;
    float t = (range != 0.0f) ? (v - vmin_) / range : 0.5f;
    switch (color_scheme_) {
    case ColorScheme::Grayscale: return interpColormap(t, kGrayscale, 2);
    case ColorScheme::Hot:       return interpColormap(t, kHot,       4);
    case ColorScheme::Viridis:   return interpColormap(t, kViridis,   5);
    case ColorScheme::Plasma:    return interpColormap(t, kPlasma,    5);
    case ColorScheme::Lukasz:    return interpColormap(t, kLukasz,   2);
    default: /* RdBu */          return interpColormap(t, kRdBu,      5);
    }
}

// ---------------------------------------------------------------------------
// Screen-space renderer: maps a W×H image to data region [x0,x1)×[y0,y1).
// Only reads the visible subset of display_matrix_, colormapped per pixel.
// Rows are processed in parallel with OpenMP.
// ---------------------------------------------------------------------------
QImage HeatmapWidget::renderRegion(int W, int H,
                                    double x0, double y0,
                                    double x1, double y1) const
{
    QImage img(W, H, QImage::Format_RGB32);
    if (rows_ <= 0 || cols_ <= 0 || display_matrix_.empty() || W <= 0 || H <= 0)
        return img;

    QRgb* bits = reinterpret_cast<QRgb*>(img.bits());
    const int bpl = img.bytesPerLine() / static_cast<int>(sizeof(QRgb));

    const double dx = (x1 - x0) / W;
    const double dy = (y1 - y0) / H;
    const int    Nr = rows_, Nc = cols_;
    const float* dm = display_matrix_.data();

#pragma omp parallel for schedule(static)
    for (int row = 0; row < H; row++) {
        const int data_row = std::clamp(static_cast<int>(y0 + (row + 0.5) * dy), 0, Nr - 1);
        const float* src   = dm + static_cast<size_t>(data_row) * Nc;
        QRgb*        line  = bits + row * bpl;
        for (int col = 0; col < W; col++) {
            const int data_col = std::clamp(static_cast<int>(x0 + (col + 0.5) * dx), 0, Nc - 1);
            line[col] = colormap(src[data_col]);
        }
    }
    return img;
}

// ---------------------------------------------------------------------------
HeatmapWidget::HeatmapWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);
    setMinimumSize(300, 300);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(22, 22, 28));
    setPalette(pal);
    setAutoFillBackground(true);
}

void HeatmapWidget::setMatrix(const std::vector<float>& data, int32_t rows, int32_t cols) {
    matrix_ = data;
    rows_   = rows;
    cols_   = cols;
    resetView();
    applyProcessing();
    update();
}

void HeatmapWidget::setGaussianSigma(float sigma) {
    gaussian_sigma_ = std::max(0.0f, sigma);
    applyProcessing();
    update();
}

void HeatmapWidget::setAbsValue(bool enabled) {
    abs_value_ = enabled;
    applyProcessing();
    update();
}

void HeatmapWidget::setPowerGamma(float gamma) {
    power_gamma_ = std::max(1.0f, gamma);
    applyProcessing();
    update();
}

void HeatmapWidget::setBinaryThreshold(bool enabled, float threshold) {
    threshold_enabled_ = enabled;
    threshold_value_   = threshold;
    applyProcessing();
    update();
}

void HeatmapWidget::setColorScheme(ColorScheme scheme) {
    color_scheme_ = scheme;
    update();
}

void HeatmapWidget::setColorRange(float vmin, float vmax) {
    if (vmax <= vmin) vmax = vmin + 1e-6f;
    vmin_ = vmin;
    vmax_ = vmax;
    update();
}

void HeatmapWidget::resetView() {
    view_x0_ = 0.0;
    view_x1_ = static_cast<double>(cols_);
    view_y0_ = 0.0;
    view_y1_ = static_cast<double>(rows_);
    update();
}

void HeatmapWidget::applyProcessing() {
    if (rows_ <= 0 || cols_ <= 0 ||
        static_cast<int64_t>(matrix_.size()) < static_cast<int64_t>(rows_) * cols_)
        return;
    const int sz = static_cast<int>(matrix_.size());
    display_matrix_.resize(matrix_.size());

    // Step 1: Gaussian blur
    gaussianBlur(matrix_.data(), display_matrix_.data(), rows_, cols_, gaussian_sigma_);

    // Step 2: Absolute value — makes negative correlations as visible as positive ones
    if (abs_value_) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++)
            display_matrix_[i] = std::abs(display_matrix_[i]);
    }

    // Step 3: Power/gamma — compresses the dynamic range.
    // Values are clamped to [0,1] before raising to the power, so the
    // mapping stays well-defined regardless of the sign or magnitude.
    // A gamma > 1 darkens the background and sharpens peaks;
    // the diagonal (value=1) stays at 1.
    if (power_gamma_ > 1.0f) {
        const float inv_range = (vmax_ != vmin_) ? 1.0f / (vmax_ - vmin_) : 1.0f;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++) {
            float t = std::clamp((display_matrix_[i] - vmin_) * inv_range, 0.0f, 1.0f);
            display_matrix_[i] = std::pow(t, power_gamma_) * (vmax_ - vmin_) + vmin_;
        }
    }

    // Step 4: Binary threshold
    if (threshold_enabled_) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < sz; i++)
            display_matrix_[i] = (std::abs(display_matrix_[i]) >= threshold_value_) ? 1.0f : 0.0f;
    }
}

void HeatmapWidget::computeClipRange(float percentile,
                                      float& out_vmin, float& out_vmax) const {
    if (display_matrix_.empty()) { out_vmin = vmin_; out_vmax = vmax_; return; }
    std::vector<float> vals = display_matrix_;
    const size_t n = vals.size();

    // Upper clip point
    size_t hi = static_cast<size_t>(std::clamp(percentile, 0.0f, 1.0f) * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + hi, vals.end());
    out_vmax = vals[hi];

    // Lower clip point (symmetric: 1 - percentile)
    size_t lo = static_cast<size_t>(std::clamp(1.0f - percentile, 0.0f, 1.0f) * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + lo, vals.end());
    out_vmin = vals[lo];

    if (out_vmax <= out_vmin) out_vmax = out_vmin + 1e-6f;
}

bool HeatmapWidget::exportPng(const QString& path, int max_pixels) const {
    if (rows_ <= 0 || cols_ <= 0 || display_matrix_.empty()) return false;
    // Scale to max_pixels on the longer side, preserving aspect ratio.
    int W = cols_, H = rows_;
    if (W > max_pixels || H > max_pixels) {
        if (W >= H) { W = max_pixels; H = std::max(1, max_pixels * rows_ / cols_); }
        else        { H = max_pixels; W = std::max(1, max_pixels * cols_ / rows_); }
    }
    QImage img = renderRegion(W, H, 0.0, 0.0,
                               static_cast<double>(cols_), static_cast<double>(rows_));
    return img.save(path, "PNG");
}

QRect HeatmapWidget::plotRect() const {
    return QRect(ML, MT, width() - ML - MR, height() - MT - MB);
}

// ---------------------------------------------------------------------------
void HeatmapWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);

    QRect pr = plotRect();

    if (rows_ <= 0 || cols_ <= 0 || display_matrix_.empty()) {
        p.setPen(QColor(180, 180, 200));
        p.drawText(rect(), Qt::AlignCenter, "No data.\nRun SCA → Cross-Correlation.");
        return;
    }

    // Render only the visible viewport — O(W×H) not O(M²)
    QImage viewport = renderRegion(pr.width(), pr.height(),
                                    view_x0_, view_y0_,
                                    view_x1_, view_y1_);
    p.drawImage(pr.topLeft(), viewport);

    // Border
    p.setPen(QColor(80, 80, 100));
    p.drawRect(pr);

    // Axis labels
    p.setPen(QColor(180, 180, 200));
    QFont f = font(); f.setPointSize(8); p.setFont(f);

    auto fmt = [](double v) -> QString {
        if (v >= 1e6) return QString("%1M").arg(v/1e6, 0,'f',1);
        if (v >= 1e3) return QString("%1k").arg(v/1e3, 0,'f',1);
        return QString::number(static_cast<int64_t>(std::round(v)));
    };

    double span_x = view_x1_ - view_x0_;
    double span_y = view_y1_ - view_y0_;

    for (int i = 0; i <= 6; i++) {
        double fx = view_x0_ + span_x * i / 6.0;
        int px = pr.left() + pr.width() * i / 6;
        p.drawLine(px, pr.bottom(), px, pr.bottom() + 4);
        p.drawText(px - 28, pr.bottom() + 5, 56, 18, Qt::AlignCenter, fmt(fx));

        double fy = view_y0_ + span_y * i / 6.0;
        int py = pr.top() + pr.height() * i / 6;
        p.drawLine(pr.left() - 4, py, pr.left(), py);
        p.drawText(0, py - 9, ML - 6, 18, Qt::AlignRight | Qt::AlignVCenter, fmt(fy));
    }

    // Colour bar
    {
        const int bar_w = 10;
        const int bar_x = pr.right() + 2;
        const int bar_h = pr.height();
        for (int yi = 0; yi < bar_h; yi++) {
            float v = vmax_ - (vmax_ - vmin_) * static_cast<float>(yi) / bar_h;
            QRgb c  = colormap(v);
            p.setPen(QColor(c));
            p.drawLine(bar_x, pr.top() + yi, bar_x + bar_w - 1, pr.top() + yi);
        }
        p.setPen(QColor(80, 80, 100));
        p.drawRect(bar_x, pr.top(), bar_w, bar_h);
        p.setPen(QColor(180, 180, 200));
        p.setFont(f);
        p.drawText(bar_x + bar_w + 2, pr.top() + 10,
                   QString::number(static_cast<double>(vmax_), 'g', 3));
        p.drawText(bar_x + bar_w + 2, pr.bottom(),
                   QString::number(static_cast<double>(vmin_), 'g', 3));
    }
}

// ---------------------------------------------------------------------------
void HeatmapWidget::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        dragging_    = true;
        drag_origin_ = e->pos();
        drag_x0_     = view_x0_;
        drag_y0_     = view_y0_;
        setCursor(Qt::ClosedHandCursor);
    }
}

void HeatmapWidget::mouseMoveEvent(QMouseEvent* e) {
    QRect pr = plotRect();

    if (dragging_) {
        double px_per_unit_x = pr.width()  / std::max(1.0, view_x1_ - view_x0_);
        double px_per_unit_y = pr.height() / std::max(1.0, view_y1_ - view_y0_);

        double dx = -(e->pos().x() - drag_origin_.x()) / px_per_unit_x;
        double dy = -(e->pos().y() - drag_origin_.y()) / px_per_unit_y;

        double span_x = view_x1_ - view_x0_;
        double span_y = view_y1_ - view_y0_;
        view_x0_ = std::clamp(drag_x0_ + dx, 0.0, static_cast<double>(cols_) - span_x);
        view_x1_ = view_x0_ + span_x;
        view_y0_ = std::clamp(drag_y0_ + dy, 0.0, static_cast<double>(rows_) - span_y);
        view_y1_ = view_y0_ + span_y;
        update();
    }

    if (rows_ > 0 && cols_ > 0 && pr.contains(e->pos())) {
        double fx = view_x0_ + (view_x1_ - view_x0_) * (e->pos().x() - pr.left()) / pr.width();
        double fy = view_y0_ + (view_y1_ - view_y0_) * (e->pos().y() - pr.top())  / pr.height();
        int s1 = static_cast<int>(fy);
        int s2 = static_cast<int>(fx);
        if (s1 >= 0 && s1 < rows_ && s2 >= 0 && s2 < cols_)
            emit hoverInfo(s1, s2, matrix_[static_cast<size_t>(s1 * cols_ + s2)]);
    }
}

void HeatmapWidget::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && dragging_) {
        dragging_ = false;
        setCursor(Qt::ArrowCursor);
    }
}

void HeatmapWidget::wheelEvent(QWheelEvent* e) {
    QRect pr = plotRect();

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    double px = e->position().x();
    double py = e->position().y();
#else
    double px = e->posF().x();
    double py = e->posF().y();
#endif

    double cx = view_x0_ + (view_x1_ - view_x0_) * (px - pr.left()) / pr.width();
    double cy = view_y0_ + (view_y1_ - view_y0_) * (py - pr.top())  / pr.height();

    double factor = (e->angleDelta().y() > 0) ? 0.7 : 1.0 / 0.7;
    double span_x = std::max(2.0, (view_x1_ - view_x0_) * factor);
    double span_y = std::max(2.0, (view_y1_ - view_y0_) * factor);

    double frac_x = (pr.width()  > 0) ? (px - pr.left()) / pr.width()  : 0.5;
    double frac_y = (pr.height() > 0) ? (py - pr.top())  / pr.height() : 0.5;
    double nx0 = cx - frac_x * span_x;
    double ny0 = cy - frac_y * span_y;

    double Md_x = static_cast<double>(cols_);
    double Md_y = static_cast<double>(rows_);
    view_x0_ = std::clamp(nx0, 0.0, Md_x - span_x);
    view_x1_ = view_x0_ + span_x;
    if (view_x1_ > Md_x) { view_x1_ = Md_x; view_x0_ = std::max(0.0, Md_x - span_x); }

    view_y0_ = std::clamp(ny0, 0.0, Md_y - span_y);
    view_y1_ = view_y0_ + span_y;
    if (view_y1_ > Md_y) { view_y1_ = Md_y; view_y0_ = std::max(0.0, Md_y - span_y); }

    update();
}

void HeatmapWidget::resizeEvent(QResizeEvent*) { update(); }
