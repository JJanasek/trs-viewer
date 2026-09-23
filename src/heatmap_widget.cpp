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
    return colormapT(t);
}

QRgb HeatmapWidget::colormapT(float t) const {
    // Contrast stretches about the midpoint, brightness shifts; interpColormap
    // clamps the result to [0,1], which is what makes both safe to over-drive.
    t = (t - 0.5f) * contrast_ + 0.5f + brightness_;
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

    // Cells per pixel decides the source level. Below one, every cell is
    // drawn anyway. Above, sample a pooled level whose block is the *smallest*
    // power of two >= cells-per-pixel, so each pooled cell is at least a pixel
    // wide and none can fall between samples — that is what keeps a small
    // hotspot on screen at any zoom (see Downsample).
    const double cpp = std::max(dx, dy);
    int f = 1;
    if (downsample_ != Downsample::Nearest && cpp > 1.0)
        while (f < cpp) f *= 2;
    if (f > 1) ensurePooled(f);
    const float* dm     = (f > 1) ? pooled_.data() : display_matrix_.data();
    const int    stride = (f > 1) ? pooled_cols_    : Nc;

    // Inside a selected region, each cell's normalised value is blended from
    // the global scale toward the region's own scale by region_boost_ — the
    // rest of the heatmap is untouched, so the region's strongest cells stand
    // out against a background that still reads normally. Optionally the
    // outside is darkened too (not hidden — it still gives context).
    const bool  boost = sel_valid_ && region_boost_ > 0.0f;
    const bool  dim   = sel_valid_ && dim_outside_;
    const float g_rng = vmax_ - vmin_;
    const float l_rng = sel_hi_ - sel_lo_;
    auto dimmed = [](QRgb c) {
        return qRgb(qRed(c) * 35 / 100, qGreen(c) * 35 / 100, qBlue(c) * 35 / 100);
    };

#pragma omp parallel for schedule(static)
    for (int row = 0; row < H; row++) {
        const int data_row = std::clamp(static_cast<int>(y0 + (row + 0.5) * dy), 0, Nr - 1);
        const float* src   = dm + static_cast<size_t>(data_row / f) * stride;
        QRgb*        line  = bits + row * bpl;
        const bool row_in  = sel_valid_ && data_row >= sel_r0_ && data_row <= sel_r1_;
        for (int col = 0; col < W; col++) {
            const int   data_col = std::clamp(static_cast<int>(x0 + (col + 0.5) * dx), 0, Nc - 1);
            const bool  inside   = row_in && data_col >= sel_c0_ && data_col <= sel_c1_;
            const float v        = src[data_col / f];
            QRgb c;
            if (inside && boost) {
                const float tg = (g_rng != 0.0f) ? (v - vmin_)  / g_rng : 0.5f;
                const float tl = (l_rng != 0.0f) ? (v - sel_lo_) / l_rng : 0.5f;
                c = colormapT(tg + (tl - tg) * region_boost_);
            } else {
                c = colormap(v);
                if (dim && !inside) c = dimmed(c);
            }
            line[col] = c;
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

void HeatmapWidget::setContrast(float contrast) {
    contrast_ = std::max(0.0f, contrast);
    update();
}

void HeatmapWidget::setBrightness(float brightness) {
    brightness_ = brightness;
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

void HeatmapWidget::ensurePooled(int f) const {
    if (f <= 1) return;
    if (pool_factor_ == f && !pooled_.empty()) return;
    pooled_rows_ = (rows_ + f - 1) / f;
    pooled_cols_ = (cols_ + f - 1) / f;
    pooled_.assign(static_cast<size_t>(pooled_rows_) * pooled_cols_, 0.0f);
    const float* dm = display_matrix_.data();
    const bool peak = (downsample_ == Downsample::Peak);

#pragma omp parallel for schedule(static)
    for (int pr = 0; pr < pooled_rows_; pr++) {
        const int r0 = pr * f, r1 = std::min(rows_, r0 + f);
        float* out = pooled_.data() + static_cast<size_t>(pr) * pooled_cols_;
        for (int pc = 0; pc < pooled_cols_; pc++) {
            const int c0 = pc * f, c1 = std::min(cols_, c0 + f);
            if (peak) {
                float best = 0.0f, best_abs = -1.0f;
                for (int r = r0; r < r1; r++) {
                    const float* src = dm + static_cast<size_t>(r) * cols_;
                    for (int c = c0; c < c1; c++) {
                        const float a = std::abs(src[c]);
                        if (a > best_abs) { best_abs = a; best = src[c]; }
                    }
                }
                out[pc] = best;
            } else {
                double sum = 0.0;
                for (int r = r0; r < r1; r++) {
                    const float* src = dm + static_cast<size_t>(r) * cols_;
                    for (int c = c0; c < c1; c++) sum += src[c];
                }
                out[pc] = static_cast<float>(sum / ((r1 - r0) * (c1 - c0)));
            }
        }
    }
    pool_factor_ = f;
}

void HeatmapWidget::setDownsample(Downsample d) {
    downsample_ = d;
    invalidatePooled();
    update();
}

void HeatmapWidget::setView(double x0, double y0, double x1, double y1) {
    if (rows_ <= 0 || cols_ <= 0) return;
    const double Mx = static_cast<double>(cols_), My = static_cast<double>(rows_);
    double sx = std::clamp(x1 - x0, 2.0, Mx);
    double sy = std::clamp(y1 - y0, 2.0, My);
    view_x0_ = std::clamp(x0, 0.0, Mx - sx); view_x1_ = view_x0_ + sx;
    view_y0_ = std::clamp(y0, 0.0, My - sy); view_y1_ = view_y0_ + sy;
    update();
}

void HeatmapWidget::setSelection(int row0, int col0, int row1, int col1) {
    if (rows_ <= 0 || cols_ <= 0) return;
    sel_r0_ = std::clamp(std::min(row0, row1), 0, rows_ - 1);
    sel_r1_ = std::clamp(std::max(row0, row1), 0, rows_ - 1);
    sel_c0_ = std::clamp(std::min(col0, col1), 0, cols_ - 1);
    sel_c1_ = std::clamp(std::max(col0, col1), 0, cols_ - 1);
    sel_valid_ = true;
    selecting_ = false;
    refreshSelectionRange();
    emit regionSelected(sel_r0_, sel_c0_, sel_r1_, sel_c1_);
    update();
}

void HeatmapWidget::zoomToSelection(double margin_frac) {
    if (!sel_valid_) return;
    const double w = sel_c1_ + 1 - sel_c0_, h = sel_r1_ + 1 - sel_r0_;
    const double mx = w * margin_frac, my = h * margin_frac;
    setView(sel_c0_ - mx, sel_r0_ - my, sel_c1_ + 1 + mx, sel_r1_ + 1 + my);
}

std::vector<HeatmapWidget::Hotspot> HeatmapWidget::findHotspots(int count, int block) const {
    std::vector<Hotspot> out;
    if (rows_ <= 0 || cols_ <= 0 || display_matrix_.empty() || count <= 0) return out;
    block = std::max(1, block);
    const int br = (rows_ + block - 1) / block, bc = (cols_ + block - 1) / block;
    const bool square = (rows_ == cols_);
    std::vector<Hotspot> per_block(static_cast<size_t>(br) * bc, Hotspot{-1, -1, 0.0f});

#pragma omp parallel for schedule(static)
    for (int pr = 0; pr < br; pr++) {
        for (int pc = 0; pc < bc; pc++) {
            if (square && std::abs(pr - pc) <= 1) continue;   // diagonal band: trivially strong
            const int r0 = pr * block, r1 = std::min(rows_, r0 + block);
            const int c0 = pc * block, c1 = std::min(cols_, c0 + block);
            Hotspot best{-1, -1, 0.0f};
            float best_abs = -1.0f;
            for (int r = r0; r < r1; r++) {
                const float* src = display_matrix_.data() + static_cast<size_t>(r) * cols_;
                for (int c = c0; c < c1; c++) {
                    const float a = std::abs(src[c]);
                    if (a > best_abs) { best_abs = a; best = Hotspot{r, c, src[c]}; }
                }
            }
            per_block[static_cast<size_t>(pr) * bc + pc] = best;
        }
    }
    for (const auto& h : per_block) if (h.row >= 0) out.push_back(h);
    const size_t keep = std::min<size_t>(static_cast<size_t>(count), out.size());
    std::partial_sort(out.begin(), out.begin() + static_cast<ptrdiff_t>(keep), out.end(),
                      [](const Hotspot& a, const Hotspot& b) { return std::abs(a.value) > std::abs(b.value); });
    out.resize(keep);
    return out;
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
    invalidatePooled();
    refreshSelectionRange();   // the region's own range follows the processed values
}

// Symmetric percentile clip of `vals` (consumed — it is partially sorted in
// place): vmax at `percentile`, vmin at 1-percentile.
static void percentileRange(std::vector<float>& vals, float percentile,
                             float& out_vmin, float& out_vmax) {
    const size_t n = vals.size();
    size_t hi = static_cast<size_t>(std::clamp(percentile, 0.0f, 1.0f) * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + hi, vals.end());
    out_vmax = vals[hi];

    size_t lo = static_cast<size_t>(std::clamp(1.0f - percentile, 0.0f, 1.0f) * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + lo, vals.end());
    out_vmin = vals[lo];

    if (out_vmax <= out_vmin) out_vmax = out_vmin + 1e-6f;
}

void HeatmapWidget::computeClipRange(float percentile,
                                      float& out_vmin, float& out_vmax) const {
    if (display_matrix_.empty()) { out_vmin = vmin_; out_vmax = vmax_; return; }
    std::vector<float> vals = display_matrix_;
    percentileRange(vals, percentile, out_vmin, out_vmax);
}

bool HeatmapWidget::computeClipRangeInSelection(float percentile,
                                                 float& out_vmin, float& out_vmax) const {
    if (!sel_valid_ || display_matrix_.empty()) return false;
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(sel_r1_ - sel_r0_ + 1) * (sel_c1_ - sel_c0_ + 1));
    for (int r = sel_r0_; r <= sel_r1_; r++)
        for (int c = sel_c0_; c <= sel_c1_; c++)
            vals.push_back(display_matrix_[static_cast<size_t>(r) * cols_ + c]);
    percentileRange(vals, percentile, out_vmin, out_vmax);
    return true;
}

void HeatmapWidget::setSelectMode(bool on) {
    select_mode_ = on;
    setCursor(on ? Qt::CrossCursor : Qt::ArrowCursor);
}

void HeatmapWidget::clearSelection() {
    sel_valid_ = false;
    selecting_ = false;
    update();
}

void HeatmapWidget::setDimOutsideSelection(bool on) {
    dim_outside_ = on;
    update();
}

void HeatmapWidget::setRegionBoost(float strength01) {
    region_boost_ = std::clamp(strength01, 0.0f, 1.0f);
    update();
}

void HeatmapWidget::setSelectionPercentile(float percentile) {
    sel_percentile_ = std::clamp(percentile, 0.0f, 1.0f);
    refreshSelectionRange();
    update();
}

void HeatmapWidget::refreshSelectionRange() {
    if (!sel_valid_) return;
    computeClipRangeInSelection(sel_percentile_, sel_lo_, sel_hi_);
}

bool HeatmapWidget::exportPng(const QString& path, int max_pixels) const {
    if (rows_ <= 0 || cols_ <= 0 || display_matrix_.empty()) return false;
    // Export exactly what is on screen — the current zoomed view, not the
    // whole matrix — so the PNG matches what the user is looking at
    // (contrast, brightness, colour range, region boost and all). On a huge
    // matrix, exporting the full extent would pool it down to max_pixels and
    // lose whatever region the user had zoomed into. Same aspect ratio as the
    // visible plot rectangle, capped at max_pixels on the longer side.
    const double vw = view_x1_ - view_x0_, vh = view_y1_ - view_y0_;
    if (vw <= 0.0 || vh <= 0.0) return false;
    const double aspect = vw / vh;
    int W, H;
    if (aspect >= 1.0) { W = max_pixels; H = std::max(1, static_cast<int>(max_pixels / aspect)); }
    else               { H = max_pixels; W = std::max(1, static_cast<int>(max_pixels * aspect)); }
    QImage img = renderRegion(W, H, view_x0_, view_y0_, view_x1_, view_y1_);
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

    // Selection: the live rubber band while dragging, the committed region
    // outline afterwards (in data coordinates, so it tracks pan/zoom).
    if (selecting_) {
        p.setPen(QPen(QColor(255, 255, 255, 220), 1, Qt::DashLine));
        p.drawRect(QRect(sel_press_, sel_cur_).normalized().intersected(pr));
    } else if (sel_valid_) {
        auto sx = [&](double col) { return pr.left() + (col - view_x0_) / (view_x1_ - view_x0_) * pr.width(); };
        auto sy = [&](double row) { return pr.top()  + (row - view_y0_) / (view_y1_ - view_y0_) * pr.height(); };
        QRectF r(QPointF(sx(sel_c0_), sy(sel_r0_)), QPointF(sx(sel_c1_ + 1), sy(sel_r1_ + 1)));
        p.setClipRect(pr);
        p.setPen(QPen(QColor(255, 255, 255, 230), 2));
        p.drawRect(r);
        p.setClipping(false);
    }

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
    if (e->button() == Qt::LeftButton &&
        (select_mode_ || (e->modifiers() & Qt::ShiftModifier)) &&
        plotRect().contains(e->pos())) {
        selecting_ = true;
        sel_press_ = sel_cur_ = e->pos();
        update();
        return;
    }
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

    if (selecting_) {
        sel_cur_ = e->pos();
        update();
    } else if (dragging_) {
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
    if (e->button() == Qt::LeftButton && selecting_) {
        selecting_ = false;
        sel_cur_   = e->pos();   // the release point, not the last move's — they can differ
        if (rows_ > 0 && cols_ > 0) {
            QRect pr = plotRect();
            auto toCell = [&](const QPoint& pt, int& r, int& c) {
                double fx = view_x0_ + (view_x1_ - view_x0_) * (pt.x() - pr.left()) / std::max(1, pr.width());
                double fy = view_y0_ + (view_y1_ - view_y0_) * (pt.y() - pr.top())  / std::max(1, pr.height());
                r = std::clamp(static_cast<int>(std::floor(fy)), 0, rows_ - 1);
                c = std::clamp(static_cast<int>(std::floor(fx)), 0, cols_ - 1);
            };
            int r0, c0, r1, c1;
            toCell(sel_press_, r0, c0);
            toCell(sel_cur_,   r1, c1);
            sel_r0_ = std::min(r0, r1); sel_r1_ = std::max(r0, r1);
            sel_c0_ = std::min(c0, c1); sel_c1_ = std::max(c0, c1);
            sel_valid_ = true;
            refreshSelectionRange();
            emit regionSelected(sel_r0_, sel_c0_, sel_r1_, sel_c1_);
        }
        update();
        return;
    }
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

    double Md_x = static_cast<double>(cols_);
    double Md_y = static_cast<double>(rows_);

    double factor = (e->angleDelta().y() > 0) ? 0.7 : 1.0 / 0.7;
    double span_x = std::clamp((view_x1_ - view_x0_) * factor, 2.0, Md_x);
    double span_y = std::clamp((view_y1_ - view_y0_) * factor, 2.0, Md_y);

    double frac_x = (pr.width()  > 0) ? (px - pr.left()) / pr.width()  : 0.5;
    double frac_y = (pr.height() > 0) ? (py - pr.top())  / pr.height() : 0.5;
    double nx0 = cx - frac_x * span_x;
    double ny0 = cy - frac_y * span_y;

    view_x0_ = std::clamp(nx0, 0.0, Md_x - span_x);
    view_x1_ = view_x0_ + span_x;
    if (view_x1_ > Md_x) { view_x1_ = Md_x; view_x0_ = std::max(0.0, Md_x - span_x); }

    view_y0_ = std::clamp(ny0, 0.0, Md_y - span_y);
    view_y1_ = view_y0_ + span_y;
    if (view_y1_ > Md_y) { view_y1_ = Md_y; view_y0_ = std::max(0.0, Md_y - span_y); }

    update();
}

void HeatmapWidget::resizeEvent(QResizeEvent*) { update(); }
