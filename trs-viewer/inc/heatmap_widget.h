#pragma once

#include <QImage>
#include <QPoint>
#include <QWidget>

#include <vector>

enum class ColorScheme { RdBu, Grayscale, Hot, Viridis, Plasma, Lukasz };

// Displays a square M×M float matrix as a 2D false-colour heatmap.
// Supports pan (left-drag) and zoom (scroll wheel), adjustable colour range,
// and PNG export.
class HeatmapWidget : public QWidget {
    Q_OBJECT
public:
    explicit HeatmapWidget(QWidget* parent = nullptr);

    // Load rows×cols row-major matrix.  Resets view to full extent.
    void setMatrix(const std::vector<float>& data, int32_t rows, int32_t cols);
    // Convenience: square M×M matrix.
    void setMatrix(const std::vector<float>& data, int32_t M) { setMatrix(data, M, M); }

    // Update colour-map range; rebuilds the colour image.
    void setColorRange(float vmin, float vmax);

    // Post-processing applied before colour-mapping (in this order):
    //   1. Gaussian blur  (sigma=0 → off)
    //   2. Absolute value (off by default)
    //   3. Power/gamma    (gamma=1 → off; gamma=2 → square)
    //   4. Binary threshold (off by default)
    void setGaussianSigma(float sigma);
    void setAbsValue(bool enabled);
    void setPowerGamma(float gamma);
    void setBinaryThreshold(bool enabled, float threshold = 0.5f);
    void setColorScheme(ColorScheme scheme);

    // Compute a percentile-based colour range from the current display data.
    // Pass e.g. percentile=0.98 to saturate the top 2% ("auto-clip").
    void computeClipRange(float percentile, float& out_vmin, float& out_vmax) const;

    void resetView();

    // Export the current fully-rendered image to PNG, scaled to at most
    // max_pixels on each side (~300 DPI at 8").  Returns false on error.
    bool exportPng(const QString& path, int max_pixels = 2400) const;

    QSize sizeHint() const override { return {700, 640}; }

signals:
    // Emitted while mouse is inside the plot area.
    void hoverInfo(int s1, int s2, float value);

protected:
    void paintEvent(QPaintEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private:
    QRect plotRect() const;
    void  applyProcessing();          // raw → display_matrix_
    // Render the visible region of display_matrix_ into a W×H image.
    // Used by both paintEvent (viewport size) and exportPng (export size).
    QImage renderRegion(int W, int H,
                        double x0, double y0,
                        double x1, double y1) const;
    QRgb  colormap(float v) const;

    // Data
    std::vector<float> matrix_;          // raw data
    std::vector<float> display_matrix_;  // after all processing steps
    int32_t  rows_ = 0;
    int32_t  cols_ = 0;
    float    vmin_ = -1.0f;
    float    vmax_ =  1.0f;

    // Processing parameters
    float       gaussian_sigma_    = 0.0f;
    bool        abs_value_         = false;
    float       power_gamma_       = 1.0f;
    bool        threshold_enabled_ = false;
    float       threshold_value_   = 0.5f;
    ColorScheme color_scheme_      = ColorScheme::RdBu;

    // View: visible rectangle in matrix coordinates [0,M)
    double view_x0_ = 0.0, view_x1_ = 1.0;
    double view_y0_ = 0.0, view_y1_ = 1.0;

    // Pan state
    bool   dragging_     = false;
    QPoint drag_origin_;
    double drag_x0_     = 0.0;
    double drag_y0_     = 0.0;

    // Plot margins
    static constexpr int ML = 55, MR = 12, MT = 12, MB = 55;
};
