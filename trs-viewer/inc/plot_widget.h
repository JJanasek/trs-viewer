#pragma once

#include "trs_file.h"
#include "processing.h"

#include <QColor>
#include <QPoint>
#include <QWidget>

#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
struct PlotTheme {
    QColor bg_outer;    // widget background outside the plot area
    QColor bg_plot;     // plot area background
    QColor grid;        // grid lines
    QColor border;      // plot border
    QColor axis_text;   // tick labels
    QColor legend_text; // legend text

    static PlotTheme dark();
    static PlotTheme light();
};

// ---------------------------------------------------------------------------
// Interaction mode
// ---------------------------------------------------------------------------
enum class InteractionMode {
    Pan,        // left-drag to pan, scroll wheel to zoom (default)
    Measure,    // left-click sets P1 then P2; third click resets
    BoxZoom,    // left-drag to rubber-band zoom
    CropSelect, // left-drag to add a sample range to the crop list
};

// ---------------------------------------------------------------------------
// Per-trace render cache (rebuilt when view or widget width changes)
// ---------------------------------------------------------------------------
struct TraceCache {
    // Key — what the cache was built for
    int64_t view_start = -1;
    int64_t view_end   = -1;
    int     W          =  0;
    bool    valid      = false;

    // Zoomed-in path (view_len <= W): processed sample array
    std::vector<float> samples;   // length = out_count after pipeline
    int64_t            raw_read = 0;

    // Zoomed-out path (view_len > W): per-pixel min/max aggregates
    std::vector<float> pix_min;
    std::vector<float> pix_max;

    // Data range (no padding), used by computeYRange
    float ymin = 0;
    float ymax = 0;
};

// ---------------------------------------------------------------------------
// Trace entry
// ---------------------------------------------------------------------------
struct TraceEntry {
    TrsFile*  file      = nullptr;
    int32_t   trace_idx = 0;
    QColor    color;
    QString   label;
    bool      visible   = true;
    std::shared_ptr<std::vector<float>> mem_data; // non-null for in-memory traces
    std::vector<std::shared_ptr<ITransform>> transforms;
    TraceCache cache;
};

// ---------------------------------------------------------------------------
// PlotWidget
// ---------------------------------------------------------------------------
class PlotWidget : public QWidget {
    Q_OBJECT
public:
    explicit PlotWidget(QWidget* parent = nullptr);

    void addTrace(TrsFile* file, int32_t trace_idx,
                  QColor color, const QString& label = {});
    void addTrace(std::shared_ptr<std::vector<float>> data,
                  QColor color, const QString& label = {});
    void clearTraces();
    void setTransforms(const std::vector<std::shared_ptr<ITransform>>& tx);

    void setTheme(const PlotTheme& theme);
    void setMode(InteractionMode mode);
    InteractionMode mode() const { return mode_; }

    void resetView();       // resets view AND clears measurement points
    void clearMeasurement();
    void zoomIn();
    void zoomOut();
    // Y-axis (amplitude) zoom.  Also triggered by Ctrl+scroll.
    // y_scale_ < 1 → taller traces; y_scale_ > 1 → shorter traces.
    void zoomInY();         // shrink y range → taller traces
    void zoomOutY();        // expand y range → shorter traces
    void resetYZoom();
    float yScale() const { return y_scale_; }
    void setThresholds(bool show, double pos = 4.5, double neg = -4.5);
    // When one_sided=true only the positive threshold line is drawn (use after abs() preprocessing).
    void setThresholdOneSided(bool one_sided);
    // Optional title rendered above the plot area (empty = no title, no extra margin).
    void setTitle(const QString& title);
    // Trace line width in pixels (default 1.5).
    void setTraceWidth(float w);
    // Change the color of an already-added trace by index.
    void setTraceColor(int idx, const QColor& c);

    // View accessors
    int64_t viewStart()     const { return view_start_; }
    int64_t viewEnd()       const { return view_end_;   }
    int64_t totalSamples()  const { return total_samples_; }

    // Crop ranges (used by CropSelect mode and crop dialog)
    void    addCropRange(int64_t start, int64_t end);
    void    removeCropRangeAt(int idx);
    void    clearCropRanges();
    const   std::vector<std::pair<int64_t,int64_t>>& cropRanges() const { return crop_ranges_; }

signals:
    void viewChanged(int64_t view_start, int64_t view_end, int64_t total_samples);
    // Emitted whenever measurement points change.
    // has_p2=false means only P1 is set.
    void measurementUpdated(int64_t s1, double v1,
                             int64_t s2, double v2, bool has_p2);
    // Emitted whenever the crop ranges list changes
    void cropRangesChanged();

protected:
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private:
    QRect   plotRect() const;
    // Build (or reuse) the per-pixel cache for one trace.
    void    buildTraceCache(TraceEntry& te, int W);
    void    renderTrace(const TraceEntry& te, QPainter& p,
                        const QRect& pr, float ymin, float ymax);
    void    computeYRange(float& ymin, float& ymax);
    void    drawMeasurement(QPainter& p, const QRect& pr,
                             float ymin, float ymax);

    int64_t pixelToSample(int px,  const QRect& pr) const;
    int     sampleToPixel(int64_t s, const QRect& pr) const;
    int     valueToPixel(float v, const QRect& pr, float ymin, float ymax) const;
    double  pixelToValue(int py,  const QRect& pr, float ymin, float ymax) const;

    int64_t readTraceSamples(const TraceEntry& te,
                              int64_t sample_offset, int64_t count,
                              float* buf) const;

    // View state
    int64_t view_start_    = 0;
    int64_t view_end_      = 0;
    int64_t total_samples_ = 0;

    // Traces & pipeline
    std::vector<TraceEntry>                  traces_;
    std::vector<std::shared_ptr<ITransform>> transforms_;

    // Theme
    PlotTheme theme_ = PlotTheme::dark();

    // Interaction
    InteractionMode mode_ = InteractionMode::Pan;

    // Pan state
    bool    dragging_            = false;
    QPoint  drag_origin_;
    int64_t drag_start_at_press_ = 0;
    int64_t drag_end_at_press_   = 0;

    // BoxZoom / CropSelect rubber-band state (shared)
    bool   rubber_band_active_  = false;
    QPoint rubber_band_start_;
    QPoint rubber_band_current_;

    // Crop ranges
    std::vector<std::pair<int64_t,int64_t>> crop_ranges_;

    // Measurement state (up to 2 points)
    int     meas_count_ = 0;   // 0, 1, or 2
    int64_t meas_s1_ = 0, meas_s2_ = 0;
    double  meas_v1_ = 0, meas_v2_ = 0;

    // Cached y-range from last paint (needed to map click y → value)
    float last_ymin_ = -1.0f;
    float last_ymax_ =  1.0f;

    // Y-axis zoom: multiplier on the auto-fit half-range.
    // 1.0 = auto-fit, >1 = zoomed out (more space), <1 = zoomed in.
    // Ctrl+scroll adjusts this; resetView() restores it to 1.0.
    float y_scale_ = 1.0f;

    bool   show_thresholds_      = false;
    bool   threshold_one_sided_  = false;
    double threshold_pos_        =  4.5;
    double threshold_neg_        = -4.5;

    // Visual style
    QString plot_title_;
    float   trace_width_ = 1.5f;

    // Plot area margins (pixels)
    static constexpr int ML      = 65;
    static constexpr int MR      = 12;
    static constexpr int MT      = 12;
    static constexpr int MB      = 36;
    static constexpr int MT_TITLE = 30;  // top margin when title is set
};
