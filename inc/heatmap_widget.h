#pragma once

#include <QColor>
#include <QImage>
#include <QPoint>
#include <QString>
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
    // Render-time tone controls, applied to the normalised value after the
    // colour range and before the colour scheme lookup — so unlike gamma
    // (which rewrites display_matrix_) changing them costs nothing but a
    // repaint. contrast=1, brightness=0 is a no-op.
    void setContrast(float contrast);      // scales around mid-range
    void setBrightness(float brightness);  // shifts, in units of the full range

    // Compute a percentile-based colour range from the current display data.
    // Pass e.g. percentile=0.98 to saturate the top 2% ("auto-clip").
    void computeClipRange(float percentile, float& out_vmin, float& out_vmax) const;

    // Region selection. Shift+left-drag (or plain left-drag while select mode
    // is on) marks a rectangle of cells and emits regionSelected(). The point
    // is local contrast: computeClipRangeInSelection() derives a colour range
    // from just those cells, so the neighbourhood's own strongest correlations
    // span the full scale instead of being flattened by the global extremes
    // — and setDimOutsideSelection() darkens everything else so it stands out.
    void setSelectMode(bool on);
    bool hasSelection() const { return sel_valid_; }
    void clearSelection();
    // Region boost: cells inside the selection are colour-mapped against the
    // region's *own* value range (at setSelectionPercentile()'s clip) instead
    // of the global one, so the neighbourhood's strongest correlations light
    // up at full scale while the rest of the heatmap keeps its normal global
    // mapping. 0 = off, 1 = fully local; in between blends the two. The local
    // range is recomputed whenever the selection, percentile or processing
    // (abs/gamma/blur) changes, so it never goes stale.
    void setRegionBoost(float strength01);
    void setSelectionPercentile(float percentile);
    void setDimOutsideSelection(bool on);
    // computeClipRange() restricted to the selected cells; false if none.
    bool computeClipRangeInSelection(float percentile, float& out_vmin, float& out_vmax) const;

    void resetView();

    // --- Large-matrix support -------------------------------------------
    // How cells are combined when zoomed out far enough that one screen pixel
    // covers many of them. Nearest (the old behaviour) draws one arbitrary
    // cell per pixel — on a 30k×30k matrix that is ~900 cells per pixel, so a
    // 300×300 hotspot is mostly never sampled and simply vanishes from the
    // full view. Peak keeps the strongest-magnitude cell of each block, so
    // small hotspots stay visible at any zoom; Mean averages.
    enum class Downsample { Peak, Mean, Nearest };
    void setDownsample(Downsample d);

    // Coordinate navigation, for matrices far too large to drive by mouse
    // (one pixel spanning tens of cells makes a 300-cell drag impossible).
    void setView(double x0, double y0, double x1, double y1);    // clamped
    void setSelection(int row0, int col0, int row1, int col1);   // as if dragged
    void zoomToSelection(double margin_frac = 0.1);

    // --- Annotations -----------------------------------------------------
    // Persistent red boxes marking cells worth pointing at: a correlation
    // peak, a region a colleague should look at, the spot a figure is about.
    //
    // Deliberately separate from the region selection above. That selection is
    // a *tool* — it drives local contrast and zoom, there is only ever one,
    // and it moves the moment you drag again. These are a *record*: many at
    // once, they survive panning, zooming and every colour change, and they
    // are drawn into exportPng() so a figure carries its own markings instead
    // of needing them added afterwards in an image editor.
    //
    // Bounds are inclusive cell indices and are stored in data coordinates, so
    // a box stays on its cells no matter how the view moves.
    struct Annotation {
        int     row0 = 0, col0 = 0, row1 = 0, col1 = 0;
        QString label;                             // drawn above the box
        QColor  color = QColor(220, 40, 40);
    };

    void addAnnotation(const Annotation& a);
    void addAnnotation(int row0, int col0, int row1, int col1,
                        const QString& label = {},
                        const QColor& color = QColor(220, 40, 40));
    // Removes the topmost (most recently added) box containing that cell.
    // Returns false when no box covers it.
    bool removeAnnotationAt(int row, int col);
    void clearAnnotations();
    const std::vector<Annotation>& annotations() const { return annotations_; }

    // In annotate mode a left-drag draws a box instead of selecting a region,
    // and a right-click deletes the box under the cursor. Mutually exclusive
    // with setSelectMode() — turning one on turns the other off, since both
    // want the same left-drag.
    void setAnnotateMode(bool on);
    bool annotateMode() const { return annotate_mode_; }
    // Promotes the current region selection to a permanent box. No-op when
    // nothing is selected. Lives here rather than in the caller because the
    // selection bounds are private.
    bool annotateCurrentSelection(const QString& label = {});

    // The `count` strongest cells (by |value|), at most one per block×block
    // tile so they spread over distinct regions rather than one blob. For a
    // square matrix the main diagonal band is skipped — a self-correlation
    // matrix is trivially 1.0 there. Sorted strongest first.
    struct Hotspot { int row; int col; float value; };
    std::vector<Hotspot> findHotspots(int count, int block) const;

    // Export the current fully-rendered image to PNG, scaled to at most
    // max_pixels on each side (~300 DPI at 8").  Returns false on error.
    bool exportPng(const QString& path, int max_pixels = 2400) const;

    QSize sizeHint() const override { return {700, 640}; }

signals:
    // Emitted while mouse is inside the plot area.
    void hoverInfo(int s1, int s2, float value);
    // Emitted on release of a region drag; bounds are inclusive cell indices.
    void regionSelected(int row0, int col0, int row1, int col1);
    // Emitted whenever the annotation list changes, so a UI showing a count
    // or a list can refresh without polling.
    void annotationsChanged();

protected:
    void paintEvent(QPaintEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private:
    QRect plotRect() const;
    // Screen point -> inclusive cell index, clamped. Shared by the selection
    // drag, the annotation drag and the annotation hit-test so all three
    // agree on which cell the cursor is over.
    void  cellAt(const QPoint& pt, int& row, int& col) const;
    void  applyProcessing();          // raw → display_matrix_
    // Render the visible region of display_matrix_ into a W×H image.
    // Used by both paintEvent (viewport size) and exportPng (export size).
    //
    // pool_ref_w/h: the pixel size the *downsample factor* is derived from,
    // when it differs from the bitmap being drawn. Cells-per-pixel decides how
    // coarsely the matrix is pooled, so rendering the same view into a bigger
    // bitmap pools less — and a single-cell peak that Peak-pooling had
    // inflated into a visible block on screen collapses back to one pixel in
    // a 2400px export, which is exactly the mark the export was made to show.
    // exportPng therefore passes the on-screen plot size here, so the PNG is a
    // higher-resolution rendering of what is on screen rather than a
    // differently-pooled one. 0 = use W/H (the paintEvent case).
    QImage renderRegion(int W, int H,
                        double x0, double y0,
                        double x1, double y1,
                        int pool_ref_w = 0, int pool_ref_h = 0) const;
    QRgb  colormap(float v) const;

    // Draws every annotation box onto `p`, mapping data coordinates to
    // `target` through the view rectangle (x0,y0)-(x1,y1). Shared by
    // paintEvent and exportPng so the exported PNG carries exactly the boxes
    // that are on screen — drawing them only in paintEvent is the easy bug
    // here, and it produces a figure missing the very marks it was made for.
    // `scale` grows pen width and font for the (much larger) export bitmap.
    void drawAnnotations(QPainter& p, const QRectF& target,
                          double x0, double y0, double x1, double y1,
                          double scale) const;

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
    float       contrast_          = 1.0f;
    float       brightness_        = 0.0f;
    float       power_gamma_       = 1.0f;
    bool        threshold_enabled_ = false;
    float       threshold_value_   = 0.5f;
    ColorScheme color_scheme_      = ColorScheme::RdBu;

    // View: visible rectangle in matrix coordinates [0,M)
    double view_x0_ = 0.0, view_x1_ = 1.0;
    double view_y0_ = 0.0, view_y1_ = 1.0;

    // Region selection state
    bool   select_mode_ = false;
    bool   selecting_   = false;   // rubber band in progress
    QPoint sel_press_, sel_cur_;   // screen points of the drag
    bool   sel_valid_   = false;
    int    sel_r0_ = 0, sel_c0_ = 0, sel_r1_ = 0, sel_c1_ = 0;   // inclusive
    bool   dim_outside_ = false;

    // Annotations
    std::vector<Annotation> annotations_;
    bool   annotate_mode_ = false;
    Downsample downsample_ = Downsample::Peak;
    // Pooled copy of display_matrix_ for the current zoom level, built on
    // demand (ensurePooled) and dropped whenever display_matrix_ changes.
    // mutable: it is a render cache, filled from the const render path.
    mutable std::vector<float> pooled_;
    mutable int pool_factor_ = 0, pooled_rows_ = 0, pooled_cols_ = 0;
    void ensurePooled(int factor) const;
    void invalidatePooled() { pool_factor_ = 0; pooled_.clear(); }
    float  region_boost_    = 1.0f;
    float  sel_percentile_  = 0.98f;
    float  sel_lo_ = 0.0f, sel_hi_ = 1.0f;   // the region's own clip range
    void   refreshSelectionRange();
    QRgb   colormapT(float t) const;         // t already normalised to [0,1]

    // Pan state
    bool   dragging_     = false;
    QPoint drag_origin_;
    double drag_x0_     = 0.0;
    double drag_y0_     = 0.0;

    // Plot margins
    static constexpr int ML = 55, MR = 12, MT = 12, MB = 55;
};
