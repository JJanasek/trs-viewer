#pragma once
// Publication figures: one renderer for every plot in the app, drawing in
// physical units (points) so the same code produces a vector PDF, an SVG or
// a PNG at any resolution, sized and typeset to drop straight into a LaTeX
// paper, a thesis or a slide. The on-screen look (theme, widget size, k/M
// tick suffixes) deliberately plays no part.

#include <QColor>
#include <QRectF>
#include <QString>

#include <vector>

class QPainter;

// What to draw — extracted from a plot (see PlotWidget::figureData()).
struct FigureSeries {
    QString label;
    QColor  color;
    bool    filled  = false;        // draw as a filled band down to the axis
    bool    overlay = false;        // secondary curve (e.g. faded 1st order)
    // Either a polyline (x, y)…
    std::vector<double> x;
    std::vector<float>  y;
    // …or, when the data is denser than the figure can show, a min/max
    // envelope per column at positions x.
    std::vector<float>  ymin, ymax;
};

struct FigureBand {          // labelled region, e.g. a pattern
    double  x0 = 0, x1 = 0;
    QString label;
    QColor  color;
};

struct FigureData {
    QString title, x_label, y_label;
    double  x0 = 0, x1 = 1;  // axis range, in axis units
    double  y0 = 0, y1 = 1;
    bool    thresholds = false, one_sided = false;
    double  thr_pos = 4.5, thr_neg = -4.5;
    std::vector<FigureSeries> series;
    std::vector<FigureBand>   bands;
};

struct FigureStyle {
    double  width_cm  = 12.2;
    double  height_cm = 4.5;
    QString font_family = "Liberation Serif";     // an installed family — see resolveFontStyle()
    double  font_pt   = 8.0;
    double  line_pt   = 0.6;     // data lines
    bool    grid      = false;   // light major grid (published SCA plots mostly have none)
    bool    mono      = true;    // data in black/grey (print); false: keep the plot's colours
    bool    box       = true;    // frame on all four sides (false: left/bottom only)
    enum class Legend { Auto = 0, Inside = 1, Below = 2, Off = 3 };
    Legend  legend    = Legend::Auto;   // Auto: inside, unless > 6 entries (then off)
    bool    title     = false;   // figures get captions in LaTeX, not titles
    bool    transparent = false; // PNG/SVG: no background (slides)
    int     png_dpi   = 600;
};

// Font styles offered for figures; each resolves to the best installed family.
enum class FigureFont { LatexSerif = 0, Times = 1, Sans = 2 };
QString fontStyleName(FigureFont f);
QString resolveFontStyle(FigureFont f);      // installed family to use
bool    fontStyleIsExact(FigureFont f);      // true if the preferred family is installed

struct FigurePreset {
    QString name;
    double  width_cm, height_cm;
    FigureFont font;
    double  font_pt;
    double  line_pt;
    bool    mono;            // black data lines (papers) vs colour (slides)
    QString note;            // where it fits (shown as a tooltip)
};
const std::vector<FigurePreset>& figurePresets();

// "Nice" tick positions (1, 2, 5 × 10^k) covering [lo, hi], about `target` of them.
std::vector<double> niceTicks(double lo, double hi, int target);

// Draws the figure into `page`, in points (1 unit = 1/72 in), on any device.
void renderFigure(QPainter& p, const QRectF& page, const FigureData& d, const FigureStyle& s);

// The figure as an image, `scale` device pixels per point (preview, PNG, clipboard).
class QImage;
QImage renderFigureImage(const FigureData& d, const FigureStyle& s, double scale);

// Writes the figure; the format follows the file extension (.pdf, .svg, .png).
bool exportFigure(const QString& path, const FigureData& d, const FigureStyle& s, QString& error);
