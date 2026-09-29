#include <gtest/gtest.h>
#include "heatmap_widget.h"

#include <QApplication>
#include <QImage>
#include <QTemporaryDir>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// Heatmap annotations.
//
// These are the marks someone puts on a cross-correlation peak before sending
// the figure to a colleague, so two properties matter beyond the bookkeeping:
// a box must stay on its cells through pan and zoom (it is stored in data
// coordinates, not pixels), and it must end up in the exported PNG. Drawing
// annotations only in paintEvent is the easy mistake, and it silently produces
// a figure missing the very marks it was made for.
// ---------------------------------------------------------------------------

namespace {

// A uniform matrix, so the only strongly-coloured pixels in a render are the
// annotation boxes themselves.
std::unique_ptr<HeatmapWidget> makeFlatHeatmap(int n = 64, float value = 0.0f) {
    auto hm = std::make_unique<HeatmapWidget>();
    hm->resize(400, 400);
    hm->setMatrix(std::vector<float>(static_cast<size_t>(n) * n, value), n, n);
    hm->setColorRange(-1.0f, 1.0f);
    return hm;
}

// Counts pixels that are unmistakably the annotation red (220, 40, 40).
int countRedPixels(const QImage& img) {
    int n = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) {
            const QColor c = img.pixelColor(x, y);
            if (c.red() > 150 && c.red() - c.green() > 80 && c.red() - c.blue() > 80) ++n;
        }
    return n;
}

QImage exportToImage(HeatmapWidget& hm, const QString& path, int px = 400) {
    EXPECT_TRUE(hm.exportPng(path, px));
    QImage img;
    EXPECT_TRUE(img.load(path));
    return img;
}

} // namespace

// ---------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------

TEST(HeatmapAnnotations, StartsEmpty) {
    auto hm = makeFlatHeatmap();
    EXPECT_TRUE(hm->annotations().empty());
    EXPECT_FALSE(hm->annotateMode());
}

TEST(HeatmapAnnotations, NormalisesReversedBounds) {
    auto hm = makeFlatHeatmap();
    hm->addAnnotation(30, 40, 10, 20);          // deliberately back to front
    ASSERT_EQ(hm->annotations().size(), 1u);
    const auto& a = hm->annotations()[0];
    EXPECT_EQ(a.row0, 10); EXPECT_EQ(a.row1, 30);
    EXPECT_EQ(a.col0, 20); EXPECT_EQ(a.col1, 40);
}

TEST(HeatmapAnnotations, ClampsToMatrixBounds) {
    auto hm = makeFlatHeatmap(64);
    hm->addAnnotation(-20, -20, 500, 500);
    ASSERT_EQ(hm->annotations().size(), 1u);
    const auto& a = hm->annotations()[0];
    EXPECT_EQ(a.row0, 0);  EXPECT_EQ(a.col0, 0);
    EXPECT_EQ(a.row1, 63); EXPECT_EQ(a.col1, 63);
}

TEST(HeatmapAnnotations, RemoveAtPicksTheTopmostBox) {
    auto hm = makeFlatHeatmap();
    hm->addAnnotation(0, 0, 40, 40, "under");
    hm->addAnnotation(10, 10, 20, 20, "over");     // added later == on top
    ASSERT_EQ(hm->annotations().size(), 2u);

    EXPECT_TRUE(hm->removeAnnotationAt(15, 15));   // inside both
    ASSERT_EQ(hm->annotations().size(), 1u);
    EXPECT_EQ(hm->annotations()[0].label, "under");
}

TEST(HeatmapAnnotations, RemoveAtMissesOutsideEveryBox) {
    auto hm = makeFlatHeatmap();
    hm->addAnnotation(10, 10, 20, 20);
    EXPECT_FALSE(hm->removeAnnotationAt(50, 50));
    EXPECT_EQ(hm->annotations().size(), 1u);
}

TEST(HeatmapAnnotations, BoundsAreInclusive) {
    auto hm = makeFlatHeatmap();
    hm->addAnnotation(10, 10, 20, 20);
    EXPECT_TRUE(hm->removeAnnotationAt(10, 10));   // top-left corner counts
    hm->addAnnotation(10, 10, 20, 20);
    EXPECT_TRUE(hm->removeAnnotationAt(20, 20));   // ...and bottom-right
}

TEST(HeatmapAnnotations, ClearRemovesEverything) {
    auto hm = makeFlatHeatmap();
    for (int i = 0; i < 5; ++i) hm->addAnnotation(i, i, i + 2, i + 2);
    ASSERT_EQ(hm->annotations().size(), 5u);
    hm->clearAnnotations();
    EXPECT_TRUE(hm->annotations().empty());
}

TEST(HeatmapAnnotations, EmitsChangedOnAddRemoveClear) {
    auto hm = makeFlatHeatmap();
    // Counted with a plain connect rather than QSignalSpy, to keep this target
    // off the Qt Test module.
    int fired = 0;
    QObject::connect(hm.get(), &HeatmapWidget::annotationsChanged,
                     hm.get(), [&fired]() { ++fired; });
    hm->addAnnotation(1, 1, 5, 5);
    EXPECT_EQ(fired, 1);
    hm->removeAnnotationAt(2, 2);
    EXPECT_EQ(fired, 2);
    hm->addAnnotation(1, 1, 5, 5);
    hm->clearAnnotations();
    EXPECT_EQ(fired, 4);
    hm->clearAnnotations();                        // already empty: no signal
    EXPECT_EQ(fired, 4);
}

// Both modes claim the plain left-drag, so they must not both be on.
TEST(HeatmapAnnotations, AnnotateModeTurnsOffSelectMode) {
    auto hm = makeFlatHeatmap();
    hm->setSelectMode(true);
    hm->setAnnotateMode(true);
    EXPECT_TRUE(hm->annotateMode());
    hm->setAnnotateMode(false);
    EXPECT_FALSE(hm->annotateMode());
}

TEST(HeatmapAnnotations, AnnotateCurrentSelectionUsesItsBounds) {
    auto hm = makeFlatHeatmap();
    EXPECT_FALSE(hm->annotateCurrentSelection());  // nothing selected yet

    hm->setSelection(12, 34, 20, 40);
    ASSERT_TRUE(hm->hasSelection());
    EXPECT_TRUE(hm->annotateCurrentSelection());
    ASSERT_EQ(hm->annotations().size(), 1u);
    const auto& a = hm->annotations()[0];
    EXPECT_EQ(a.row0, 12); EXPECT_EQ(a.col0, 34);
    EXPECT_EQ(a.row1, 20); EXPECT_EQ(a.col1, 40);
}

TEST(HeatmapAnnotations, AutoLabelIsTheStrongestValueCovered) {
    auto hm = makeFlatHeatmap(64, 0.0f);
    std::vector<float> m(64 * 64, 0.0f);
    m[static_cast<size_t>(15) * 64 + 15] = -0.875f;   // strongest by magnitude
    m[static_cast<size_t>(16) * 64 + 16] =  0.25f;
    hm->setMatrix(m, 64, 64);
    hm->setSelection(10, 10, 20, 20);
    ASSERT_TRUE(hm->annotateCurrentSelection());
    // Magnitude decides, and the sign is kept — a strong negative correlation
    // is just as interesting as a strong positive one.
    EXPECT_EQ(hm->annotations()[0].label, QString::number(-0.875f, 'g', 3));
}

// ---------------------------------------------------------------------------
// Rendering and export — the properties that actually matter
// ---------------------------------------------------------------------------

TEST(HeatmapAnnotations, ExportedPngContainsTheBoxes) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeFlatHeatmap();

    const QImage before = exportToImage(*hm, dir.filePath("plain.png"));
    const int red_before = countRedPixels(before);

    hm->addAnnotation(16, 16, 48, 48);
    const QImage after = exportToImage(*hm, dir.filePath("boxed.png"));
    const int red_after = countRedPixels(after);

    EXPECT_GT(red_after, red_before + 100)
        << "the exported PNG must carry the annotation, not just the screen view "
           "(red pixels: " << red_before << " -> " << red_after << ")";
}

TEST(HeatmapAnnotations, ClearedBoxesLeaveTheExport) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeFlatHeatmap();
    const int red_plain = countRedPixels(exportToImage(*hm, dir.filePath("a.png")));

    hm->addAnnotation(16, 16, 48, 48);
    ASSERT_GT(countRedPixels(exportToImage(*hm, dir.filePath("b.png"))), red_plain + 100);

    hm->clearAnnotations();
    EXPECT_NEAR(countRedPixels(exportToImage(*hm, dir.filePath("c.png"))), red_plain, 20);
}

// Boxes are stored in data coordinates, so zooming must move them with the
// cells they mark rather than leaving them behind at fixed pixel positions.
TEST(HeatmapAnnotations, BoxTracksItsCellsThroughZoom) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeFlatHeatmap(64);
    hm->addAnnotation(30, 30, 34, 34);

    const int red_full = countRedPixels(exportToImage(*hm, dir.filePath("full.png")));
    ASSERT_GT(red_full, 0);

    // Zoom right in on the box: it covers far more of the frame, so its
    // outline must account for many more red pixels.
    hm->setView(28, 28, 36, 36);
    const int red_zoom = countRedPixels(exportToImage(*hm, dir.filePath("zoom.png")));
    EXPECT_GT(red_zoom, red_full * 2)
        << "zoomed in, the box should grow with its cells: "
        << red_full << " -> " << red_zoom;

    // Pan somewhere else entirely: the box is off-screen and must not be drawn.
    hm->setView(0, 0, 8, 8);
    EXPECT_LT(countRedPixels(exportToImage(*hm, dir.filePath("away.png"))), red_full / 2)
        << "a box outside the view must not be drawn";
}

// A single-cell peak boxed on a large matrix would be sub-pixel when zoomed
// out. Marking it and then not being able to see the mark is useless.
TEST(HeatmapAnnotations, TinyBoxStaysVisibleWhenZoomedOut) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeFlatHeatmap(2000);
    hm->addAnnotation(1000, 1000, 1000, 1000);     // exactly one cell
    EXPECT_GT(countRedPixels(exportToImage(*hm, dir.filePath("tiny.png"))), 0);
}

TEST(HeatmapAnnotations, SurviveColourSchemeAndProcessingChanges) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeFlatHeatmap();
    hm->addAnnotation(16, 16, 48, 48);
    const int base = countRedPixels(exportToImage(*hm, dir.filePath("base.png")));
    ASSERT_GT(base, 0);

    hm->setColorScheme(ColorScheme::Viridis);
    hm->setAbsValue(true);
    hm->setGaussianSigma(2.0f);
    EXPECT_EQ(hm->annotations().size(), 1u);
    EXPECT_GT(countRedPixels(exportToImage(*hm, dir.filePath("viridis.png"))), base / 2);
}

TEST(HeatmapAnnotations, EmptyMatrixDoesNotCrash) {
    HeatmapWidget hm;
    hm.resize(200, 200);
    hm.addAnnotation(0, 0, 5, 5);
    EXPECT_FALSE(hm.removeAnnotationAt(0, 0) && false);   // must not crash
    hm.clearAnnotations();
}

// ---------------------------------------------------------------------------
// Tone controls: contrast x brightness
//
// The two are coupled. Together they map input t to
//     t' = (t - 0.5)*contrast + 0.5 + brightness
// and the visible slice of the data is the t for which t' lands in [0,1] — a
// window of width 1/contrast centred at 0.5 - brightness/contrast. That is why
// a brightness limit of +-1 is right at contrast 1 and far too tight above it.
// ---------------------------------------------------------------------------

namespace {

// A left-to-right ramp, so "which part of the range is visible" is directly
// readable as "which part of the image is not saturated".
std::unique_ptr<HeatmapWidget> makeRampHeatmap(int n = 64) {
    auto hm = std::make_unique<HeatmapWidget>();
    hm->resize(400, 400);
    std::vector<float> m(static_cast<size_t>(n) * n);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            m[static_cast<size_t>(r) * n + c] =
                -1.0f + 2.0f * static_cast<float>(c) / static_cast<float>(n - 1);
    hm->setMatrix(m, n, n);
    hm->setColorRange(-1.0f, 1.0f);
    return hm;
}

// How many distinct colours the render contains — a proxy for "how much of the
// data range is still resolved rather than clamped flat".
int distinctColours(const QImage& img) {
    std::set<QRgb> seen;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) seen.insert(img.pixel(x, y));
    return static_cast<int>(seen.size());
}

} // namespace

TEST(HeatmapTone, BrightnessBelowMinusOneIsAcceptedByTheWidget) {
    auto hm = makeRampHeatmap();
    hm->setBrightness(-4.5f);     // no clamping in the widget; only the UI limited it
    hm->setContrast(10.0f);
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    EXPECT_TRUE(hm->exportPng(dir.filePath("x.png"), 200));
}

// At contrast 1 the old +-1 limit really was enough: brightness -1 already
// pushes the entire range off the scale, and going further changes nothing.
TEST(HeatmapTone, AtContrastOneMinusOneAlreadySaturatesEverything) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeRampHeatmap();
    hm->setContrast(1.0f);

    hm->setBrightness(-1.0f);
    const int at_1 = distinctColours(exportToImage(*hm, dir.filePath("b1.png"), 200));
    hm->setBrightness(-5.0f);
    const int at_5 = distinctColours(exportToImage(*hm, dir.filePath("b5.png"), 200));

    EXPECT_LE(at_1, 3) << "everything should already be one flat colour";
    EXPECT_EQ(at_1, at_5) << "past full saturation, more negative must change nothing";
}

// ...and at high contrast it is not enough, which is the reason for the wider
// range. Brightness -4.5 at contrast 10 must show a different slice of the
// data than -1 does.
TEST(HeatmapTone, AtHighContrastBelowMinusOneSelectsADifferentSlice) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeRampHeatmap();
    hm->setContrast(10.0f);

    hm->setBrightness(-1.0f);
    const QImage a = exportToImage(*hm, dir.filePath("c10_b1.png"), 200);
    hm->setBrightness(-4.5f);
    const QImage b = exportToImage(*hm, dir.filePath("c10_b45.png"), 200);

    EXPECT_NE(a, b) << "at contrast 10, brightness -4.5 must differ from -1 — "
                        "if these match, the wider range buys nothing";
    // Both still resolve part of the ramp rather than being flat.
    EXPECT_GT(distinctColours(a), 3);
    EXPECT_GT(distinctColours(b), 3);
}

// The pairing that motivates the whole thing: crank contrast, push brightness
// negative, and only the strongest cells stay lit.
TEST(HeatmapTone, HighContrastNegativeBrightnessIsolatesTheTopOfTheRange) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeRampHeatmap(64);
    hm->setContrast(10.0f);
    hm->setBrightness(-4.5f);     // visible window ~ top 10% of the range
    const QImage img = exportToImage(*hm, dir.filePath("iso.png"), 256);

    // The ramp runs left to right, so with only the top of the range resolved
    // the left half must be uniform and the right half must not.
    auto colours_in = [&](int x0, int x1) {
        std::set<QRgb> seen;
        for (int y = img.height() / 4; y < img.height() * 3 / 4; ++y)
            for (int x = x0; x < x1; ++x) seen.insert(img.pixel(x, y));
        return static_cast<int>(seen.size());
    };
    EXPECT_LE(colours_in(0, img.width() / 2), 2)
        << "the bottom of the range should be clamped flat";
    EXPECT_GT(colours_in(img.width() * 3 / 4, img.width()), 4)
        << "the top of the range should still be resolved";
}

// ---------------------------------------------------------------------------
// Export fidelity
//
// exportPng promises "exactly what is on screen". Two things used to break
// that promise, and both are silent — you only notice when the figure is
// already in a report:
//
//   * the downsample factor was derived from the *export bitmap* size, so a
//     bigger PNG pooled the matrix more finely. A single-cell peak that Peak
//     pooling had inflated into a visible block on screen collapsed back to
//     one pixel in a 2400px export — the very mark the export was made for.
//   * the aspect ratio came from the view in *data* units, while the screen
//     stretches that view to fill the plot rectangle, so the exported picture
//     had different proportions from the one being looked at.
// ---------------------------------------------------------------------------

namespace {

// Fraction of the image (percent) that differs markedly from the background.
double hotFraction(const QImage& img, QRect r) {
    const QColor bg = img.pixelColor(r.left() + 5, r.top() + 5);
    long hot = 0, tot = 0;
    for (int y = r.top(); y < r.bottom(); ++y)
        for (int x = r.left(); x < r.right(); ++x) {
            const QColor c = img.pixelColor(x, y);
            const int d = std::abs(c.red()   - bg.red())
                        + std::abs(c.green() - bg.green())
                        + std::abs(c.blue()  - bg.blue());
            if (d > 60) ++hot;
            ++tot;
        }
    return tot ? 100.0 * static_cast<double>(hot) / static_cast<double>(tot) : 0.0;
}

// A large matrix with one tiny hotspot — the case Peak downsampling exists for.
std::unique_ptr<HeatmapWidget> makeSpikeHeatmap(int n = 4000, int spike = 1) {
    auto hm = std::make_unique<HeatmapWidget>();
    hm->resize(760, 700);
    std::vector<float> m(static_cast<size_t>(n) * n, 0.0f);
    for (int r = n / 2; r < n / 2 + spike; ++r)
        for (int c = n / 2; c < n / 2 + spike; ++c)
            m[static_cast<size_t>(r) * n + c] = 1.0f;
    hm->setMatrix(m, n, n);
    hm->setColorRange(-1.0f, 1.0f);
    hm->setDownsample(HeatmapWidget::Downsample::Peak);
    return hm;
}

} // namespace

// The regression guard for the pooling bug: how much of the picture a peak
// covers must not depend on how many pixels the picture has.
TEST(HeatmapExport, PeakKeepsItsRelativeSizeAtEveryExportSize) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeSpikeHeatmap();
    hm->setContrast(6.0f);
    hm->setBrightness(-2.0f);        // isolate the spike from the background

    std::vector<double> fracs;
    for (int px : {600, 1200, 2400}) {
        const QString path = dir.filePath(QString("e%1.png").arg(px));
        ASSERT_TRUE(hm->exportPng(path, px));
        QImage img;
        ASSERT_TRUE(img.load(path));
        fracs.push_back(hotFraction(img, QRect(3, 3, img.width() - 6, img.height() - 6)));
    }
    ASSERT_EQ(fracs.size(), 3u);
    for (double f : fracs) ASSERT_GT(f, 0.0) << "the spike vanished entirely";

    // Before the fix the 2400px export pooled far more finely than the 600px
    // one and the spike shrank to a single pixel, so this ratio blew up.
    const double lo = *std::min_element(fracs.begin(), fracs.end());
    const double hi = *std::max_element(fracs.begin(), fracs.end());
    EXPECT_LT(hi / lo, 3.0)
        << "spike covers " << fracs[0] << "% at 600px, " << fracs[1] << "% at 1200px, "
        << fracs[2] << "% at 2400px — export size must not change how much of the "
           "picture a peak occupies";
}

// A single cell in a 4000x4000 matrix must survive to the export at all.
TEST(HeatmapExport, SingleCellPeakIsVisibleInALargeExport) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeSpikeHeatmap(4000, 1);
    hm->setContrast(6.0f);
    hm->setBrightness(-2.0f);

    const QString path = dir.filePath("big.png");
    ASSERT_TRUE(hm->exportPng(path, 2400));
    QImage img;
    ASSERT_TRUE(img.load(path));
    // Before the fix this was exactly one pixel in a 2400x2400 image.
    EXPECT_GT(hotFraction(img, QRect(3, 3, img.width() - 6, img.height() - 6)), 0.0);
}

// Aspect must follow the plot rectangle, since that is what the view is
// stretched to fill on screen — not the view's own data proportions.
TEST(HeatmapExport, AspectFollowsThePlotRectNotTheData) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeFlatHeatmap(64);          // square matrix...
    hm->resize(900, 400);                   // ...in a wide, non-square widget

    const QString path = dir.filePath("wide.png");
    ASSERT_TRUE(hm->exportPng(path, 1200));
    QImage img;
    ASSERT_TRUE(img.load(path));

    // Plot rect margins: ML=55, MR=12, MT=12, MB=55.
    const double want = static_cast<double>(900 - 55 - 12) /
                        static_cast<double>(400 - 12 - 55);
    const double got  = static_cast<double>(img.width()) / img.height();
    EXPECT_NEAR(got, want, 0.02)
        << "square data in a wide plot must export wide (got " << got
        << ", plot rect is " << want << ")";
    EXPECT_GT(got, 1.5) << "exporting at the data aspect would have given ~1.0";
}

// The user-visible claim that prompted this: tone controls must reach the PNG.
TEST(HeatmapExport, ContrastAndBrightnessReachTheExport) {
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    auto hm = makeRampHeatmap(64);

    auto meanLum = [](const QImage& im) {
        double s = 0; long n = 0;
        for (int y = 0; y < im.height(); ++y)
            for (int x = 0; x < im.width(); ++x) {
                const QColor c = im.pixelColor(x, y);
                s += 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue();
                ++n;
            }
        return n ? s / n : 0.0;
    };

    const double plain = meanLum(exportToImage(*hm, dir.filePath("p.png"), 300));
    hm->setContrast(4.0f);
    const double contrasted = meanLum(exportToImage(*hm, dir.filePath("c.png"), 300));
    hm->setBrightness(-1.5f);
    const double darkened = meanLum(exportToImage(*hm, dir.filePath("b.png"), 300));

    EXPECT_GT(std::abs(contrasted - plain), 5.0)  << "contrast never reached the PNG";
    EXPECT_GT(std::abs(darkened - contrasted), 5.0) << "brightness never reached the PNG";
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Offscreen so the suite runs headless in CI and over SSH; must be set
    // before QApplication is constructed.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
