// GUI soak test: the whole app, offscreen, driven through its own slots and
// dialogs the way a user would, one scenario per run:
//
//   gui_soak --scenario <name> [--loops N]
//
// Each scenario repeats N times on test files generated here, then the app
// quits normally (destructors, job shutdown and all). It fails on a crash,
// an unexpected message box, a step that times out, or memory that keeps
// growing after the first loop (RSS after loop N vs after loop 1). Run it
// under the memory tools with tools/run_asan.sh, run_valgrind.sh and
// run_heaptrack.sh; ctest runs every scenario with 3 loops.
//
// The driver is a 20 ms timer that (1) answers dialogs by their title, in
// the order the scenario expects them, and (2) runs the scenario's steps one
// at a time; a step returns false to be polled again (waiting for a job, say).
// Steps may block in a modal exec() — the timer keeps answering dialogs from
// inside that nested loop, and holds the next step until it returns.

#include "mainwindow.h"
#include "plot_widget.h"
#include "job_manager.h"
#include "leakage_model.h"

#include <QtWidgets>

#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <random>

#include <malloc.h>
#include <unistd.h>

namespace {

// ---- test data -----------------------------------------------------------

// N float traces of NS samples at 1 GS/s; per-trace data: [class][16 bytes].
// A pulse, a little class-dependent leakage, a locked tone and noise.
void writeTrs(const QString& path, int n, int ns, uint32_t seed) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) { std::fprintf(stderr, "cannot write %s\n", qPrintable(path)); std::exit(2); }
    auto tag = [&](uint8_t t, const QByteArray& v) {
        f.putChar(static_cast<char>(t));
        f.putChar(static_cast<char>(v.size()));
        f.write(v);
    };
    auto le = [](uint32_t v, int bytes) {
        QByteArray b;
        for (int i = 0; i < bytes; i++) b.append(static_cast<char>((v >> (8 * i)) & 0xFF));
        return b;
    };
    float scale_x = 1e-9f;
    uint32_t sx;
    std::memcpy(&sx, &scale_x, 4);
    tag(0x41, le(static_cast<uint32_t>(n), 4));
    tag(0x42, le(static_cast<uint32_t>(ns), 4));
    tag(0x43, le(0x14, 1));                 // float samples
    tag(0x44, le(17, 2));                   // data bytes per trace
    tag(0x4B, le(sx, 4));                   // scale_x
    f.putChar(0x5F);
    f.putChar(0);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(ns));
    for (int t = 0; t < n; t++) {
        QByteArray data;
        const int cls = static_cast<int>(rng() & 1);
        data.append(static_cast<char>(cls));
        for (int i = 0; i < 16; i++) data.append(static_cast<char>(rng() & 0xFF));
        for (int s = 0; s < ns; s++) {
            x[static_cast<size_t>(s)] = noise(rng) + 2.0f * std::sin(0.5f * static_cast<float>(M_PI) * s)
                                      + (s >= ns / 4 && s < ns / 4 + 50 ? 3.0f : 0.0f)
                                      + (s >= ns / 2 && s < ns / 2 + 10 ? 0.4f * cls : 0.0f);
        }
        f.write(data);
        f.write(reinterpret_cast<const char*>(x.data()), static_cast<qint64>(x.size() * sizeof(float)));
    }
}

double rssMB() {
    malloc_trim(0);   // hand freed memory back, so RSS reflects what is still in use
    QFile f("/proc/self/statm");
    if (!f.open(QIODevice::ReadOnly)) return 0;
    const QList<QByteArray> v = f.readAll().split(' ');
    return v.size() > 1 ? v[1].toDouble() * static_cast<double>(sysconf(_SC_PAGESIZE)) / 1048576.0 : 0;
}

// ---- driver --------------------------------------------------------------

class Driver {
public:
    using Step    = std::function<bool()>;          // true = done
    using Handler = std::function<bool(QDialog*)>;  // true = done with this dialog

    void step(Step s) { steps_.push_back(std::move(s)); }
    void then(std::function<void()> f) { step([f]() { f(); return true; }); }
    // From this point in the steps on, the next dialog whose title starts
    // with `title` goes to h (registered when the steps get here, so
    // dialogsAnswered() only counts dialogs the current steps expect).
    void onDialog(const QString& title, Handler h) {
        then([this, title, h]() { handlers_.push_back({title, h}); });
    }

    int run(int timeout_ms) {
        QTimer tick;
        QObject::connect(&tick, &QTimer::timeout, [this]() { pump(); });
        tick.start(20);
        QTimer::singleShot(timeout_ms, [this]() {
            fail(QString("timed out in step %1 (%2 handlers still waiting%3)")
                     .arg(done_steps_).arg(handlers_.size())
                     .arg(handlers_.empty() ? QString() : ", next: " + handlers_.front().title));
            QApplication::exit(2);
        });
        return QApplication::exec();
    }

    void fail(const QString& why) {
        std::fprintf(stderr, "FAIL: %s\n", qPrintable(why));
        failures_++;
    }
    int failures() const { return failures_; }
    bool dialogsAnswered() const { return handlers_.empty(); }

private:
    struct Pending { QString title; Handler h; };

    void pump() {
        answerDialogs();
        if (in_step_ || step_queued_) return;
        if (steps_.empty()) {
            if (!quitting_) {
                quitting_ = true;
                if (!handlers_.empty())
                    fail(QString("dialog never appeared: '%1'").arg(handlers_.front().title));
                QApplication::exit(failures_ ? 1 : 0);
            }
            return;
        }
        // Steps run from their own single-shot timer: Qt doesn't fire a timer
        // again while its handler is still running, so a step blocked in a
        // modal exec() here would stop this pump answering its dialog.
        step_queued_ = true;
        QTimer::singleShot(0, [this]() { runStep(); });
    }

    void runStep() {
        step_queued_ = false;
        if (steps_.empty()) return;
        if (!announced_) {
            std::fprintf(stderr, "  step %d\n", done_steps_);
            announced_ = true;
        }
        in_step_ = true;
        const bool done = steps_.front()();
        in_step_ = false;
        if (done) { steps_.pop_front(); done_steps_++; announced_ = false; }
    }

    void answerDialogs() {
        for (QWidget* w : QApplication::topLevelWidgets()) {
            auto* d = qobject_cast<QDialog*>(w);
            if (!d || !d->isVisible()) continue;
            if (!handlers_.empty() && d->windowTitle().startsWith(handlers_.front().title)) {
                if (handlers_.front().h(d)) {
                    std::fprintf(stderr, "  answered '%s'\n", qPrintable(d->windowTitle()));
                    handlers_.pop_front();
                }
                return;   // one action per tick; the dialog reacts first
            }
            if (auto* mb = qobject_cast<QMessageBox*>(d)) {
                // Marked on the box itself: a new box can reuse a freed one's address.
                if (mb->property("soak_answered").toBool()) continue;
                mb->setProperty("soak_answered", true);
                fail(QString("unexpected message box '%1': %2").arg(mb->windowTitle(), mb->text().left(200)));
                QMetaObject::invokeMethod(mb, "reject", Qt::QueuedConnection);
            }
        }
    }

    std::deque<Step>    steps_;
    std::deque<Pending> handlers_;
    bool in_step_  = false;
    bool step_queued_ = false;
    bool announced_ = false;
    bool quitting_ = false;
    int  done_steps_ = 0;
    int  failures_ = 0;
};

// ---- small widget helpers ------------------------------------------------

void clickLater(QAbstractButton* b) { QMetaObject::invokeMethod(b, "click", Qt::QueuedConnection); }

// The visible button starting with `prefix`, else any (a tab's toolbar isn't
// laid out on the offscreen platform until it is painted).
QAbstractButton* button(QWidget* d, const QString& prefix) {
    QAbstractButton* any = nullptr;
    for (auto* b : d->findChildren<QAbstractButton*>()) {
        if (!b->text().startsWith(prefix)) continue;
        if (b->isVisible()) return b;
        if (!any) any = b;
    }
    return any;
}
bool clickOk(QDialog* d) {
    for (auto* bb : d->findChildren<QDialogButtonBox*>())
        if (auto* ok = bb->button(QDialogButtonBox::Ok)) { clickLater(ok); return true; }
    return false;
}
bool clickButton(QWidget* d, const QString& prefix) {
    if (auto* b = button(d, prefix)) { clickLater(b); return true; }
    std::fprintf(stderr, "no button '%s' in '%s'\n", qPrintable(prefix), qPrintable(d->windowTitle()));
    return false;
}
// Selects the item starting with `item` in whichever combo box has one.
bool selectItem(QWidget* d, const QString& item) {
    for (auto* cb : d->findChildren<QComboBox*>())
        for (int i = 0; i < cb->count(); i++)
            if (cb->itemText(i).startsWith(item)) { cb->setCurrentIndex(i); return true; }
    return false;
}

}  // namespace

// ---- scenarios -------------------------------------------------------------

struct GuiTestAccess {
    MainWindow& w;
    Driver&     drv;
    QString     file, list, long_file;

    bool jobsIdle() const { return w.jobs_->activeCount() == 0; }
    int  tabs() const { return static_cast<int>(w.datasets_.size()); }

    void waitJobs() { drv.step([this]() { return jobsIdle(); }); }
    // Analyses start from the trace file's tab, as a user would.
    void useFileTab() {
        drv.then([this]() {
            for (int i = 0; i < tabs(); i++)
                if (!w.datasets_[static_cast<size_t>(i)].is_result) { w.tab_bar_->setCurrentIndex(i); return; }
        });
    }
    void openTestFile() { drv.then([this]() { w.openFile(file); }); }
    void closeResultTabs() {
        drv.then([this]() {
            for (int i = tabs() - 1; i >= 0; --i)
                if (w.datasets_[static_cast<size_t>(i)].is_result) w.closeDatasetAt(i);
        });
        drv.step([]() { QApplication::processEvents(); return true; });   // deleteLater
    }
    void closeAllTabs() {
        drv.then([this]() { for (int i = tabs() - 1; i >= 0; --i) w.closeDatasetAt(i); });
    }
    // Clicks a button on the active result tab's toolbar.
    void clickToolbar(const QString& prefix) {
        drv.then([this, prefix]() {
            QWidget* bar = w.activeDs().extra_toolbar;
            if (!bar || !clickButton(bar, prefix)) {
                QStringList have;
                if (bar) for (auto* b : bar->findChildren<QAbstractButton*>()) have << b->text();
                drv.fail(QString("no toolbar button %1 on tab '%2' (has: %3)")
                             .arg(prefix, w.activeDs().display_name, have.join(", ")));
            }
        });
    }

    void ttest(const QString& order) {
        useFileTab();
        drv.onDialog("Welch t-test", [order](QDialog* d) {
            selectItem(d, order);
            return clickOk(d);
        });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
    }

    void scenarioTtest() {
        ttest("1st order");
        ttest("2nd order, univariate");
        ttest("Any order, per sample");
        closeResultTabs();
    }

    void scenarioAverage() {
        useFileTab();
        drv.onDialog("Average Trace", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.onRunAverage(); });
        waitJobs();
        closeResultTabs();
    }

    void fft(const QString& source, bool remove_and_ifft) {
        useFileTab();
        drv.onDialog("FFT Spectrum — configuration", [source](QDialog* d) {
            selectItem(d, source);
            return clickOk(d);
        });
        drv.then([this]() { w.onRunFFT(); });
        waitJobs();
        drv.step([this]() { return w.activeDs().is_result && w.activeDs().extra_toolbar; });
        if (!remove_and_ifft) return;
        drv.onDialog("Remove frequencies", [](QDialog* d) {
            auto dsp = d->findChildren<QDoubleSpinBox*>();
            dsp[0]->setValue(250.0);
            dsp[1]->setValue(4.0);
            clickButton(d, "Add");
            return true;
        });
        drv.onDialog("Remove frequencies", [](QDialog* d) { return clickButton(d, "Show inverse FFT"); });
        clickToolbar("Remove frequencies");
        waitJobs();
        drv.step([this]() { return w.activeDs().display_name.startsWith("Inverse FFT"); });
        drv.then([]() {
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->isVisible() && t->windowTitle() == "Remove frequencies") t->close();
        });
    }

    void scenarioFft() {
        fft("The average trace", true);
        fft("Each trace", true);
        closeResultTabs();
    }

    void scenarioCpa() {
        useFileTab();
        drv.onDialog("CPA — Configuration", [](QDialog* d) {
            for (auto* sp : d->findChildren<QSpinBox*>())
                if (sp->maximum() == 65536) sp->setValue(16);   // M
            return clickOk(d);
        });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) {
            return clickButton(d, "Run  Test Model");
        });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) {
            QAbstractButton* run = button(d, "Run CPA");
            if (!run || !run->isEnabled()) return false;   // test still running
            clickLater(run);
            return true;
        });
        drv.onDialog("CPA  n=", [](QDialog* d) {
            QMetaObject::invokeMethod(d, "close", Qt::QueuedConnection);
            return true;
        });
        drv.then([this]() { w.onRunCpa(); });
        // Until the result window has come and gone.
        drv.step([this]() {
            if (!drv.dialogsAnswered()) return false;
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->isVisible() && t->windowTitle().startsWith("CPA")) return false;
            return true;
        });
    }

    // CPA with less memory free than every curve would need (pretended):
    // it keeps the strongest curves and shows them as curves, not a heatmap.
    void scenarioCpaTopCurves() {
        drv.then([]() { qputenv("TRS_MEMORY_AVAILABLE_MB", "40"); });
        drv.onDialog("CPA — Configuration", [](QDialog* d) {
            for (auto* sp : d->findChildren<QSpinBox*>())
                if (sp->maximum() == 65536) sp->setValue(256);   // M (remembered from earlier runs)
            return clickOk(d);
        });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) {
            return clickButton(d, "Run  Test Model");
        });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) {
            QAbstractButton* run = button(d, "Run CPA");
            if (!run || !run->isEnabled()) return false;
            clickLater(run);
            return true;
        });
        drv.onDialog("CPA  n=", [this](QDialog* d) {
            const auto plots = d->findChildren<PlotWidget*>();
            if (plots.isEmpty() || plots[0]->traceCount() != 10)
                drv.fail(QString("expected the 10 strongest curves, got %1 plot(s)").arg(plots.size()));
            QMetaObject::invokeMethod(d, "close", Qt::QueuedConnection);
            return true;
        });
        useFileTab();
        drv.then([this]() { w.onRunCpa(); });
        drv.step([this]() {
            if (!drv.dialogsAnswered()) return false;
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->isVisible() && t->windowTitle().startsWith("CPA")) return false;
            return true;
        });
        drv.then([]() { qunsetenv("TRS_MEMORY_AVAILABLE_MB"); });
    }

    // A long trace with a step that needs whole traces (Spectral Excise):
    // zoomed out it is left out, quickly and with a note, instead of being
    // run over the whole view on every repaint; zoomed in it is applied.
    void scenarioPlotWholeTrace() {
        drv.then([this]() {
            w.openFile(long_file);
            w.addTransformToPipeline(std::make_shared<SpectralExciseTransform>(
                std::vector<SpectralBand>{SpectralBand{0.5, 0.01}}));
        });
        drv.then([this]() {
            PlotWidget* pw = w.plotWidget();
            pw->setViewRange(1, pw->totalSamples());   // a view not cached yet: a real rebuild
            QElapsedTimer t;
            t.start();
            pw->grab();   // paints, building the trace caches
            const qint64 ms = t.elapsed();
            const QString skipped = pw->traces_.empty() ? QString() : pw->traces_[0].cache.skipped;
            std::fprintf(stderr, "  zoomed out (%lld samples): %lld ms, left out: '%s'\n",
                         static_cast<long long>(pw->viewEnd() - pw->viewStart()), static_cast<long long>(ms),
                         qPrintable(skipped));
            if (skipped.isEmpty()) drv.fail("the whole-trace step was not left out at full zoom");
            if (ms > 3000) drv.fail(QString("drawing the long view took %1 ms").arg(ms));

            pw->setViewRange(0, 1'000'000);
            t.restart();
            pw->grab();
            std::fprintf(stderr, "  zoomed in (1M samples): %lld ms, left out: '%s'\n",
                         static_cast<long long>(t.elapsed()), qPrintable(pw->traces_[0].cache.skipped));
            if (!pw->traces_[0].cache.skipped.isEmpty()) drv.fail("the step was left out zoomed in");
        });
        drv.then([this]() { w.closeDatasetAt(tabs() - 1); });
        drv.step([]() { QApplication::processEvents(); return true; });
    }

    // Export as NPZ through the dialogs, then open the archive as traces.
    void scenarioExportNpz() {
        const QString out = QFileInfo(file).dir().filePath("export.npz");
        drv.then([out]() { QFile::remove(out); });   // no "replace it?" question
        drv.onDialog("Export as NPZ — configuration", [](QDialog* d) { return clickOk(d); });
        drv.onDialog("Export traces as NPZ", [out](QDialog* d) {
            auto* fd = qobject_cast<QFileDialog*>(d);
            if (!fd) return false;
            if (auto* name = fd->findChild<QLineEdit*>("fileNameEdit")) name->setText(out);
            QMetaObject::invokeMethod(fd, "accept", Qt::QueuedConnection);
            return true;
        });
        drv.onDialog("Exported", [](QDialog* d) { return clickOk(d); });
        useFileTab();
        drv.then([this]() { w.onExportNpz(); });
        drv.step([this]() { return drv.dialogsAnswered(); });
        drv.then([this, out]() {
            const int before = tabs();
            w.openNpyTraces(out);
            if (tabs() != before + 1) { drv.fail("the exported .npz did not open"); return; }
            const TrsHeader& h = w.datasets_.back().file->header();
            if (h.num_traces != 400 || h.num_samples != 4000)
                drv.fail(QString("re-opened .npz has %1 × %2").arg(h.num_traces).arg(h.num_samples));
            const float a = w.datasets_.back().file->readSample(7, 1234);
            const float b = w.datasets_[0].file->readSample(7, 1234);
            if (a != b) drv.fail(QString("sample differs after the round trip: %1 vs %2").arg(a).arg(b));
            w.closeDatasetAt(tabs() - 1);
        });
        drv.step([]() { QApplication::processEvents(); return true; });
    }

    void scenarioSnapshots() {
        useFileTab();
        drv.then([this]() {
            PlotWidget* pw = w.plotWidget();
            const int64_t n = pw->totalSamples();
            for (int i = 0; i < 70; i++) {        // past the 50-snapshot cap
                w.saveSnapshot();
                pw->setViewRange(i * n / 200, n / 2 + i * n / 200);
            }
            for (int i = 0; i < 20; i++) w.onUndoAction();
        });
    }

    void scenarioCombined() {
        drv.then([this]() { w.openFile(list); });
        drv.step([this]() { return tabs() == 2; });
        drv.then([this]() { w.closeDatasetAt(1); });
        drv.step([]() { QApplication::processEvents(); return true; });
    }

    // ---- crash recipes: each once reached freed memory or a null file ----

    // Quit with a Template Match result window open, showing its peaks on
    // the main plot: its destroyed() handler ran during ~MainWindow.
    void scenarioCrashQuitXcorr() {
        drv.onDialog("Cross-Correlation — configuration", [](QDialog* d) {
            for (auto* rb : d->findChildren<QRadioButton*>())
                if (rb->text().startsWith("Template Match")) rb->setChecked(true);
            return clickOk(d);
        });
        drv.onDialog("Template Match", [](QDialog* d) {
            auto* show = qobject_cast<QCheckBox*>(button(d, "Show on main plot"));
            if (show) show->setChecked(true);
            return true;
        });
        drv.then([this]() { w.onRunXCorr(); });
        drv.step([this]() { return drv.dialogsAnswered(); });
    }

    // A job finishing while a modal dialog is open adds a result tab and
    // makes it active; the dialog's run then read activeDs() — the result.
    void scenarioCrashJobDuringModal() {
        drv.then([this]() {
            w.jobs_->submit("slow job", [](JobContext&) { QThread::msleep(400); return true; },
                            [this](bool) {
                                w.addResultTab(std::vector<float>(100, 1.0f), "late result",
                                               QColor(Qt::red), "late");
                            });
        });
        drv.onDialog("Average Trace", [this](QDialog* d) {
            if (!jobsIdle()) return false;   // press Run only after the job's tab is there
            return clickOk(d);
        });
        drv.then([this]() { w.onRunAverage(); });
        waitJobs();
    }

    // Close the file's tab while drawing the t-test range, then Run.
    void scenarioCrashCloseTabWhileDrawing() {
        drv.onDialog("Welch t-test", [](QDialog* d) { return clickButton(d, "Draw on plot"); });
        drv.onDialog("Welch t-test", [this](QDialog* d) {
            if (!button(d, "Drawing")) return false;
            w.closeDatasetAt(0);
            return clickOk(d);
        });
        // The run then finds its file gone, and says so.
        drv.onDialog("Trace file closed", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        openTestFile();   // for the next loop
    }

    // A non-modal dialog's Run pressed while a result tab is active.
    void scenarioCrashNonModalOtherTab() {
        useFileTab();        // a result tab to switch to
        drv.onDialog("Average Trace", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.onRunAverage(); });
        waitJobs();
        useFileTab();
        drv.onDialog("FFT Spectrum — configuration", [this](QDialog* d) {
            // Switch to the result tab, then press Run.
            for (int i = 0; i < tabs(); i++)
                if (w.datasets_[static_cast<size_t>(i)].is_result) w.tab_bar_->setCurrentIndex(i);
            return clickOk(d);
        });
        drv.then([this]() { w.onRunFFT(); });
        waitJobs();
        closeResultTabs();
    }

    // Non-modal dialogs left open at quit: the destructors must cope.
    void scenarioQuitDialogs() {
        drv.then([this]() {
            w.onRunFFT();
            w.onRunCpa();
            w.onChainEditor();
            w.onPatternEditor();
            w.showAlignDialog();
        });
    }
};

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", qgetenv("QT_QPA_PLATFORM").isEmpty() ? "offscreen" : qgetenv("QT_QPA_PLATFORM"));
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);

    QString scenario = "ttest";
    int loops = 3;
    int timeout_s = 300;
    const QStringList args = app.arguments();
    for (int i = 1; i + 1 < args.size(); i++) {
        if (args[i] == "--scenario") scenario = args[i + 1];
        if (args[i] == "--loops") loops = std::max(1, args[i + 1].toInt());
        if (args[i] == "--timeout") timeout_s = std::max(1, args[i + 1].toInt());
    }

    QTemporaryDir dir;
    if (qEnvironmentVariableIsSet("SOAK_KEEP_DIR")) {   // to look at the files afterwards
        dir.setAutoRemove(false);
        std::fprintf(stderr, "test files kept in %s\n", qPrintable(dir.path()));
    }
    const QString file = dir.filePath("soak.trs");
    writeTrs(file, 400, 4000, 1);
    writeTrs(dir.filePath("soak_part000.trs"), 100, 4000, 2);
    writeTrs(dir.filePath("soak_part001.trs"), 100, 4000, 3);
    const QString list = dir.filePath("soak.trslist");
    {
        QFile l(list);
        if (!l.open(QIODevice::WriteOnly | QIODevice::Text)) return 2;
        l.write("soak_part000.trs\nsoak_part001.trs\n");
    }

    Driver drv;
    int rc = 0;
    {
        MainWindow win;
        win.show();
        QString long_file;
        if (scenario == "plot_whole_trace") {
            long_file = dir.filePath("long.trs");
            writeTrs(long_file, 2, 10'000'000, 4);
        }
        GuiTestAccess t{win, drv, file, list, long_file};
        double rss_first = 0, rss_last = 0;

        t.openTestFile();
        for (int loop = 0; loop < loops; loop++) {
            if (scenario == "ttest")             t.scenarioTtest();
            else if (scenario == "average")      t.scenarioAverage();
            else if (scenario == "fft")          t.scenarioFft();
            else if (scenario == "cpa")          t.scenarioCpa();
            else if (scenario == "cpa_top_curves") t.scenarioCpaTopCurves();
            else if (scenario == "plot_whole_trace") t.scenarioPlotWholeTrace();
            else if (scenario == "export_npz")   t.scenarioExportNpz();
            else if (scenario == "snapshots")    t.scenarioSnapshots();
            else if (scenario == "combined")     t.scenarioCombined();
            else if (scenario == "quit_dialogs") { t.scenarioQuitDialogs(); loops = 1; break; }
            else if (scenario == "crash_quit_xcorr")   { t.scenarioCrashQuitXcorr(); loops = 1; break; }
            else if (scenario == "crash_job_during_modal")      t.scenarioCrashJobDuringModal();
            else if (scenario == "crash_close_tab_drawing")     t.scenarioCrashCloseTabWhileDrawing();
            else if (scenario == "crash_nonmodal_other_tab")    t.scenarioCrashNonModalOtherTab();
            else { std::fprintf(stderr, "unknown scenario %s\n", qPrintable(scenario)); return 2; }
            if (loop == 0) drv.then([&]() { rss_first = rssMB(); });
        }
        drv.then([&]() { rss_last = rssMB(); });
        rc = drv.run(timeout_s * 1000);

        std::printf("scenario %s: %d loops, RSS after loop 1 %.1f MB, after loop %d %.1f MB\n",
                    qPrintable(scenario), loops, rss_first, loops, rss_last);
        // Loop 1 warms caches; later loops should hold steady.
        const double allowed = 30.0 + 0.10 * rss_first;
#if defined(__SANITIZE_ADDRESS__)
        const bool check_rss = false;   // freed memory sits in ASan's quarantine
#else
        const bool check_rss = loops > 1;
#endif
        if (rc == 0 && check_rss && rss_last - rss_first > allowed) {
            std::fprintf(stderr, "FAIL: memory grew by %.1f MB over %d loops (allowed %.1f)\n",
                         rss_last - rss_first, loops - 1, allowed);
            rc = 1;
        }
    }   // MainWindow destroyed here, with whatever dialogs are still open
    LeakageModel::globalShutdown();
    std::printf("%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
