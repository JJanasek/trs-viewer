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
#include "crash_handler.h"
#include "npz.h"

#include <csignal>

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

// A capture the way a scope script leaves it: int16 traces.npy with labels.npz
// next to it. Bit 1 of byte 3 of "lam" leaks at kLabelLeakAt; the top bit of its
// last byte never changes.
constexpr int kLabelLeakAt = 700;

std::vector<uint8_t> npyBytes(const std::string& descr, const std::string& shape, const void* data, size_t bytes) {
    std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    dict.resize(((dict.size() + 11 + 63) / 64) * 64 - 11, ' ');
    dict += '\n';
    std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                static_cast<uint8_t>(dict.size()), static_cast<uint8_t>(dict.size() >> 8)};
    out.insert(out.end(), dict.begin(), dict.end());
    const auto* d = static_cast<const uint8_t*>(data);
    out.insert(out.end(), d, d + bytes);
    return out;
}

void writeNpyFile(const QString& path, const std::vector<uint8_t>& bytes) {
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) { std::fprintf(stderr, "cannot write %s\n", qPrintable(path)); std::exit(2); }
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<qint64>(bytes.size()));
}

void writeNpyCapture(const QString& dir, int n, int ns, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, 100.0f);
    std::vector<uint8_t> lam(static_cast<size_t>(n) * 32), x0(static_cast<size_t>(n) * 32), ok(static_cast<size_t>(n), 1);
    std::vector<int32_t> retval(static_cast<size_t>(n), 0);
    std::vector<double>  t_unix(static_cast<size_t>(n));
    std::vector<float>   capture_ms(static_cast<size_t>(n));
    std::vector<int16_t> traces(static_cast<size_t>(n) * static_cast<size_t>(ns));
    for (int t = 0; t < n; t++) {
        uint8_t* row = lam.data() + static_cast<size_t>(t) * 32;
        for (int b = 0; b < 32; b++) {
            row[b] = static_cast<uint8_t>(rng());
            x0[static_cast<size_t>(t) * 32 + static_cast<size_t>(b)] = static_cast<uint8_t>(rng());
        }
        row[31] &= 0x7F;
        t_unix[static_cast<size_t>(t)] = 1.8e9 + t;
        capture_ms[static_cast<size_t>(t)] = 700.0f + static_cast<float>(t % 10);
        const bool leak = (row[3] >> 1) & 1;
        for (int s = 0; s < ns; s++) {
            const float v = noise(rng) + (leak && s == kLabelLeakAt ? 300.0f : 0.0f);
            traces[static_cast<size_t>(t) * static_cast<size_t>(ns) + static_cast<size_t>(s)] = static_cast<int16_t>(v);
        }
    }
    const std::string rows = std::to_string(n);
    writeNpyFile(dir + "/traces.npy", npyBytes("<i2", "(" + rows + ", " + std::to_string(ns) + ")",
                                               traces.data(), traces.size() * sizeof(int16_t)));
    QString err;
    if (!saveNpz(dir + "/labels.npz", {
            {"lam.npy",        npyBytes("|u1", "(" + rows + ", 32)", lam.data(), lam.size())},
            {"x0.npy",         npyBytes("|u1", "(" + rows + ", 32)", x0.data(), x0.size())},
            {"ok.npy",         npyBytes("|u1", "(" + rows + ",)", ok.data(), ok.size())},
            {"retval.npy",     npyBytes("<i4", "(" + rows + ",)", retval.data(), retval.size() * 4)},
            {"t_unix.npy",     npyBytes("<f8", "(" + rows + ",)", t_unix.data(), t_unix.size() * 8)},
            {"capture_ms.npy", npyBytes("<f4", "(" + rows + ",)", capture_ms.data(), capture_ms.size() * 4)},
        }, err)) {
        std::fprintf(stderr, "%s\n", qPrintable(err));
        std::exit(2);
    }
}

// Picks a label, byte and bit in an analysis dialog's GroupPicker.
bool pickGroup(QDialog* d, const QString& field, int byte, const QString& bit) {
    auto* f = d->findChild<QComboBox*>("group_field");
    auto* b = d->findChild<QSpinBox*>("group_byte");
    auto* m = d->findChild<QComboBox*>("group_bit");
    if (!f || !b) return false;
    const int i = f->findText(field, Qt::MatchStartsWith);
    if (i < 0) return false;
    f->setCurrentIndex(i);
    b->setValue(byte);
    if (m && !bit.isEmpty()) {
        const int j = m->findText(bit, Qt::MatchStartsWith);
        if (j < 0) return false;
        m->setCurrentIndex(j);
    }
    return true;
}

// ---- scenarios -------------------------------------------------------------

struct GuiTestAccess {
    MainWindow& w;
    Driver&     drv;
    QString     file, list, long_file, capture_dir;

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

    // The active result tab's curve and where its largest magnitude is.
    int peakOfActiveCurve(float* magnitude = nullptr) {
        PlotWidget* pw = w.plotWidget();
        if (pw->traces_.empty() || !pw->traces_[0].mem_data || pw->traces_[0].mem_data->empty()) return -1;
        const std::vector<float>& c = *pw->traces_[0].mem_data;
        size_t peak = 0;
        for (size_t i = 0; i < c.size(); i++)
            if (std::fabs(c[i]) > std::fabs(c[peak])) peak = i;
        if (magnitude) *magnitude = std::fabs(c[peak]);
        return static_cast<int>(peak);
    }
    void useCaptureTab() {
        drv.then([this]() {
            for (int i = 0; i < tabs(); i++)
                if (w.datasets_[static_cast<size_t>(i)].display_name.endsWith("traces.npy")) {
                    w.tab_bar_->setCurrentIndex(i);
                    return;
                }
            drv.fail("the capture's tab is gone");
        });
    }

    // traces.npy with labels.npz next to it: the labels arrive as per-trace
    // data, and every analysis can split on or read them.
    void scenarioNpyLabels() {
        const QString npy = capture_dir + "/traces.npy";
        drv.then([this, npy]() {
            w.openAnyFile(npy);
            const TrsFile& f = *w.activeDs().file;
            const TrsHeader& h = f.header();
            if (h.num_traces != 400 || h.num_samples != 2000 || h.sample_type != SampleType::INT16)
                return drv.fail("the matrix did not open as it is on disk");
            for (const char* name : {"lam", "x0", "ok", "retval", "t_unix", "capture_ms"})
                if (!h.param_map.count(name)) return drv.fail(QString("no label %1").arg(name));
            if (h.data_length != 32 + 32 + 1 + 4 + 8 + 4) return drv.fail(QString("%1 data bytes").arg(h.data_length));
            if (!w.statusBar()->currentMessage().contains("lam")) return drv.fail("the status bar doesn't say");
            w.updateTraceDataDisplay();
            if (!w.lbl_trace_data_->text().contains("lam")) return drv.fail("the trace data panel doesn't show lam");
        });

        // t-test on the bit that leaks
        drv.onDialog("Welch t-test", [this](QDialog* d) {
            if (!pickGroup(d, "lam", 3, "bit 1")) return false;
            if (auto* c = d->findChild<QLabel*>("group_counts"); !c || c->text().contains("empty"))
                drv.fail("the group sizes aren't shown properly");
            selectItem(d, "1st order");
            return clickOk(d);
        });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        drv.then([this]() {
            if (!w.activeDs().display_name.contains("lam[3] bit 1"))
                return drv.fail("tab title: " + w.activeDs().display_name);
            float t = 0;
            const int at = peakOfActiveCurve(&t);
            std::fprintf(stderr, "  t-test lam[3] bit 1: max |t| %.1f at %d\n", t, at);
            if (at != kLabelLeakAt || t < 10.0f) drv.fail(QString("max |t| %1 at %2").arg(t).arg(at));
        });

        // a bit that doesn't: nothing stands out
        useCaptureTab();
        drv.onDialog("Welch t-test", [](QDialog* d) { return pickGroup(d, "lam", 3, "bit 2") && clickOk(d); });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        drv.then([this]() {
            float t = 0;
            peakOfActiveCurve(&t);
            if (t > 6.0f) drv.fail(QString("lam[3] bit 2 shows |t| %1").arg(t));
        });

        // a bit that never changes: the dialog says so, and the run doesn't crash
        useCaptureTab();
        drv.onDialog("Welch t-test", [this](QDialog* d) {
            if (!pickGroup(d, "lam", 31, "bit 7")) return false;
            auto* c = d->findChild<QLabel*>("group_counts");
            if (!c || !c->text().contains("empty")) drv.fail("no word about the empty group");
            return clickOk(d);
        });
        drv.onDialog("T-test", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        drv.step([this]() { return drv.dialogsAnswered(); });

        // Every bit of the label: one window with all of them, saved to a file
        const QString scan_out = capture_dir + "/scan.npz", scan_csv = capture_dir + "/scan.csv";
        auto acceptFile = [](QDialog* d, const QString& path) {
            auto* fd = qobject_cast<QFileDialog*>(d);
            if (!fd) return false;
            if (auto* name = fd->findChild<QLineEdit*>("fileNameEdit")) name->setText(path);
            QMetaObject::invokeMethod(fd, "accept", Qt::QueuedConnection);
            return true;
        };
        useCaptureTab();
        drv.then([scan_out, scan_csv]() { QFile::remove(scan_out); QFile::remove(scan_csv); });
        drv.onDialog("Welch t-test", [](QDialog* d) { return pickGroup(d, "lam", 0, "every bit") && clickOk(d); });
        drv.onDialog("T-test on every bit", [this](QDialog* d) {
            if (!d->windowTitle().contains("lam[3] bit 1")) drv.fail("wrong strongest bit: " + d->windowTitle());
            auto* tbl = d->findChild<QTableWidget*>("bit_scan_table");
            if (!tbl || tbl->rowCount() != 256) { drv.fail("the table doesn't have 256 bits"); return false; }
            return clickButton(d, "Save all");
        });
        drv.onDialog("Save the t-test on every bit", [=](QDialog* d) { return acceptFile(d, scan_out); });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        drv.step([this, scan_out]() { return drv.dialogsAnswered() && QFile::exists(scan_out); });
        drv.then([this, scan_out]() {
            std::map<std::string, std::vector<uint8_t>> e;
            QString err;
            NpyArray t, peak;
            if (!loadNpz(scan_out, e, err) || !parseNpyArray(e["t.npy"], t, err) || !parseNpyArray(e["peak_t.npy"], peak, err))
                return drv.fail("the saved results don't read back: " + err);
            if (t.shape != std::vector<int64_t>{256, 2000}) return drv.fail("saved t has the wrong shape");
            float at_leak;
            std::memcpy(&at_leak, t.payload.data() + (8u * 3u + 1u) * 2000u * 4u + kLabelLeakAt * 4u, 4);
            if (std::fabs(at_leak) < 10.0f) drv.fail(QString("saved t at the leak: %1").arg(at_leak));
        });
        drv.onDialog("T-test on every bit", [](QDialog* d) { return clickButton(d, "Save table"); });
        drv.onDialog("Save the table", [=](QDialog* d) { return acceptFile(d, scan_csv); });
        drv.step([this, scan_csv]() { return drv.dialogsAnswered() && QFile::exists(scan_csv); });
        drv.then([this, scan_csv]() {
            QFile f(scan_csv);
            if (!f.open(QIODevice::ReadOnly) || f.readAll().count('\n') != 257) drv.fail("the table has the wrong number of lines");
        });
        drv.onDialog("T-test on every bit", [](QDialog* d) { return clickButton(d, "Show curve"); });
        drv.step([this]() {
            if (!drv.dialogsAnswered()) return false;
            // The strongest row is selected: its curve becomes the active tab.
            if (!w.activeDs().display_name.contains("lam[3] bit 1")) return false;
            const int at = peakOfActiveCurve();
            if (at != kLabelLeakAt) drv.fail(QString("the bit's curve peaks at %1").arg(at));
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->windowTitle().startsWith("T-test on every bit")) t->close();
            return true;
        });
        drv.step([]() { QApplication::processEvents(); return true; });

        // Average per group
        useCaptureTab();
        drv.onDialog("Average Trace", [](QDialog* d) {
            selectItem(d, "Each t-test group separately");
            return pickGroup(d, "lam", 3, "bit 1") && clickOk(d);
        });
        drv.then([this]() { w.onRunAverage(); });
        waitJobs();
        drv.then([this]() {
            if (!w.activeDs().display_name.startsWith("Difference of means"))
                return drv.fail("no difference of means, but: " + w.activeDs().display_name);
            const int at = peakOfActiveCurve();
            if (at != kLabelLeakAt) drv.fail(QString("the difference peaks at %1").arg(at));
        });

        // SNR over a label's byte
        useCaptureTab();
        drv.onDialog("SNR — configuration", [](QDialog* d) {
            selectItem(d, "Hamming weight");
            return pickGroup(d, "lam", 3, {}) && clickOk(d);
        });
        drv.then([this]() { w.onRunSNR(); });
        waitJobs();
        drv.then([this]() { if (peakOfActiveCurve() < 0) drv.fail("no SNR curve"); });

        // CPA: the model finds the label through the offsets the editor lists
        useCaptureTab();
        drv.onDialog("CPA — Configuration", [](QDialog* d) {
            for (auto* sp : d->findChildren<QSpinBox*>())
                if (sp->maximum() == 65536) sp->setValue(8);   // M: one per bit of the byte
            return clickOk(d);
        });
        drv.onDialog("CPA — Leakage Model Editor", [this](QDialog* d) {
            const int off = w.activeDs().file->header().param_map.at("lam").offset;
            bool listed = false;
            for (auto* l : d->findChildren<QLabel*>())
                listed = listed || l->text().contains(QString("lam = plaintexts[:, %1:%2]").arg(off).arg(off + 32));
            if (!listed) drv.fail("the model editor doesn't say where lam is");
            auto* ed = d->findChild<QPlainTextEdit*>();
            if (!ed) return false;
            ed->setPlainText(QString("import numpy as np\n"
                                     "def get_leakages(plaintexts, ciphertexts, key_guess):\n"
                                     "    return ((plaintexts[:, %1] >> key_guess) & 1).astype(np.float32)\n").arg(off + 3));
            return clickButton(d, "Run  Test Model");
        });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) {
            QAbstractButton* run = button(d, "Run CPA");
            if (!run || !run->isEnabled()) return false;   // test still running
            clickLater(run);
            return true;
        });
        drv.onDialog("CPA  n=", [this](QDialog* d) {
            if (!d->windowTitle().contains("best=h1"))
                drv.fail("CPA picked the wrong bit: " + d->windowTitle());
            QMetaObject::invokeMethod(d, "close", Qt::QueuedConnection);
            return true;
        });
        drv.then([this]() { w.onRunCpa(); });
        drv.step([this]() {
            if (!drv.dialogsAnswered()) return false;
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->isVisible() && t->windowTitle().startsWith("CPA")) return false;
            return true;
        });

        // Export TRS keeps the labels and their names
        const QString out = capture_dir + "/exported.trs";
        useCaptureTab();
        drv.then([out]() { QFile::remove(out); });
        drv.onDialog("Export TRS — select range", [](QDialog* d) { return clickOk(d); });
        drv.onDialog("Export processed TRS", [out](QDialog* d) {
            auto* fd = qobject_cast<QFileDialog*>(d);
            if (!fd) return false;
            if (auto* name = fd->findChild<QLineEdit*>("fileNameEdit")) name->setText(out);
            QMetaObject::invokeMethod(fd, "accept", Qt::QueuedConnection);
            return true;
        });
        drv.onDialog("Export complete", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.onExportTrs(); });
        drv.step([this]() { return drv.dialogsAnswered(); });
        drv.then([this, out]() {
            auto src = w.activeDs().file;
            w.openFile(out);
            auto copy = w.activeDs().file;
            if (copy == src) return drv.fail("the exported TRS did not open");
            if (!(copy->header().param_map == src->header().param_map)) return drv.fail("the labels' names were lost");
            if (copy->readData(5) != src->readData(5)) return drv.fail("the labels changed in the export");
            if (copy->readSample(5, kLabelLeakAt) != src->readSample(5, kLabelLeakAt))
                return drv.fail("the samples changed in the export");
            w.closeDatasetAt(w.active_idx_);
        });
        drv.step([]() { QApplication::processEvents(); return true; });

        // A .npy without labels still opens, and the t-test says what to do
        const QString bare = capture_dir + "/bare/traces.npy";
        drv.onDialog("T-test", [this](QDialog* d) {
            auto* m = qobject_cast<QMessageBox*>(d);
            if (!m || !m->text().contains("labels.npz")) drv.fail("no hint about labels.npz");
            return clickOk(d);
        });
        drv.then([this, bare]() {
            w.openAnyFile(bare);
            if (w.activeDs().file->header().data_length != 0) return drv.fail("labels out of nowhere");
            w.showTTestDialog(nullptr);
        });
        drv.step([this]() { return drv.dialogsAnswered(); });
        drv.then([this]() { w.closeDatasetAt(w.active_idx_); });

        // Labels of another length are not used, and that is said
        const QString other = capture_dir + "/other/traces.npy";
        drv.onDialog("Labels not loaded", [](QDialog* d) { return clickOk(d); });
        drv.then([this, other]() {
            w.openAnyFile(other);
            if (w.activeDs().file->header().data_length != 0) drv.fail("labels of the wrong length were used");
            w.closeDatasetAt(w.active_idx_);
        });
        drv.step([this]() { return drv.dialogsAnswered(); });

        closeResultTabs();
        drv.then([this]() {
            for (int i = tabs() - 1; i >= 0; --i)
                if (w.datasets_[static_cast<size_t>(i)].display_name.endsWith("traces.npy")) w.closeDatasetAt(i);
        });
        drv.step([]() { QApplication::processEvents(); return true; });
    }

    // Two captures, each a folder with traces.npy and labels.npz, opened as one
    // set from the folder holding them; then everything in the Analyze menu.
    void scenarioCombinedNpy() {
        const QString list = capture_dir + "/run.trslist";
        drv.then([list]() { QFile::remove(list); });
        drv.onDialog("Open as One Dataset", [](QDialog* d) { return clickButton(d, "Add Folders"); });
        drv.onDialog("Add capture folders", [this](QDialog* d) {
            auto* fd = qobject_cast<QFileDialog*>(d);
            if (!fd) return false;
            if (auto* name = fd->findChild<QLineEdit*>("fileNameEdit")) name->setText(capture_dir);
            QMetaObject::invokeMethod(fd, "accept", Qt::QueuedConnection);
            return true;
        });
        drv.onDialog("Open as One Dataset", [this, list](QDialog* d) {
            auto* tbl = d->findChild<QTableWidget*>();
            if (!tbl || tbl->rowCount() != 2) return false;   // the folder isn't in yet
            if (!tbl->item(0, 0)->text().endsWith("run_a/traces.npy")) drv.fail("row: " + tbl->item(0, 0)->text());
            if (auto* ed = d->findChild<QLineEdit*>(); !ed || ed->text() != list)
                drv.fail("list: " + (ed ? ed->text() : QString()));
            return clickOk(d);
        });
        drv.then([this]() { w.onOpenCombined(); });
        drv.step([this]() { return drv.dialogsAnswered() && w.hasActiveDs() && w.activeDs().display_name == "run.trslist"; });

        auto checkSet = [this]() {
            const TrsFile& f = *w.activeDs().file;
            if (f.parts().size() != 2 || f.header().num_traces != 800) return drv.fail("not one set of 800 traces");
            if (!f.header().param_map.count("lam") || !f.dataInMemory()) return drv.fail("the labels didn't come along");
            if (f.readData(0).empty() || f.readData(799).empty()) drv.fail("no labels for the set's traces");
            if (f.readData(0) == f.readData(400)) drv.fail("the second capture has the first one's labels");
        };
        drv.then([this, checkSet]() {
            checkSet();
            if (!w.statusBar()->currentMessage().contains("lam")) drv.fail("the status bar doesn't say");
            w.spin_data_idx_->setValue(500);
            w.updateTraceDataDisplay();
            if (!w.lbl_trace_data_->text().contains("run_b/traces.npy")) drv.fail("the panel doesn't name the part");
        });
        const auto useSetTab = [this]() {
            drv.then([this]() {
                for (int i = 0; i < tabs(); i++)
                    if (w.datasets_[static_cast<size_t>(i)].display_name == "run.trslist") {
                        w.tab_bar_->setCurrentIndex(i);
                        return;
                    }
                drv.fail("the set's tab is gone");
            });
        };

        // t-test on one bit
        drv.onDialog("Welch t-test", [](QDialog* d) {
            auto* c = d->findChild<QLabel*>("group_counts");
            return c && pickGroup(d, "lam", 3, "bit 1") && clickOk(d);
        });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        drv.then([this]() {
            float t = 0;
            const int at = peakOfActiveCurve(&t);
            std::fprintf(stderr, "  combined t-test: %s, max |t| %.1f at %d\n", qPrintable(w.activeDs().display_name), t, at);
            if (!w.activeDs().display_name.contains("800 traces")) drv.fail("not all traces: " + w.activeDs().display_name);
            if (at != kLabelLeakAt || t < 15.0f) drv.fail(QString("max |t| %1 at %2").arg(t).arg(at));
        });
        // ... and on every bit
        useSetTab();
        drv.onDialog("Welch t-test", [](QDialog* d) { return pickGroup(d, "lam", 0, "every bit") && clickOk(d); });
        drv.onDialog("T-test on every bit", [this](QDialog* d) {
            if (!d->windowTitle().contains("800 traces") || !d->windowTitle().contains("lam[3] bit 1"))
                drv.fail("every bit: " + d->windowTitle());
            QMetaObject::invokeMethod(d, "close", Qt::QueuedConnection);
            return true;
        });
        drv.then([this]() { w.showTTestDialog(nullptr); });
        waitJobs();
        drv.step([this]() { return drv.dialogsAnswered(); });
        // Leakage vs Frequency
        useSetTab();
        drv.onDialog("Leakage vs Frequency", [](QDialog* d) { return pickGroup(d, "lam", 3, "bit 1") && clickOk(d); });
        drv.then([this]() { w.onRunSpectralLeakage(); });
        waitJobs();
        drv.step([this]() { return drv.dialogsAnswered(); });
        // Average per group
        useSetTab();
        drv.onDialog("Average Trace", [](QDialog* d) {
            selectItem(d, "Each t-test group separately");
            return pickGroup(d, "lam", 3, "bit 1") && clickOk(d);
        });
        drv.then([this]() { w.onRunAverage(); });
        waitJobs();
        drv.then([this]() {
            if (peakOfActiveCurve() != kLabelLeakAt) drv.fail("the difference of means misses the leak");
        });
        // SNR
        useSetTab();
        drv.onDialog("SNR — configuration", [](QDialog* d) {
            selectItem(d, "Hamming weight");
            return pickGroup(d, "lam", 3, {}) && clickOk(d);
        });
        drv.then([this]() { w.onRunSNR(); });
        waitJobs();
        drv.step([this]() { return drv.dialogsAnswered(); });
        // Static SNR
        useSetTab();
        drv.onDialog("Static SNR", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.onRunStaticSNR(); });
        waitJobs();
        drv.step([this]() { return drv.dialogsAnswered(); });
        // CPA, with the model made from the label
        useSetTab();
        drv.onDialog("CPA — Configuration", [this](QDialog* d) {
            auto* m = d->findChild<QComboBox*>("cpa_model");
            if (!m || m->findText("lam: each bit") < 0) { drv.fail("no model from the label"); return false; }
            m->setCurrentIndex(m->findText("lam: each bit"));
            for (auto* sp : d->findChildren<QSpinBox*>())
                if (sp->maximum() == 65536 && sp->value() != 256) drv.fail("the model didn't set M");
            return clickOk(d);
        });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) { return clickButton(d, "Run  Test Model"); });
        drv.onDialog("CPA — Leakage Model Editor", [](QDialog* d) {
            QAbstractButton* run = button(d, "Run CPA");
            if (!run || !run->isEnabled()) return false;
            clickLater(run);
            return true;
        });
        drv.onDialog("CPA  n=", [this](QDialog* d) {
            if (!d->windowTitle().contains("n=800") || !d->windowTitle().contains("best=h25"))
                drv.fail("CPA: " + d->windowTitle());
            QMetaObject::invokeMethod(d, "close", Qt::QueuedConnection);
            return true;
        });
        drv.then([this]() { w.onRunCpa(); });
        drv.step([this]() {
            if (!drv.dialogsAnswered()) return false;
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->isVisible() && t->windowTitle().startsWith("CPA")) return false;
            return true;
        });
        // Cross-correlation
        useSetTab();
        drv.onDialog("Cross-Correlation — configuration", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.onRunXCorr(); });
        drv.step([this]() { return drv.dialogsAnswered() && jobsIdle(); });
        drv.then([]() {
            for (QWidget* t : QApplication::topLevelWidgets())
                if (t->isVisible() && (t->windowTitle().startsWith("Cross") || t->windowTitle().startsWith("Template")))
                    t->close();
        });
        // FFT
        useSetTab();
        drv.onDialog("FFT Spectrum — configuration", [](QDialog* d) { return clickOk(d); });
        drv.then([this]() { w.onRunFFT(); });
        drv.step([this]() { return drv.dialogsAnswered() && jobsIdle(); });

        closeResultTabs();
        useSetTab();
        // Reopened from its list: the same.
        drv.then([this, list]() {
            for (int i = tabs() - 1; i >= 0; --i)
                if (w.datasets_[static_cast<size_t>(i)].display_name == "run.trslist") w.closeDatasetAt(i);
            w.openAnyFile(list);
        });
        drv.then(checkSet);
        drv.then([this]() {
            for (int i = tabs() - 1; i >= 0; --i)
                if (w.datasets_[static_cast<size_t>(i)].display_name == "run.trslist") w.closeDatasetAt(i);
        });
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

    // SIGTERM while a job runs and a dialog is open: the quit goes through the
    // event loop, the job is cancelled, and the teardown frees everything
    // (the memory tools check that at exit).
    void scenarioSigtermDuringJob() {
        drv.then([this]() {
            w.jobs_->submit("long job", [](JobContext& ctx) {
                for (int i = 0; i < 3000 && !ctx.cancelled(); i++) QThread::msleep(10);   // up to 30 s
                return !ctx.cancelled();
            }, [](bool) {});
            w.onRunFFT();   // a non-modal dialog left open
        });
        drv.then([]() { ::kill(::getpid(), SIGTERM); });
        drv.step([]() { return false; });   // until the signal ends the run
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
    installQuitOnTermSignals();   // as the app does: SIGTERM quits through the event loop

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
    // Not the user's settings: dialogs remember what they were last set to,
    // and every file opened here would land in their recent files.
    QSettings::setPath(QSettings::NativeFormat, QSettings::UserScope, dir.path());
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, dir.path());
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
        QString capture_dir;
        if (scenario == "npy_labels") {
            capture_dir = dir.filePath("capture");
            for (const char* sub : {"", "/bare", "/other"}) QDir().mkpath(capture_dir + sub);
            writeNpyCapture(capture_dir, 400, 2000, 5);
            // The same matrix alone, and with labels that don't match it.
            QFile::copy(capture_dir + "/traces.npy", capture_dir + "/bare/traces.npy");
            QFile::copy(capture_dir + "/traces.npy", capture_dir + "/other/traces.npy");
            const uint8_t three[3] = {1, 2, 3};
            QString err;
            saveNpz(capture_dir + "/other/labels.npz", {{"lam.npy", npyBytes("|u1", "(3,)", three, 3)}}, err);
        }
        if (scenario == "combined_npy") {
            capture_dir = dir.filePath("runs");
            for (const char* sub : {"run_a", "run_b"}) QDir().mkpath(capture_dir + "/" + sub);
            writeNpyCapture(capture_dir + "/run_a", 400, 2000, 6);
            writeNpyCapture(capture_dir + "/run_b", 400, 2000, 7);
        }
        GuiTestAccess t{win, drv, file, list, long_file, capture_dir};
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
            else if (scenario == "npy_labels")   t.scenarioNpyLabels();
            else if (scenario == "combined_npy") t.scenarioCombinedNpy();
            else if (scenario == "snapshots")    t.scenarioSnapshots();
            else if (scenario == "combined")     t.scenarioCombined();
            else if (scenario == "quit_dialogs") { t.scenarioQuitDialogs(); loops = 1; break; }
            else if (scenario == "sigterm_during_job") { t.scenarioSigtermDuringJob(); loops = 1; break; }
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
