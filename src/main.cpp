#include "mainwindow.h"

#include <QApplication>
#include <QByteArray>
#include <QCommandLineParser>
#include <QIcon>
#include <QShortcut>

#if defined(__linux__)
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

// Works around a startup SIGSEGV inside libfontconfig (FcCharSetHasChar /
// FcCharSetFindLeafForward), reached when Qt measures a widget's text during
// the first layout pass — e.g. QPushButton::sizeHint() while MainWindow is
// being built. The fault is a corrupt/version-mismatched fontconfig *cache*
// (common on Fedora after a fontconfig/freetype update leaves stale files in
// ~/.cache/fontconfig or /var/cache/fontconfig); it is not our code and it is
// not recoverable once it fires, since it is a segfault deep in a third-party
// library, not a C++ exception.
//
// The only way to prevent it from inside the app is to stop fontconfig from
// ever reading those caches. We point it at a private, app-owned config whose
// only <cachedir> is under our cache directory, so fontconfig rebuilds a fresh
// cache on first launch (a one-time cost) and never touches the machine's bad
// one. We deliberately do NOT <include> /etc/fonts/fonts.conf, because that is
// where the system/xdg cachedirs are declared — pulling it in would re-add the
// corrupt caches to the read list and the crash could still happen. We list
// the standard font directories ourselves and include /etc/fonts/conf.d for
// the alias rules ("Monospace", "Sans", ...) the UI relies on; conf.d holds
// matching rules only, no cachedirs. Must run before QApplication constructs
// the font database.
static void useIsolatedFontconfigCache() {
    namespace fs = std::filesystem;

    // Respect an existing explicit override rather than fighting it.
    if (const char* existing = std::getenv("FONTCONFIG_FILE"); existing && *existing)
        return;

    const char* xdg = std::getenv("XDG_CACHE_HOME");
    const char* home = std::getenv("HOME");
    fs::path base;
    if (xdg && *xdg)        base = xdg;
    else if (home && *home) base = fs::path(home) / ".cache";
    else                    return;  // nowhere safe to put a private cache

    const fs::path dir   = base / "trs-viewer" / "fontconfig";
    const fs::path cache = dir / "cache";
    const fs::path conf  = dir / "fonts.conf";

    std::error_code ec;
    fs::create_directories(cache, ec);
    if (ec) return;

    // Minimal XML escaping for the one interpolated path (cache dir).
    auto xmlEscape = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '&':  out += "&amp;";  break;
                case '<':  out += "&lt;";   break;
                case '>':  out += "&gt;";   break;
                default:   out += c;         break;
            }
        }
        return out;
    };

    std::ostringstream cfg;
    cfg << "<?xml version=\"1.0\"?>\n"
           "<!DOCTYPE fontconfig SYSTEM \"fonts.dtd\">\n"
           "<fontconfig>\n"
           "  <dir>/usr/share/fonts</dir>\n"
           "  <dir>/usr/local/share/fonts</dir>\n"
           "  <dir>/usr/share/X11/fonts</dir>\n"
           "  <dir prefix=\"xdg\">fonts</dir>\n"
           "  <dir>~/.fonts</dir>\n"
           "  <cachedir>" << xmlEscape(cache.string()) << "</cachedir>\n"
           "  <include ignore_missing=\"yes\">/etc/fonts/conf.d</include>\n"
           "</fontconfig>\n";
    const std::string desired = cfg.str();

    // Only rewrite when the content changed, so fontconfig doesn't treat the
    // config as freshly modified on every launch.
    std::string current;
    if (std::ifstream in{conf}; in) {
        std::ostringstream buf;
        buf << in.rdbuf();
        current = buf.str();
    }
    if (current != desired) {
        std::ofstream out{conf, std::ios::trunc};
        if (!out) return;
        out << desired;
        if (!out) return;
    }

    qputenv("FONTCONFIG_FILE", QByteArray::fromStdString(conf.string()));
}
#endif  // __linux__

int main(int argc, char* argv[]) {
#if defined(__linux__)
    useIsolatedFontconfigCache();
#endif

    // Works around a Qt6/KDE Plasma Wayland crash: the native dialog helper
    // used by QMessageBox/QFileDialog under KDE's platform theme integration
    // (QDialogPrivate::setNativeDialogVisible()) frees itself via
    // helper->hide() without resetting nativeDialogInUse, so the dialog's
    // own destructor calls setVisible(false) a second time and touches
    // already-freed platform resources — SIGSEGV inside
    // QMessageBoxPrivate::setVisible(), reproducible on any QMessageBox/
    // QFileDialog, not anything specific to this app. Falling back to Qt's
    // own (non-native) dialogs sidesteps the bug entirely; the rest of the
    // platform theme (icons, palette, fonts) is untouched.
    // https://forum.qt.io/topic/164868/qmessagebox-crash-on-close
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QApplication app(argc, argv);
    app.setApplicationName("TRS Viewer");
    app.setApplicationVersion("1.0");
    app.setWindowIcon(QIcon(":/docs/logo.svg"));

    QCommandLineParser parser;
    parser.setApplicationDescription("Memory-efficient viewer for Riscure TRS power trace files.");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("file", "TRS file to open on startup", "[file]");
    parser.process(app);

    MainWindow win;
    win.show();

    const QStringList& args = parser.positionalArguments();
    if (!args.isEmpty())
        win.openFile(args.first());

    return app.exec();
}
