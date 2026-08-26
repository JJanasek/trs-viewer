#include "mainwindow.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QIcon>
#include <QShortcut>

int main(int argc, char* argv[]) {
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
