#include "mainwindow.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QIcon>
#include <QShortcut>

int main(int argc, char* argv[]) {
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
