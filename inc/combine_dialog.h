#pragma once
// "Open as One Dataset" dialog: several TRS files read one after another as
// one trace set (TrsFile::openCombined), optionally saved as a .trslist.

#include <QString>
#include <QStringList>

#include <string>
#include <vector>

class QWidget;

struct CombineChoice {
    std::vector<std::string> paths;   // in the order to read them
    QString list_path;                // the .trslist written, or empty
};

// Shows the files (sorted by name, numbers in numeric order) with their
// trace counts and anything that keeps them from being combined, lets the
// user reorder, add and remove files, and choose whether to save the list.
// Returns false if cancelled.
bool showCombineDialog(QWidget* parent, const QStringList& paths, const QString& browse_dir,
                       CombineChoice& out);

// A name for the list of these files: their common name without the
// trailing number ("capture_000.trs", "capture_001.trs" → "capture"),
// or "combined".
QString traceListBaseName(const QStringList& paths);
