#include "combine_dialog.h"
#include "theme.h"
#include "trs_file.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QCollator>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QTableWidget>
#include <QVBoxLayout>

#include <algorithm>

namespace {

struct Entry {
    QString   path;
    TrsHeader h;
    QString   notes;   // e.g. cut off
    QString   error;   // could not be opened
};

Entry readEntry(const QString& path) {
    Entry e;
    e.path = path;
    TrsFile f;
    std::string err;
    if (!f.open(path.toStdString(), err)) {
        e.error = QString::fromStdString(err);
        return e;
    }
    e.h     = f.header();
    e.notes = QString::fromStdString(f.openNotes());
    return e;
}

void sortByName(std::vector<Entry>& entries) {
    QCollator c;
    c.setNumericMode(true);   // capture_2 before capture_10
    std::stable_sort(entries.begin(), entries.end(), [&c](const Entry& a, const Entry& b) {
        return c.compare(QFileInfo(a.path).fileName(), QFileInfo(b.path).fileName()) < 0;
    });
}

}  // namespace

QString traceListBaseName(const QStringList& paths) {
    if (paths.isEmpty()) return "combined";
    QString prefix = QFileInfo(paths.front()).completeBaseName();
    for (const QString& p : paths) {
        const QString n = QFileInfo(p).completeBaseName();
        int k = 0;
        while (k < prefix.size() && k < n.size() && prefix[k] == n[k]) k++;
        prefix.truncate(k);
    }
    // One file: its own name without the number. Several: what they share,
    // without a trailing part number ("…_part000", "…_part001" → "…").
    auto trim = [&prefix](bool digits) {
        while (!prefix.isEmpty() && ((digits && prefix.back().isDigit()) || QString("_-. ").contains(prefix.back())))
            prefix.chop(1);
    };
    trim(true);
    if (prefix.endsWith("part", Qt::CaseInsensitive)) { prefix.chop(4); trim(false); }
    return prefix.isEmpty() ? QString("combined") : prefix;
}

bool showCombineDialog(QWidget* parent, const QStringList& paths, const QString& browse_dir,
                       CombineChoice& out)
{
    std::vector<Entry> entries;
    for (const QString& p : paths) entries.push_back(readEntry(p));
    sortByName(entries);

    QDialog dlg(parent);
    dlg.setWindowTitle("Open as One Dataset");
    dlg.resize(760, 480);
    auto* vl = new QVBoxLayout(&dlg);

    auto* intro = new QLabel("The traces of these files are read one after another as one trace "
                             "set, without copying anything. Trace numbers run on from file to "
                             "file; alignment, patterns and every analysis use all of them.");
    intro->setWordWrap(true);
    vl->addWidget(intro);

    auto* table = new QTableWidget(0, 4);
    table->setHorizontalHeaderLabels({"File", "Traces", "Trace numbers", "Note"});
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::ExtendedSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->verticalHeader()->setVisible(false);
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setStretchLastSection(true);

    auto* btn_add    = new QPushButton("Add Files…");
    auto* btn_remove = new QPushButton("Remove");
    auto* btn_up     = new QPushButton("Move Up");
    auto* btn_down   = new QPushButton("Move Down");
    auto* btn_sort   = new QPushButton("Sort by Name");
    btn_sort->setToolTip("Numbers in names in numeric order: capture_2 before capture_10.");
    auto* side = new QVBoxLayout;
    for (QPushButton* b : {btn_add, btn_remove, btn_up, btn_down, btn_sort}) side->addWidget(b);
    side->addStretch();
    auto* mid = new QHBoxLayout;
    mid->addWidget(table, 1);
    mid->addLayout(side);
    vl->addLayout(mid, 1);

    auto* lbl_total = new QLabel;
    lbl_total->setWordWrap(true);
    vl->addWidget(lbl_total);

    QSettings st("trs-viewer", "trs-viewer");
    auto* chk_save = new QCheckBox("Save the file list, to reopen the set later:");
    chk_save->setChecked(st.value("combine/save_list", true).toBool());
    chk_save->setToolTip("A small text file (.trslist) naming these files, one per line, "
                         "relative to its folder. It appears in Recent files and opens like a "
                         ".trs file; you can also edit it by hand.");
    auto* ed_list    = new QLineEdit;
    auto* btn_browse = new QPushButton("Browse…");
    auto* save_row = new QHBoxLayout;
    save_row->addWidget(ed_list, 1);
    save_row->addWidget(btn_browse);
    vl->addWidget(chk_save);
    vl->addLayout(save_row);

    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QPushButton* btn_open = bb->button(QDialogButtonBox::Ok);
    btn_open->setText("Open");
    vl->addWidget(bb);
    QObject::connect(bb, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);

    bool list_edited = false;
    QObject::connect(ed_list, &QLineEdit::textEdited, &dlg, [&]() { list_edited = true; });
    QObject::connect(chk_save, &QCheckBox::toggled, ed_list, &QWidget::setEnabled);
    QObject::connect(chk_save, &QCheckBox::toggled, btn_browse, &QWidget::setEnabled);
    ed_list->setEnabled(chk_save->isChecked());
    btn_browse->setEnabled(chk_save->isChecked());

    const QLocale loc;
    auto refresh = [&]() {
        // The first file that opened is the reference the others must match.
        const Entry* ref = nullptr;
        for (const Entry& e : entries)
            if (e.error.isEmpty()) { ref = &e; break; }
        table->setRowCount(static_cast<int>(entries.size()));
        int64_t total = 0;
        int n_bad = 0;
        for (size_t r = 0; r < entries.size(); r++) {
            const Entry& e = entries[r];
            QString note = e.error;
            bool bad = !e.error.isEmpty();
            if (!bad && ref && &e != ref) {
                const QString why = QString::fromStdString(combineMismatch(ref->h, e.h));
                if (!why.isEmpty()) { note = "Can't be combined: " + why; bad = true; }
            }
            if (!bad && e.h.num_traces <= 0) note = "No complete traces; left out.";
            else if (!bad && !e.notes.isEmpty()) note = e.notes;
            if (!bad && ref && &e != ref && (e.h.scale_x != ref->h.scale_x || e.h.scale_y != ref->h.scale_y))
                note += QString(note.isEmpty() ? "" : " ") + "Different time or value scale than the first file.";
            const int32_t n = bad ? 0 : std::max(0, e.h.num_traces);
            auto* it_file  = new QTableWidgetItem(QFileInfo(e.path).fileName());
            auto* it_n     = new QTableWidgetItem(bad ? QString("–") : loc.toString(n));
            auto* it_range = new QTableWidgetItem(bad || n == 0 ? QString("–")
                : QString("%1 – %2").arg(loc.toString(static_cast<qlonglong>(total)),
                                         loc.toString(static_cast<qlonglong>(total + n - 1))));
            auto* it_note  = new QTableWidgetItem(note);
            it_file->setToolTip(e.path);
            it_note->setToolTip(note);
            it_n->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            const QColor c = bad ? currentTheme().danger : !note.isEmpty() ? currentTheme().warn : QColor();
            if (c.isValid()) it_note->setForeground(c);
            if (bad) it_file->setForeground(currentTheme().danger);
            table->setItem(static_cast<int>(r), 0, it_file);
            table->setItem(static_cast<int>(r), 1, it_n);
            table->setItem(static_cast<int>(r), 2, it_range);
            table->setItem(static_cast<int>(r), 3, it_note);
            if (bad) n_bad++;
            total += n;
        }
        if (entries.empty()) {
            lbl_total->setStyleSheet(themeErrorStyle());
            lbl_total->setText("Add the files to combine.");
        } else if (n_bad > 0) {
            lbl_total->setStyleSheet(themeErrorStyle());
            lbl_total->setText(QString("%1 file%2 in red can't be read together with the first one — "
                                       "remove %3 to open the rest.")
                                   .arg(n_bad).arg(n_bad == 1 ? "" : "s").arg(n_bad == 1 ? "it" : "them"));
        } else {
            lbl_total->setStyleSheet(QString());
            lbl_total->setText(QString("<b>%1 files, %2 traces</b> × %3 samples.")
                                   .arg(entries.size())
                                   .arg(loc.toString(static_cast<qlonglong>(total)))
                                   .arg(ref ? loc.toString(ref->h.num_samples) : QString("?")));
        }
        btn_open->setEnabled(!entries.empty() && n_bad == 0 && total > 0 && total <= INT32_MAX);
        if (!list_edited && !entries.empty()) {
            QStringList ps;
            for (const Entry& e : entries) ps << e.path;
            ed_list->setText(QDir(QFileInfo(entries.front().path).absolutePath())
                                 .filePath(traceListBaseName(ps) + ".trslist"));
        }
    };

    auto selectedRows = [&]() {
        std::vector<int> rows;
        for (const QModelIndex& i : table->selectionModel()->selectedRows()) rows.push_back(i.row());
        std::sort(rows.begin(), rows.end());
        return rows;
    };
    auto select = [&](const std::vector<int>& rows) {
        table->clearSelection();
        for (int r : rows) table->selectRow(r);
    };
    // Moves the selected rows one step (dir −1 up, +1 down), keeping them selected.
    auto move = [&](int dir) {
        std::vector<int> rows = selectedRows();
        if (rows.empty()) return;
        if (dir < 0 && rows.front() == 0) return;
        if (dir > 0 && rows.back() == static_cast<int>(entries.size()) - 1) return;
        if (dir > 0) std::reverse(rows.begin(), rows.end());
        for (int& r : rows) {
            std::swap(entries[static_cast<size_t>(r)], entries[static_cast<size_t>(r + dir)]);
            r += dir;
        }
        refresh();
        select(rows);
    };
    QObject::connect(btn_up,   &QPushButton::clicked, &dlg, [&]() { move(-1); });
    QObject::connect(btn_down, &QPushButton::clicked, &dlg, [&]() { move(+1); });
    QObject::connect(btn_sort, &QPushButton::clicked, &dlg, [&]() { sortByName(entries); refresh(); });
    QObject::connect(btn_remove, &QPushButton::clicked, &dlg, [&]() {
        std::vector<int> rows = selectedRows();
        for (auto it = rows.rbegin(); it != rows.rend(); ++it)
            entries.erase(entries.begin() + *it);
        refresh();
    });
    QObject::connect(btn_add, &QPushButton::clicked, &dlg, [&]() {
        const QString dir = entries.empty() ? browse_dir : QFileInfo(entries.back().path).absolutePath();
        const QStringList more = QFileDialog::getOpenFileNames(&dlg, "Add TRS files", dir,
                                                               "TRS files (*.trs);;All files (*)");
        for (const QString& p : more) {
            const QString abs = QFileInfo(p).absoluteFilePath();
            const bool dup = std::any_of(entries.begin(), entries.end(), [&](const Entry& e) {
                return QFileInfo(e.path).absoluteFilePath() == abs;
            });
            if (!dup) entries.push_back(readEntry(p));
        }
        refresh();
    });
    QObject::connect(btn_browse, &QPushButton::clicked, &dlg, [&]() {
        const QString p = QFileDialog::getSaveFileName(&dlg, "Save file list as", ed_list->text(),
                                                       "Trace lists (*.trslist)");
        if (p.isEmpty()) return;
        ed_list->setText(p.endsWith(".trslist", Qt::CaseInsensitive) ? p : p + ".trslist");
        list_edited = true;
    });
    QObject::connect(bb, &QDialogButtonBox::accepted, &dlg, [&]() {
        if (chk_save->isChecked()) {
            QString lp = ed_list->text().trimmed();
            if (lp.isEmpty()) {
                QMessageBox::information(&dlg, "Open as One Dataset", "Choose where to save the file list, "
                                         "or untick saving it.");
                return;
            }
            if (!lp.endsWith(".trslist", Qt::CaseInsensitive)) lp += ".trslist";
            if (QFileInfo::exists(lp)
                && QMessageBox::question(&dlg, "Open as One Dataset",
                       QString("%1 already exists. Replace it?").arg(QFileInfo(lp).fileName()))
                       != QMessageBox::Yes)
                return;
            ed_list->setText(lp);
        }
        dlg.accept();
    });

    refresh();
    if (dlg.exec() != QDialog::Accepted) return false;
    st.setValue("combine/save_list", chk_save->isChecked());

    out = CombineChoice{};
    for (const Entry& e : entries)
        if (e.h.num_traces > 0) out.paths.push_back(e.path.toStdString());
    if (chk_save->isChecked()) {
        std::string err;
        if (writeTraceList(ed_list->text().toStdString(), out.paths, err))
            out.list_path = ed_list->text();
        else
            QMessageBox::warning(parent, "Open as One Dataset",
                                 QString("The file list could not be saved (%1). The files are opened "
                                         "anyway, but the set can't be reopened from Recent files.")
                                     .arg(QString::fromStdString(err)));
    }
    return true;
}
