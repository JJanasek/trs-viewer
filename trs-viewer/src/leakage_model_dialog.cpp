#include "leakage_model_dialog.h"

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMessageBox>
#include <QRegularExpression>
#include <QSplitter>
#include <QStandardPaths>
#include <QTextCharFormat>
#include <QTextDocument>
#include <QTextStream>
#include <QVBoxLayout>

// ---- PythonHighlighter ----

PythonHighlighter::PythonHighlighter(QTextDocument* doc) : QSyntaxHighlighter(doc) {
    QTextCharFormat kwFmt, strFmt, commentFmt, numFmt, funcFmt;
    kwFmt.setForeground(QColor("#c792ea")); kwFmt.setFontWeight(QFont::Bold);
    strFmt.setForeground(QColor("#c3e88d"));
    commentFmt.setForeground(QColor("#546e7a")); commentFmt.setFontItalic(true);
    numFmt.setForeground(QColor("#f78c6c"));
    funcFmt.setForeground(QColor("#82aaff"));

    static const QStringList keywords = {
        "def", "return", "import", "from", "as", "if", "else", "elif",
        "for", "in", "while", "True", "False", "None", "not", "and",
        "or", "lambda", "class", "pass", "break", "continue", "with"
    };
    for (const auto& kw : keywords) {
        Rule r;
        r.pat = QRegularExpression(QString("\\b%1\\b").arg(kw));
        r.fmt = kwFmt;
        rules_.append(r);
    }
    { Rule r; r.pat = QRegularExpression("\"[^\"]*\"|'[^']*'"); r.fmt = strFmt; rules_.append(r); }
    { Rule r; r.pat = QRegularExpression("\\b[0-9]+(\\.[0-9]*)?\\b"); r.fmt = numFmt; rules_.append(r); }
    { Rule r; r.pat = QRegularExpression("\\b[A-Za-z_][A-Za-z0-9_]*(?=\\s*\\()"); r.fmt = funcFmt; rules_.append(r); }
    { Rule r; r.pat = QRegularExpression("#[^\n]*"); r.fmt = commentFmt; rules_.append(r); }
}

void PythonHighlighter::highlightBlock(const QString& text) {
    for (const auto& rule : rules_) {
        auto it = rule.pat.globalMatch(text);
        while (it.hasNext()) {
            auto m = it.next();
            setFormat(m.capturedStart(), m.capturedLength(), rule.fmt);
        }
    }
}

// ---- LeakageModelDialog ----

const QString LeakageModelDialog::BOILERPLATE = R"(import numpy as np

# AES S-Box
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

# Hamming weight lookup (0..255 -> popcount)
HW = np.array([bin(x).count('1') for x in range(256)], dtype=np.float32)


def get_leakages(plaintexts, ciphertexts, key_guess):
    """
    plaintexts : uint8 array (n_traces, data_len)
                 Convention: first 16 bytes = plaintext, next 16 = ciphertext
    ciphertexts: uint8 array (n_traces, 0)  -- unused placeholder
    key_guess  : int 0-255

    Returns    : float32 array (n_traces,)
    """
    pt = plaintexts[:, :16]    # first 16 bytes: AES plaintext
    # ct = plaintexts[:, 16:]  # next 16 bytes:  AES ciphertext (if needed)

    # Hamming weight of AES S-Box( PT[0] ^ KEY ) -- SubBytes output for byte 0
    return HW[SBOX[pt[:, 0] ^ key_guess]]
)";

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

QString LeakageModelDialog::modelsDir() {
    QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation)
                  + "/models";
    QDir().mkpath(dir);
    return dir;
}

void LeakageModelDialog::refreshLibrary() {
    const QString current = combo_library_->currentText();
    combo_library_->blockSignals(true);
    combo_library_->clear();
    combo_library_->addItem("— select saved model —");

    QDir d(modelsDir());
    const auto files = d.entryList({"*.py"}, QDir::Files, QDir::Name);
    for (const auto& f : files)
        combo_library_->addItem(QFileInfo(f).completeBaseName(), d.filePath(f));

    // Restore previous selection if it still exists
    int idx = combo_library_->findText(current);
    combo_library_->setCurrentIndex(idx >= 1 ? idx : 0);
    combo_library_->blockSignals(false);

    btn_lib_delete_->setEnabled(combo_library_->currentIndex() >= 1);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

LeakageModelDialog::LeakageModelDialog(TrsFile* file, int32_t first_trace,
                                       int32_t n_preview, QWidget* parent)
    : QDialog(parent), file_(file), first_trace_(first_trace), n_preview_(n_preview)
{
    setWindowTitle("CPA — Leakage Model Editor");
    resize(1000, 700);

    // ---- Library bar (top) ----
    combo_library_ = new QComboBox;
    combo_library_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    combo_library_->setToolTip("Saved models in " + modelsDir());

    auto* btn_lib_save   = new QPushButton("Save to library…");
    btn_lib_delete_      = new QPushButton("Delete");
    btn_lib_delete_->setEnabled(false);

    auto* lib_row = new QHBoxLayout;
    lib_row->addWidget(new QLabel("Library:"));
    lib_row->addWidget(combo_library_, 1);
    lib_row->addWidget(btn_lib_save);
    lib_row->addWidget(btn_lib_delete_);

    refreshLibrary();

    // ---- Left pane: code editor ----
    auto* left_box = new QGroupBox("Leakage Model (Python)");
    auto* left_vl  = new QVBoxLayout(left_box);
    editor_ = new QPlainTextEdit;
    editor_->setFont(QFont("Monospace", 10));
    editor_->setPlainText(BOILERPLATE);
    new PythonHighlighter(editor_->document());
    left_vl->addWidget(editor_);

    // ---- Right pane: info + test + console ----
    auto* right_box = new QGroupBox("Inspector");
    auto* right_vl  = new QVBoxLayout(right_box);

    QString info;
    if (file_) {
        const auto& h = file_->header();
        info = QString("Traces: %1   Data bytes: %2   Samples: %3   Preview: first %4 traces")
               .arg(h.num_traces).arg(h.data_length).arg(h.num_samples).arg(n_preview_);
    } else {
        info = "(no file loaded)";
    }
    auto* lbl_info = new QLabel(info);
    lbl_info->setWordWrap(true);
    right_vl->addWidget(lbl_info);

    auto* btn_test = new QPushButton("Run  Test Model  (first 5 traces, key_guess = 0)");
    right_vl->addWidget(btn_test);

    console_ = new QTextEdit;
    console_->setReadOnly(true);
    console_->setFont(QFont("Monospace", 9));
    console_->setStyleSheet("background: #1e1e2e; color: #cdd6f4;");
    console_->setPlaceholderText("Test output will appear here...");
    right_vl->addWidget(console_);

    // ---- Splitter ----
    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(left_box);
    splitter->addWidget(right_box);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);

    // ---- Bottom bar ----
    auto* btn_load = new QPushButton("Load .py…");
    auto* btn_save = new QPushButton("Save .py…");
    btn_run_ = new QPushButton("Run CPA");
    btn_run_->setEnabled(false);
    btn_run_->setStyleSheet("QPushButton:enabled { font-weight: bold; background: #4CAF50; color: white; }");
    auto* btn_cancel = new QPushButton("Cancel");

    auto* bottom = new QHBoxLayout;
    bottom->addWidget(btn_load);
    bottom->addWidget(btn_save);
    bottom->addStretch();
    bottom->addWidget(btn_run_);
    bottom->addWidget(btn_cancel);

    // ---- Main layout ----
    auto* vl = new QVBoxLayout(this);
    vl->addLayout(lib_row);
    vl->addWidget(splitter, 1);
    vl->addLayout(bottom);

    // ---- Connections ----
    connect(combo_library_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &LeakageModelDialog::onLibrarySelected);
    connect(btn_lib_save,   &QPushButton::clicked, this, &LeakageModelDialog::onSaveToLibrary);
    connect(btn_lib_delete_,&QPushButton::clicked, this, &LeakageModelDialog::onDeleteFromLibrary);
    connect(btn_test,       &QPushButton::clicked, this, &LeakageModelDialog::onTest);
    connect(btn_load,       &QPushButton::clicked, this, &LeakageModelDialog::onLoad);
    connect(btn_save,       &QPushButton::clicked, this, &LeakageModelDialog::onSave);
    connect(btn_run_,       &QPushButton::clicked, this, &QDialog::accept);
    connect(btn_cancel,     &QPushButton::clicked, this, &QDialog::reject);
}

QString LeakageModelDialog::code() const {
    return editor_->toPlainText();
}

// ---------------------------------------------------------------------------
// Library slots
// ---------------------------------------------------------------------------

void LeakageModelDialog::onLibrarySelected(int index) {
    btn_lib_delete_->setEnabled(index >= 1);
    if (index < 1) return;
    const QString path = combo_library_->itemData(index).toString();
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    editor_->setPlainText(QTextStream(&f).readAll());
    btn_run_->setEnabled(false);
    model_.reset();
    console_->clear();
}

void LeakageModelDialog::onSaveToLibrary() {
    bool ok = false;
    // Suggest the current library selection name or a default
    QString suggest = combo_library_->currentIndex() >= 1
                      ? combo_library_->currentText()
                      : "my_model";
    QString name = QInputDialog::getText(this, "Save to library",
                                         "Model name (no spaces, no .py):",
                                         QLineEdit::Normal, suggest, &ok);
    if (!ok || name.trimmed().isEmpty()) return;

    // Sanitise: keep only alphanumeric, dash, underscore
    name = name.trimmed().replace(QRegularExpression("[^A-Za-z0-9_\\-]"), "_");
    const QString path = modelsDir() + "/" + name + ".py";

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(this, "Save failed", "Could not write to:\n" + path);
        return;
    }
    QTextStream(&f) << editor_->toPlainText();
    refreshLibrary();

    // Select the just-saved model in the combo
    int idx = combo_library_->findText(name);
    if (idx >= 0) combo_library_->setCurrentIndex(idx);
}

void LeakageModelDialog::onDeleteFromLibrary() {
    const int idx = combo_library_->currentIndex();
    if (idx < 1) return;
    const QString name = combo_library_->currentText();
    const QString path = combo_library_->itemData(idx).toString();
    if (QMessageBox::question(this, "Delete model",
            QString("Delete '%1' from the library?").arg(name),
            QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes)
        return;
    QFile::remove(path);
    refreshLibrary();
}

// ---------------------------------------------------------------------------
// Arbitrary-path load / save
// ---------------------------------------------------------------------------

void LeakageModelDialog::onLoad() {
    QString path = QFileDialog::getOpenFileName(
        this, "Load leakage model", modelsDir(),
        "Python files (*.py);;All files (*)");
    if (path.isEmpty()) return;
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    editor_->setPlainText(QTextStream(&f).readAll());
    btn_run_->setEnabled(false);
    model_.reset();
    console_->clear();

    // If the file lives inside the models dir, select it in the combo
    const QString dir = QFileInfo(path).absolutePath();
    if (dir == QDir(modelsDir()).absolutePath()) {
        refreshLibrary();
        const QString stem = QFileInfo(path).completeBaseName();
        int idx = combo_library_->findText(stem);
        if (idx >= 0) {
            combo_library_->blockSignals(true);
            combo_library_->setCurrentIndex(idx);
            combo_library_->blockSignals(false);
            btn_lib_delete_->setEnabled(true);
        }
    }
}

void LeakageModelDialog::onSave() {
    QString path = QFileDialog::getSaveFileName(
        this, "Save leakage model", modelsDir(),
        "Python files (*.py);;All files (*)");
    if (path.isEmpty()) return;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream(&f) << editor_->toPlainText();

    // Auto-refresh if saved into the models dir
    if (QFileInfo(path).absolutePath() == QDir(modelsDir()).absolutePath())
        refreshLibrary();
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

void LeakageModelDialog::onTest() {
    btn_run_->setEnabled(false);
    console_->clear();
    model_.reset();

    std::string err;
    if (!LeakageModel::isInitialized()) {
        if (!LeakageModel::globalInit(err)) {
            console_->setTextColor(QColor("#f38ba8"));
            console_->append("Compile error — Python init failed: " + QString::fromStdString(err));
            return;
        }
    }

    auto model = std::make_unique<LeakageModel>();
    if (!model->compile(editor_->toPlainText(), err)) {
        console_->setTextColor(QColor("#f38ba8"));
        console_->append("Compile error:\n" + QString::fromStdString(err));
        return;
    }

    if (!file_) {
        console_->setTextColor(QColor("#fab387"));
        console_->append("Warning: No file loaded — compile succeeded but cannot test with real data.");
        model_ = std::move(model);
        btn_run_->setEnabled(true);
        return;
    }

    const int n_test = std::min(5, file_->header().num_traces - first_trace_);
    const int DL     = file_->header().data_length;

    std::vector<uint8_t> data_flat(static_cast<size_t>(n_test) * std::max(DL, 0), 0);
    for (int i = 0; i < n_test; i++) {
        auto d = file_->readData(first_trace_ + i);
        size_t copy = std::min<size_t>(d.size(), static_cast<size_t>(DL));
        if (copy > 0) memcpy(data_flat.data() + i * DL, d.data(), copy);
    }

    std::vector<float> out;
    if (!model->evaluate(data_flat, DL, n_test, 0, out, err)) {
        console_->setTextColor(QColor("#f38ba8"));
        console_->append("Runtime error:\n" + QString::fromStdString(err));
        return;
    }

    QString preview = "[";
    for (int i = 0; i < (int)out.size(); i++) {
        if (i) preview += ", ";
        preview += QString::number(static_cast<double>(out[i]), 'g', 4);
    }
    preview += "]";

    console_->setTextColor(QColor("#a6e3a1"));
    console_->append("Success");
    console_->setTextColor(QColor("#cdd6f4"));
    console_->append(QString("   Output shape : (%1,)").arg(out.size()));
    console_->append(QString("   dtype         : float32"));
    console_->append(QString("   Preview       : %1").arg(preview));

    model_ = std::move(model);
    btn_run_->setEnabled(true);
}
