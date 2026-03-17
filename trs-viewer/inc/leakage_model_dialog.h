#pragma once

#include "leakage_model.h"
#include "trs_file.h"

#include <QComboBox>
#include <QDialog>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QTextEdit>

#include <memory>

// Minimal Python syntax highlighter for the code editor.
class PythonHighlighter : public QSyntaxHighlighter {
    Q_OBJECT
public:
    explicit PythonHighlighter(QTextDocument* doc);
protected:
    void highlightBlock(const QString& text) override;
private:
    struct Rule { QRegularExpression pat; QTextCharFormat fmt; };
    QList<Rule> rules_;
};

// Dialog for editing and testing a Python leakage model before launching CPA.
// On accept(), compiledModel() returns the verified LeakageModel.
class LeakageModelDialog : public QDialog {
    Q_OBJECT
public:
    explicit LeakageModelDialog(TrsFile* file, int32_t first_trace,
                                int32_t n_preview, QWidget* parent = nullptr);

    QString      code() const;
    LeakageModel* compiledModel() { return model_.get(); }

private slots:
    void onTest();
    void onLibrarySelected(int index);
    void onSaveToLibrary();
    void onDeleteFromLibrary();
    void onLoad();   // load from arbitrary path
    void onSave();   // save to arbitrary path

private:
    static QString modelsDir();         // ~/.local/share/trs-viewer/models
    void           refreshLibrary();    // repopulate combo_library_

    TrsFile* file_;
    int32_t  first_trace_;
    int32_t  n_preview_;

    QPlainTextEdit* editor_          = nullptr;
    QTextEdit*      console_         = nullptr;
    QPushButton*    btn_run_         = nullptr;
    QComboBox*      combo_library_   = nullptr;
    QPushButton*    btn_lib_delete_  = nullptr;

    std::unique_ptr<LeakageModel> model_;

    static const QString BOILERPLATE;
};
