     #pragma once

#include "trs_file.h"
#include "processing.h"
#include "plot_widget.h"
#include "ttest.h"

#include <QButtonGroup>
#include <QComboBox>
#include <QLabel>
#include <QListWidget>
#include <QMainWindow>
#include <QPushButton>
#include <QSpinBox>

#include <memory>
#include <vector>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    void openFile(const QString& path);

private slots:
    void onOpenFile();
    void onApplyTraces();
    void onAddTransform();
    void onRemoveTransform();
    void onMoveTransformUp();
    void onMoveTransformDown();
    void onResetView();
    void onViewChanged(int64_t start, int64_t end, int64_t total);
    void onMeasurementUpdated(int64_t s1, double v1,
                               int64_t s2, double v2, bool has_p2);
    void onThemeChanged(int index);
    void onExportTrs();
    void onExportPng();
    void onExportPdf();
    void onRunTTest();
    void onCropEdit();
    void onRunXCorr();
    void onLoadNpyTTest();
    void onLoadNpyHeatmap();
    void onOpenNpyTraces();
    void onExportNpy();
    void onExportNpz();

private:
    void setupMenuBar();
    void updateFileInfo();
    void rebuildTransformList();
    std::shared_ptr<ITransform> createTransform(int combo_index);

    // Data
    std::unique_ptr<TrsFile> trs_file_;
    std::vector<std::shared_ptr<ITransform>> pipeline_;

    // Widgets
    PlotWidget*  plot_widget_     = nullptr;

    // Side panel
    QLabel*      lbl_file_        = nullptr;
    QLabel*      lbl_info_        = nullptr;
    QSpinBox*    spin_first_      = nullptr;
    QSpinBox*    spin_count_      = nullptr;
    QPushButton* btn_apply_       = nullptr;
    QLabel*      lbl_view_        = nullptr;
    QLabel*      lbl_measure_     = nullptr;  // measurement readout
    QComboBox*   combo_transform_ = nullptr;
    QListWidget* list_transforms_ = nullptr;
    QPushButton* btn_add_tx_      = nullptr;
    QPushButton* btn_rm_tx_       = nullptr;
    QPushButton* btn_up_tx_       = nullptr;
    QPushButton* btn_dn_tx_       = nullptr;

    // Toolbar (above plot)
    QPushButton*  btn_mode_pan_      = nullptr;
    QPushButton*  btn_mode_measure_  = nullptr;
    QPushButton*  btn_mode_box_zoom_ = nullptr;
    QPushButton*  btn_zoom_in_       = nullptr;
    QPushButton*  btn_zoom_out_     = nullptr;
    QPushButton*  btn_reset_        = nullptr;
    QComboBox*    combo_theme_      = nullptr;
    QButtonGroup* mode_group_       = nullptr;

    static const QColor TRACE_COLORS[];
    static constexpr int NUM_COLORS = 8;
};
