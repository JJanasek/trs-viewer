#pragma once
// "Export Figure" dialog and clipboard copy for any PlotWidget (see figure.h).

class PlotWidget;
class QWidget;

// Opens the export dialog for the plot's current view.
void showFigureExportDialog(PlotWidget* pw, QWidget* parent);

// Puts the figure on the clipboard (SVG + 300 dpi PNG) with the last-used
// export settings — for pasting into slides and documents.
void copyFigureToClipboard(PlotWidget* pw);
