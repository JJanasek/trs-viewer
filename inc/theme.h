#pragma once
// "Graphite" app theme: one set of colour tokens per light/dark variant,
// applied app-wide as a QPalette (Fusion style) plus a small stylesheet, and
// mirrored into PlotTheme so plots match the widgets around them.

#include <QColor>
#include <QIcon>
#include <QString>
#include <array>

struct ThemeTokens {
    QColor window, panel, raised, input;
    QColor border, border_strong;
    QColor text, muted;
    QColor accent, accent_bg;
    QColor warn, warn_bg;        // "baked"
    QColor success, success_bg;  // "aligned"
    QColor result, result_bg;    // result tabs
    QColor danger;
    std::array<QColor, 8> traces;  // Okabe–Ito (+2), lightness tuned per variant
};

enum class ThemeChoice { Dark = 0, Light = 1, System = 2 };

const ThemeTokens& themeTokens(bool dark);

// The tokens currently applied (dark or light).
const ThemeTokens& currentTheme();
bool currentThemeIsDark();

// Resolves System against the desktop's colour scheme.
bool themeChoiceIsDark(ThemeChoice c);

// Applies palette + stylesheet to the whole application and makes
// PlotTheme::current() return the matching plot theme.
void applyAppTheme(bool dark);

// Trace colour for trace index i in the current theme.
QColor themeTraceColor(int i);

// Ready-made style sheets for the few widgets that need an explicit colour
// (built from the current theme when the widget is created).
QString themeNoteStyle();    // amber "measure before and after" call-out box
QString themeHintStyle();    // small muted hint text
QString themeErrorStyle();   // small danger-coloured text

class QAbstractButton;

// Line icon from :/resources/icons/<name>.svg, drawn in the current theme's
// text colour (muted when disabled).
QIcon themeIcon(const QString& name);

// Like button->setIcon(themeIcon(name)), but the icon is redrawn whenever
// the theme changes.
void setThemedIcon(QAbstractButton* button, const QString& name);
