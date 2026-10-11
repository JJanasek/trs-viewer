#include "theme.h"

#include <QApplication>
#include <QGuiApplication>
#include <QPalette>
#include <QStyleFactory>
#include <QStyleHints>

namespace {

const ThemeTokens kDark = {
    QColor("#15181c"), QColor("#1b1f24"), QColor("#242931"), QColor("#101317"),
    QColor("#313842"), QColor("#3d4550"),
    QColor("#e2e5e9"), QColor("#949ea9"),
    QColor("#4c9be8"), QColor("#1d3550"),
    QColor("#e3a33b"), QColor("#3a2c13"),
    QColor("#4cc38a"), QColor("#15342a"),
    QColor("#a78bfa"), QColor("#1f1c2f"),
    QColor("#ef6b63"),
    {QColor("#56B4E9"), QColor("#E69F00"), QColor("#2CC09A"),
     QColor("#F07A3A"), QColor("#D98CB8"), QColor("#F0E442"),
     QColor("#B9C2CC"), QColor("#8A9BFF")},
};

const ThemeTokens kLight = {
    QColor("#eceef1"), QColor("#f6f7f9"), QColor("#ffffff"), QColor("#ffffff"),
    QColor("#d3d8de"), QColor("#b9c0c8"),
    QColor("#1b1f24"), QColor("#5b6570"),
    QColor("#1f6bc2"), QColor("#dceafb"),
    QColor("#9a5a00"), QColor("#fbefd6"),
    QColor("#1b7a4b"), QColor("#dff3e8"),
    QColor("#6247c4"), QColor("#f3f0fd"),
    QColor("#c0362c"),
    // Yellow fails contrast on white, so the 6th slot is slate here.
    {QColor("#0072B2"), QColor("#C27C00"), QColor("#008F68"),
     QColor("#D55E00"), QColor("#B5639A"), QColor("#5D6B78"),
     QColor("#1B1F24"), QColor("#3A86C8")},
};

bool g_dark = true;

// Kept deliberately small: the palette does most of the work under Fusion,
// and styling inputs (spin boxes, combos) via QSS would replace their native
// sub-controls. Tab text colour is left to the palette so per-tab colours
// (result tabs, set with setTabTextColor) still show. @tokens are
// substituted in applyAppTheme().
const char* kQss = R"QSS(
QGroupBox {
    border: 1px solid @border;
    border-radius: 6px;
    margin-top: 14px;
    padding: 8px 6px 6px 6px;
    background: @panel;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 3px;
    color: @muted;
    font-weight: 600;
}
QPushButton {
    background: @raised;
    border: 1px solid @border_strong;
    border-radius: 4px;
    padding: 3px 10px;
    min-height: 20px;
}
QPushButton:hover    { border-color: @accent; }
QPushButton:pressed  { background: @accent_bg; }
QPushButton:checked  { background: @accent_bg; border-color: @accent; }
QPushButton:default  { background: @accent; color: #ffffff; border-color: @accent; }
QPushButton:disabled { color: @muted; background: @panel; border-color: @border; }
QTabBar::tab {
    background: transparent;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: @panel;
    border-color: @border;
    border-bottom: 2px solid @accent;
}
QTabBar::tab:hover:!selected { background: @panel; }
QStatusBar { background: @panel; border-top: 1px solid @border; }
QStatusBar QLabel { padding: 0 4px; }
QToolTip {
    background: @raised;
    color: @text;
    border: 1px solid @border_strong;
    padding: 4px;
}
QProgressBar {
    border: 1px solid @border;
    border-radius: 4px;
    background: @input;
    text-align: center;
    min-height: 16px;
}
QProgressBar::chunk { background: @accent; border-radius: 3px; }
QMenu { border: 1px solid @border_strong; }
QMenu::separator { height: 1px; background: @border; margin: 4px 8px; }
QScrollArea { background: transparent; }
QDockWidget::title { background: @panel; padding: 4px; }
)QSS";

QString buildQss(const ThemeTokens& t) {
    QString s = QString::fromUtf8(kQss);
    const std::pair<const char*, QColor> subs[] = {
        // Longest names first so "@border_strong" isn't eaten by "@border".
        {"@border_strong", t.border_strong}, {"@accent_bg", t.accent_bg},
        {"@border", t.border}, {"@window", t.window}, {"@panel", t.panel},
        {"@raised", t.raised}, {"@input", t.input}, {"@text", t.text},
        {"@muted", t.muted}, {"@accent", t.accent},
    };
    for (const auto& [k, c] : subs) s.replace(QLatin1String(k), c.name());
    return s;
}

QPalette buildPalette(const ThemeTokens& t) {
    QPalette p;
    p.setColor(QPalette::Window,          t.window);
    p.setColor(QPalette::WindowText,      t.text);
    p.setColor(QPalette::Base,            t.input);
    p.setColor(QPalette::AlternateBase,   t.panel);
    p.setColor(QPalette::Text,            t.text);
    p.setColor(QPalette::Button,          t.raised);
    p.setColor(QPalette::ButtonText,      t.text);
    p.setColor(QPalette::BrightText,      t.danger);
    p.setColor(QPalette::Highlight,       t.accent);
    p.setColor(QPalette::HighlightedText, QColor("#ffffff"));
    p.setColor(QPalette::ToolTipBase,     t.raised);
    p.setColor(QPalette::ToolTipText,     t.text);
    p.setColor(QPalette::PlaceholderText, t.muted);
    p.setColor(QPalette::Link,            t.accent);
    p.setColor(QPalette::LinkVisited,     t.result);
    p.setColor(QPalette::Light,           t.raised.lighter(115));
    p.setColor(QPalette::Midlight,        t.raised);
    p.setColor(QPalette::Mid,             t.border_strong);
    p.setColor(QPalette::Dark,            t.border);
    p.setColor(QPalette::Shadow,          QColor(0, 0, 0, 120));
    for (auto role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText})
        p.setColor(QPalette::Disabled, role, t.muted);
    p.setColor(QPalette::Disabled, QPalette::Base, t.panel);
    p.setColor(QPalette::Disabled, QPalette::Button, t.panel);
    return p;
}

}  // namespace

const ThemeTokens& themeTokens(bool dark) { return dark ? kDark : kLight; }
const ThemeTokens& currentTheme() { return themeTokens(g_dark); }
bool currentThemeIsDark() { return g_dark; }

bool themeChoiceIsDark(ThemeChoice c) {
    if (c == ThemeChoice::Dark)  return true;
    if (c == ThemeChoice::Light) return false;
    return QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark;
}

void applyAppTheme(bool dark) {
    g_dark = dark;
    const ThemeTokens& t = currentTheme();
    if (QStyle* fusion = QStyleFactory::create("Fusion"))
        QApplication::setStyle(fusion);
    QApplication::setPalette(buildPalette(t));
    qApp->setStyleSheet(buildQss(t));
}

QColor themeTraceColor(int i) {
    const auto& tr = currentTheme().traces;
    const int n = static_cast<int>(tr.size());
    return tr[static_cast<size_t>(((i % n) + n) % n)];
}
