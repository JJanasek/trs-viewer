#pragma once

// ---------------------------------------------------------------------------
// Left-to-right layout that wraps onto a new row when it runs out of width,
// the way text does.
//
// The result-tab toolbars (t-test especially, ~19 controls) used a plain
// QHBoxLayout, which never wraps: its minimum width is the sum of every
// child's, that minimum propagates up through the central widget to the main
// window, and Qt then grows the window to satisfy it — past the edge of the
// screen, with no way to shrink it back. With this layout the minimum width
// is just the widest single control; everything else moves to another row.
// ---------------------------------------------------------------------------

#include <QLayout>
#include <QRect>
#include <QSize>

#include <vector>

class FlowLayout : public QLayout {
public:
    explicit FlowLayout(QWidget* parent = nullptr, int margin = 0,
                         int h_spacing = 6, int v_spacing = 4);
    ~FlowLayout() override;

    void         addItem(QLayoutItem* item) override;
    int          count() const override;
    QLayoutItem* itemAt(int index) const override;
    QLayoutItem* takeAt(int index) override;

    Qt::Orientations expandingDirections() const override { return {}; }
    bool  hasHeightForWidth() const override { return true; }
    int   heightForWidth(int width) const override;
    QSize sizeHint() const override;
    QSize minimumSize() const override;
    void  setGeometry(const QRect& rect) override;

private:
    // Lays the items out inside `rect`; returns the total height needed.
    // test_only = measure without actually moving anything (heightForWidth).
    int doLayout(const QRect& rect, bool test_only) const;

    std::vector<QLayoutItem*> items_;
    int h_space_;
    int v_space_;
};
