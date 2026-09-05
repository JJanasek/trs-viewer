#include "flow_layout.h"

#include <algorithm>

FlowLayout::FlowLayout(QWidget* parent, int margin, int h_spacing, int v_spacing)
    : QLayout(parent), h_space_(h_spacing), v_space_(v_spacing)
{
    setContentsMargins(margin, margin, margin, margin);
}

FlowLayout::~FlowLayout() {
    while (QLayoutItem* item = takeAt(0)) delete item;
}

void FlowLayout::addItem(QLayoutItem* item) { items_.push_back(item); }

int FlowLayout::count() const { return static_cast<int>(items_.size()); }

QLayoutItem* FlowLayout::itemAt(int index) const {
    return (index >= 0 && index < count()) ? items_[static_cast<size_t>(index)] : nullptr;
}

QLayoutItem* FlowLayout::takeAt(int index) {
    if (index < 0 || index >= count()) return nullptr;
    QLayoutItem* item = items_[static_cast<size_t>(index)];
    items_.erase(items_.begin() + index);
    return item;
}

int FlowLayout::heightForWidth(int width) const {
    return doLayout(QRect(0, 0, width, 0), true);
}

void FlowLayout::setGeometry(const QRect& rect) {
    QLayout::setGeometry(rect);
    doLayout(rect, false);
}

QSize FlowLayout::sizeHint() const {
    // One row's worth: what the toolbar wants when there's room for it. The
    // window is free to be narrower — minimumSize() is what actually bounds
    // it, and the rows re-flow to fit.
    QSize size(0, 0);
    for (const QLayoutItem* item : items_) {
        const QSize hint = item->sizeHint();
        size.setWidth(size.width() + hint.width() + h_space_);
        size.setHeight(std::max(size.height(), hint.height()));
    }
    const QMargins m = contentsMargins();
    return size + QSize(m.left() + m.right(), m.top() + m.bottom());
}

QSize FlowLayout::minimumSize() const {
    // Deliberately the *widest single item*, not their sum — this is the whole
    // point of using this layout for a toolbar (see flow_layout.h).
    QSize size(0, 0);
    for (const QLayoutItem* item : items_) size = size.expandedTo(item->minimumSize());
    const QMargins m = contentsMargins();
    return size + QSize(m.left() + m.right(), m.top() + m.bottom());
}

int FlowLayout::doLayout(const QRect& rect, bool test_only) const {
    const QMargins m = contentsMargins();
    const QRect eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom());
    int x = eff.x();
    int y = eff.y();
    int line_height = 0;

    for (QLayoutItem* item : items_) {
        const QSize hint = item->sizeHint();
        int next_x = x + hint.width() + h_space_;
        if (next_x - h_space_ > eff.right() && line_height > 0) {
            x = eff.x();
            y += line_height + v_space_;
            next_x = x + hint.width() + h_space_;
            line_height = 0;
        }
        if (!test_only) item->setGeometry(QRect(QPoint(x, y), hint));
        x = next_x;
        line_height = std::max(line_height, hint.height());
    }
    return y + line_height - rect.y() + m.bottom();
}
