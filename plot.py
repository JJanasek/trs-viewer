#!/usr/bin/env python3
"""
Ultra-large .trs powertrace viewer (pyqtgraph, lazy loading).
- Only loads visible window from disk (OOM-safe for 100M+ samples).
- Accurate on zoom: slice is read fresh per view and decimated with 'peak'.
- No Matplotlib, no browser. Fast native GUI.

Usage:
  python plot_trs_lazy.py file.trs          # first 10 traces
  python plot_trs_lazy.py file.trs -n 2     # first 2 traces
"""

import argparse
import math
import os
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pg.setConfigOptions(
    useOpenGL=False,       # set False if your stack has GL issues
    antialias=False,
    foreground='k',
    background='w',
)

# Pens
_PENS = [
    pg.mkPen('#1f77b4', width=1.0),
    pg.mkPen('#d62728', width=1.0),
    pg.mkPen('#2ca02c', width=1.0),
    pg.mkPen('#ff7f0e', width=1.0),
    pg.mkPen('#9467bd', width=1.0),
    pg.mkPen('#8c564b', width=1.0),
    pg.mkPen('#e377c2', width=1.0),
    pg.mkPen('#7f7f7f', width=1.0),
    pg.mkPen('#bcbd22', width=1.0),
    pg.mkPen('#17becf', width=1.0),
]

# How many points we ever push to the GPU per curve for a single render
# (raise a bit if you have a beefy machine; keep ~1-2M for snappy pans).
_MAX_POINTS_OVERVIEW = 2_000_000
_MAX_POINTS_VISIBLE  = 5_000_000
# Margin to prefetch beyond current view (in samples)
_PREFETCH_MARGIN_FRAC = 0.15   # 15% of view width


class LazyTrace:
    """Plot a huge trace by loading only what is needed for the current view."""
    def __init__(self, ts, trace_index: int, plot: pg.PlotItem, pen, name: Optional[str] = None):
        tr = ts[trace_index]
        self.tr = tr
        self.N = int(len(tr.samples))
        self.plot = plot
        self.pen  = pen
        self.name = name or f"trace {trace_index}"

        # PlotDataItem (supports autoDownsample/peak & y-only or x+y)
        self.item = plot.plot(pen=self.pen, name=self.name, connect='finite')
        try:
            self.item.setSkipFiniteCheck(True)
        except Exception:
            pass

        # Debounced viewport updates
        self._pending = False
        self._debounce = QtCore.QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(self._update_for_current_view)

        # Initial overview
        self._show_overview()

        # React to viewport changes
        vb: pg.ViewBox = self.plot.getViewBox()
        vb.sigXRangeChanged.connect(self._schedule_update)

    def _show_overview(self):
        step = max(1, self.N // _MAX_POINTS_OVERVIEW)
        # NOTE: read with stride to keep memory small
        y = np.asarray(self.tr.samples[::step], dtype=np.float32)
        x = np.arange(0, self.N, step, dtype=np.int64)
        self._set_item_data(x, y)

    def _schedule_update(self, *_):
        if not self._pending:
            self._pending = True
            self._debounce.start()

    def _update_for_current_view(self):
        self._pending = False
        vb: pg.ViewBox = self.plot.getViewBox()
        xlo, xhi = vb.viewRange()[0]
        if not np.isfinite([xlo, xhi]).all():
            return

        # Convert view range to integer sample indices
        start = max(0, int(math.floor(xlo)))
        end   = min(self.N, int(math.ceil(xhi)))
        if end <= start:
            return

        # Prefetch margin
        width = end - start
        margin = max(0, int(width * _PREFETCH_MARGIN_FRAC))
        s = max(0, start - margin)
        e = min(self.N, end + margin)

        # Downsample to cap
        span = e - s
        step = max(1, span // _MAX_POINTS_VISIBLE)

        # Slice from trsfile; convert to float32 to minimize memory
        y = np.asarray(self.tr.samples[s:e:step], dtype=np.float32)
        if step == 1:
            x = np.arange(s, e, dtype=np.int64)
        else:
            x = np.arange(s, e, step, dtype=np.int64)

        self._set_item_data(x, y)

    def _set_item_data(self, x: np.ndarray, y: np.ndarray):
        # peak-preserving downsample handled by pyqtgraph rendering too,
        # but we already bounded points; still request peak to keep spikes
        self.item.setData(x=x, y=y, autoDownsample=True, downsampleMethod='peak')


def plot_trsfile(ts_or_path, indices, title: Optional[str] = None) -> pg.PlotItem:
    import trsfile
    ts = ts_or_path
    if isinstance(ts_or_path, (str, os.PathLike)):
        ts = trsfile.open(str(ts_or_path), "r")

    valid = [i for i in map(int, indices) if 0 <= i < len(ts)]
    if not valid:
        raise IndexError(f"No valid trace indices in {list(indices)} (0..{len(ts)-1}).")

    plot = pg.PlotItem()
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.setLabel('bottom', 'sample')
    plot.setLabel('left', 'value')
    if title:
        plot.setTitle(title)

    for j, idx in enumerate(valid):
        LazyTrace(ts, idx, plot, _PENS[j % len(_PENS)], name=f"trace {idx}")

    # Let the view auto-range once; it will show overview then refine on zoom
    plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
    return plot


def install_shortcuts(win: pg.GraphicsLayoutWidget, plot: pg.PlotItem):
    # Ctrl+S → export PNG (and SVG if exporter available)
    sc_save = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), win)
    def do_save():
        base, _ = QtWidgets.QFileDialog.getSaveFileName(win, "Save image (basename)", "", "PNG (*.png)")
        if not base:
            return
        base = os.path.splitext(base)[0]
        try:
            from pyqtgraph.exporters import ImageExporter, SVGExporter
            exp = ImageExporter(win.scene()); exp.parameters()['width'] = 2400; exp.export(base + ".png")
            try:
                SVGExporter(win.scene()).export(base + ".svg")
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Export failed", str(e))
    sc_save.activated.connect(do_save)

    # A → autorange
    QtGui.QShortcut(QtGui.QKeySequence("A"), win).activated.connect(
        lambda: plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
    )
    # R → reset (autoRange now)
    QtGui.QShortcut(QtGui.QKeySequence("R"), win).activated.connect(
        lambda: plot.getViewBox().autoRange()
    )


def main():
    parser = argparse.ArgumentParser(description="Plot huge .trs powertraces (lazy, zoom-accurate).")
    parser.add_argument("filename", type=str, help="Path to the .trs file")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of traces to plot (default: 10)")
    args = parser.parse_args()

    import trsfile
    ts = trsfile.open(args.filename, "r")
    n = min(max(1, args.num), len(ts))

    app = QtWidgets.QApplication([])
    
    # --- 1. Create a main window and layout ---
    # We use a standard QWidget as the main window
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"{os.path.basename(args.filename)} — first {n} traces")
    win.resize(2000, 1200)
    # Main layout is vertical (toolbar on top, plot below)
    main_layout = QtWidgets.QVBoxLayout()
    win.setLayout(main_layout)

    # --- 2. Create the plot widget itself ---
    # This GraphicsLayoutWidget holds the plot
    plot_widget = pg.GraphicsLayoutWidget()
    plot = plot_trsfile(ts, indices=range(n), title=None)
    plot_widget.addItem(plot)
    
    # Get the ViewBox, which controls zoom/pan
    vb = plot.getViewBox()
    
    # Disable the right-click context menu (which blocks default zoom)
    vb.setMenuEnabled(False)

    # --- 3. Create the toolbar ---
    toolbar_layout = QtWidgets.QHBoxLayout()
    
    # --- 4. Create buttons ---
    btn_pan = QtWidgets.QPushButton("Pan")
    btn_zoom = QtWidgets.QPushButton("Zoom")
    btn_reset = QtWidgets.QPushButton("Reset View")
    btn_save = QtWidgets.QPushButton("Save Image")
    
    # Make mode buttons checkable (to show active mode)
    btn_pan.setCheckable(True)
    btn_zoom.setCheckable(True)
    
    # Group them so only one can be checked at a time
    mode_group = QtWidgets.QButtonGroup(win)
    mode_group.addButton(btn_pan)
    mode_group.addButton(btn_zoom)
    mode_group.setExclusive(True)
    
    # Set default mode to Pan
    vb.setMouseMode(pg.ViewBox.PanMode)
    btn_pan.setChecked(True)

    # Add buttons to toolbar
    toolbar_layout.addWidget(btn_pan)
    toolbar_layout.addWidget(btn_zoom)
    toolbar_layout.addWidget(btn_reset)
    toolbar_layout.addStretch(1) # Spacer (pushes save to the right)
    toolbar_layout.addWidget(btn_save)

    # --- 5. Connect buttons to actions ---
    
    # Mode buttons change the behavior of the *left* mouse button
    btn_pan.clicked.connect(lambda: vb.setMouseMode(pg.ViewBox.PanMode))
    btn_zoom.clicked.connect(lambda: vb.setMouseMode(pg.ViewBox.RectMode))
    
    # Reset button
    btn_reset.clicked.connect(lambda: vb.autoRange())

    # Save action
    def do_save_image():
        base, _ = QtWidgets.QFileDialog.getSaveFileName(win, "Save image (basename)", "", "PNG (*.png)")
        if not base:
            return
        base = os.path.splitext(base)[0]
        try:
            from pyqtgraph.exporters import ImageExporter, SVGExporter
            # Export the plot_widget's scene
            exp = ImageExporter(plot_widget.scene())
            exp.parameters()['width'] = 2400
            exp.export(base + ".png")
            try:
                SVGExporter(plot_widget.scene()).export(base + ".svg")
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Export failed", str(e))
            
    btn_save.clicked.connect(do_save_image)

    # --- 5b. Add coordinate label ---
    lbl_coords = QtWidgets.QLabel("Clicked: None")
    lbl_coords.setStyleSheet("padding-left: 10px; font-weight: bold;")
    toolbar_layout.addWidget(lbl_coords)

    # State for distance measurement
    click_state = {"last_point": None}

    # --- 5c. Handle plot clicks ---
    def on_plot_clicked(event):
        # Only handle left clicks (button 1)
        if event.button() == QtCore.Qt.LeftButton:
            # Map scene coordinates to view coordinates
            pos = event.scenePos()
            if plot.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                x = int(round(mouse_point.x()))
                y = mouse_point.y()
                
                last = click_state["last_point"]
                if last is None:
                    click_state["last_point"] = (x, y)
                    lbl_coords.setText(f"Pt1: ({x}, {y:0.4f})")
                else:
                    x0, y0 = last
                    dx = x - x0
                    dy = y - y0
                    lbl_coords.setText(f"Pt2: ({x}, {y:0.4f}) | dx: {dx}, dy: {dy:0.4f}")
                    click_state["last_point"] = None # Reset for next pair

    # Connect the scene's click signal
    # Note: we use scene().sigMouseClicked because the plot widget itself might consume events
    plot_widget.scene().sigMouseClicked.connect(on_plot_clicked)

    # --- 6. Add shortcuts (from your old function) ---
    # Ctrl+S → save
    QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), win).activated.connect(do_save_image)
    # A → autorange
    QtGui.QShortcut(QtGui.QKeySequence("A"), win).activated.connect(lambda: vb.autoRange())
    # R → autorange
    QtGui.QShortcut(QtGui.QKeySequence("R"), win).activated.connect(lambda: vb.autoRange())

    # --- 7. Add layouts to the main window ---
    main_layout.addLayout(toolbar_layout) # Add toolbar
    main_layout.addWidget(plot_widget)    # Add plot widget

    # --- 8. Show the window and run ---
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
