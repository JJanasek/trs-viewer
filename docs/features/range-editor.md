# Range Editor

**Crop → Range Editor…**

Mark sample regions and write them out as a new `.trs`.

- **Enable drag-select on plot**, then drag to add ranges — or click **Add current view** to capture whatever the plot is currently showing
- **Repeat selected range** — takes the selected range as a base window and stamps out `count − 1` further copies, `period` samples apart. Built for marking every occurrence of a repeating operation (AES rounds, say) after hand-picking one
- Ranges are **concatenated per trace** on export by default. Tick **Export ranges separately** to emit one output trace per range instead — this requires every range to be the same length
- The dialog keeps a running total of selected samples and ranges
