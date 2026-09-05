# Export (TRS / NPY / NPZ)

**Export → Export TRS… / Export traces as NPY… / Export traces as NPZ…**

Each writes out the currently-configured pipeline applied to a chosen trace range:

- **Export TRS** preserves the source file's original sample coding — an `int16` file re-exported after processing stays `int16` (rounded and clamped to range, rather than silently forced to `float32` and doubling in size). NPY/NPZ are always `float32`, per the format.
- **Apply last alignment shifts** — shown whenever the active dataset has a stored alignment (from [Trace Alignment](alignment.md) or a Chain's Align step), ticked by default. Each trace is read with its stored shift applied and zero-padded at the boundary it introduces; traces marked discarded (below the alignment's correlation threshold) are dropped from the output entirely rather than exported unshifted.
