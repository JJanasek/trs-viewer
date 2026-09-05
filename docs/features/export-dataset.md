# Export Dataset

**Export → Export Dataset (NPZ)…**

Writes an NPZ holding the processed trace matrix plus **any number of named label arrays derived from the auxiliary data bytes** — the machine-learning-friendly counterpart to plain NPZ export, which can only dump the raw data block.

Each label is defined by a byte offset, a byte count and an extraction type:

| Label type | Produces |
|---|---|
| **Raw value** | The bytes read as an integer (endianness selectable) |
| **Hamming weight** | Popcount over the selected bytes |
| **Bit** | A single bit (0 = LSB) as 0/1 |
| **Bit range** | An inclusive low–high bit slice, read as an integer |

dtype is chosen automatically from the width — 1 byte → `uint8`, 2 → `uint16`, 3–4 → `uint32`, 5–8 → `uint64`; Hamming weight and single-bit labels are always `uint8`. Each label becomes an array named after the label inside the archive, alongside `traces`. Label definitions are pre-populated from the file's TRS parameter map where one is present, and the dialog estimates the output matrix size before writing.
