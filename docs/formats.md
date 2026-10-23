# File Formats

## TRS (Riscure Trace Set)


Riscure TRS, versions 1 and 2. The file begins with a sequence of TLV header tags followed by the trace data. Each trace contains an optional title, optional auxiliary data bytes, then the sample block.

Supported sample types:

| Type tag | C type | Bytes/sample |
|---|---|---|
| `0x01` | `int8_t` | 1 |
| `0x02` | `int16_t` | 2 |
| `0x04` | `int32_t` | 4 |
| `0x14` | `float32` | 4 |

Traces are read on demand — the full file is never loaded into RAM.

**Auxiliary data bytes** (per trace, optional): arbitrary bytes attached to each trace — typically plaintext, ciphertext, key material, or any other per-trace metadata. Their presence is not required; trs-viewer works fine without them. When present, they are used by the t-test for group assignment and fed as the `data` matrix to the CPA leakage model. If the header parameter map contains a `ttest` key, its `offset` field selects the group byte for the t-test automatically.

## TRS v2 metadata

TRS v2 files (e.g. written by Riscure's `trsfile` with `TRS_VERSION: 2`) can
describe themselves with two parameter blocks. Both are read:

- **Trace set parameters** (`TRACE_SET_PARAMETERS`, tag `0x76`): metadata of
  the whole set — scope, sample rate, gain, keys, firmware, the capture
  command. **File → File Details…** (or **Details…** in the side panel) lists
  them with their types, next to the description, scales and the per-trace
  fields; **Copy as Text** puts it all on the clipboard.
- **Per-trace fields** (`TRACE_PARAMETER_DEFINITIONS`, tag `0x77`): named
  fields in each trace's data bytes, each with a type and a number of
  elements (a `double` is one element of 8 bytes). The *Trace Data* panel
  shows them by name and type: numbers as numbers, byte arrays (messages,
  signatures, keys) in hex. A field named `ttest` selects the t-test group
  byte.

Types: `byte` (`0x01`), `short` (`0x02`), `int` (`0x04`), `long` (`0x08`),
`float` (`0x14`), `double` (`0x18`), `string` (`0x20`, UTF-8), `bool`
(`0x31`).

**Export TRS** keeps the description and both parameter blocks (the data
bytes are copied unchanged, so the field definitions still fit), the time
scale when the pipeline keeps sample positions, and the value scale when
there is no pipeline.

A file whose capture stopped early opens with its complete traces (and a
note), even if its header announces more, or none.

## LEGACY_DATA (optional)

Riscure's older trace sets may store cryptographic I/O in a fixed 32-byte layout, advertised via a TRS parameter map entry with key `LEGACY_DATA`, `offset = 0`, `length = 32`. This field is entirely optional — trs-viewer auto-detects it and changes the data display, but will load and analyse any TRS file regardless of whether it is present.

```
bytes  0–15   plaintext  (PT)
bytes 16–31   ciphertext (CT)
```

When trs-viewer detects this parameter, the data inspector in the side panel displays the bytes with `PT:` and `CT:` labels instead of a raw hex dump. The bytes are still passed as-is to the CPA leakage model — `data[:, 0:16]` is the plaintext, `data[:, 16:32]` is the ciphertext.

Generating a compatible TRS file in Python (e.g. with the `trsfile` library):
```python
import trsfile, numpy as np

traces = np.random.randn(1000, 5000).astype(np.float32)
keys   = np.random.randint(0, 256, (1000, 16), dtype=np.uint8)

with trsfile.open("out.trs", "w", trs_version=1,
                  num_samples=5000, sample_coding=trsfile.SampleCoding.FLOAT) as ts:
    for t, k in zip(traces, keys):
        ts.append(trsfile.Trace(trsfile.SampleCoding.FLOAT, t, data=bytes(k)))
```


## NPY


A 2-D NumPy array file:

- **dtype**: `float32` (`<f4`, little-endian)
- **shape**: `(n_traces, n_samples)`

```python
import numpy as np
traces = np.random.randn(1000, 5000).astype(np.float32)
np.save("traces.npy", traces)
```


## NPZ


A NumPy ZIP archive with one or two arrays:

| Array key | dtype | shape | Required |
|---|---|---|---|
| `traces` | `float32` | `(n_traces, n_samples)` | Yes |
| `data` | `uint8` | `(n_traces, data_length)` | No — needed for CPA and t-test |

The archive must use **STORE** compression (no deflate):
```python
import numpy as np
traces = np.random.randn(1000, 5000).astype(np.float32)
data   = np.random.randint(0, 256, (1000, 16), dtype=np.uint8)
np.savez("traces.npz", traces=traces, data=data)
# np.savez uses STORE by default — do NOT use np.savez_compressed
```

The `data` array feeds the CPA leakage model and the t-test group assignment, equivalent to the auxiliary data bytes in a TRS file.

---
