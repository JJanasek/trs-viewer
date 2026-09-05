# File Formats

## TRS (Riscure Trace Set)


Standard Riscure TRS v1 format. The file begins with a sequence of TLV header tags followed by the trace data. Each trace contains optional auxiliary data bytes followed by the sample block.

Supported sample types:

| Type tag | C type | Bytes/sample |
|---|---|---|
| `0x01` | `int8_t` | 1 |
| `0x02` | `int16_t` | 2 |
| `0x04` | `int32_t` | 4 |
| `0x14` | `float32` | 4 |

Traces are read on demand — the full file is never loaded into RAM.

**Auxiliary data bytes** (per trace, optional): arbitrary bytes attached to each trace — typically plaintext, ciphertext, key material, or any other per-trace metadata. Their presence is not required; trs-viewer works fine without them. When present, they are used by the t-test for group assignment and fed as the `data` matrix to the CPA leakage model. If the header parameter map contains a `ttest` key, its `offset` field selects the group byte for the t-test automatically.

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
