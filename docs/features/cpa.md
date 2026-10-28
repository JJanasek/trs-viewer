# Correlation Power Analysis (CPA)

**Analyze → CPA…**

CPA correlates a user-defined leakage model with every sample column of the trace matrix and ranks the results across M hypotheses.

**How it runs.** The Python leakage model is evaluated first, for every
guess, in the foreground; this step is short. The correlation then runs in
the background like the other analyses, with progress and Cancel in the
jobs panel. It goes through the samples in blocks: each trace's block is
read once for all guesses, so the range can be as long as the traces, even
62.5M samples. The full trace matrix is never held in memory.

**Configuration:**
- Trace range and sample range. **Draw on plot →** switches the main plot to
  select mode: drag the region to analyse and it fills in *First sample* and
  *Count* (the dialog stays open meanwhile).
- **M (hypotheses)** — number of key guesses to test (1–65536); for AES key byte attacks use 256, for direct data correlation set M to the data field length
- **Alignment** — apply stored alignment shifts. Ticking it fills in the
  aligned traces and samples as a starting point; both ranges stay editable,
  so you can run on any part of the aligned set (each trace uses its own
  shift). Traces the alignment doesn't cover, or marks as discarded, are left
  out.

The result's sample numbers (hover, *Top candidates*) are raw sample
positions when the pipeline keeps them; with a resampling pipeline they are
indices of the pipeline's output from *First sample* on.

**Leakage model editor:**
- Python code editor with syntax highlighting and boilerplate (AES S-box + Hamming weight)
- **Test Model** — runs the model against the loaded trace data and shows the output; CPA can only be launched after a successful test
- **Load / Save** — import or export `.py` model files
- **Model library** — persistent library at `~/.local/share/trs-viewer/models/`; use the **Library** drop-down to load saved models, **Save to library…** to store new ones

Model function signature:
```python
def get_leakages(data: np.ndarray, hypothesis: int) -> np.ndarray:
    """
    data       — uint8 array, shape (n_traces, data_length)
    hypothesis — current key hypothesis (0 .. M-1)
    returns    — float32 array, shape (n_traces,)
    """
```

For direct data correlation (no key model), set M to the data field length and use `hypothesis` as a byte index:
```python
def get_leakages(data, hypothesis):
    return data[:, hypothesis].astype(np.float32)
```

For LEGACY_DATA files (PT+CT in 32 bytes), correlate against plaintext byte `b` by setting M = 16 and indexing into the first half:
```python
def get_leakages(data, hypothesis):
    # hypothesis = byte index 0..15 into plaintext
    return data[:, hypothesis].astype(np.float32)
```
Or against ciphertext (offset 16):
```python
def get_leakages(data, hypothesis):
    return data[:, 16 + hypothesis].astype(np.float32)
```

**Result window:**
- Heatmap of the M × N_samples correlation matrix, when every guess's curve
  fits in memory.
- On a long range they may not fit: 256 guesses × 62.5M samples would be
  64 GB. CPA then keeps the curves of the **10 strongest guesses** only and
  shows them as curves. A note says so, and the table still ranks every guess.
- **Top candidates** table — ranked by peak |r|, showing hypothesis value, hex (M ≤ 256), peak correlation, and sample index
