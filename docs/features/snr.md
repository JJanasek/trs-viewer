# Signal-to-Noise Ratio

**SCA → SNR…**

Per-sample SNR — `SNR[s] = Var(E[T[s] | class]) / E[Var(T[s] | class)]` — computed with an online accumulator, so trace count is not bounded by RAM.

Class labels are derived from one auxiliary data byte:

| Class mode | Classes |
|---|---|
| Raw byte value | 256 (0–255) |
| Hamming weight | 9 (0–8) |
| AES S-box output | 256 |
| AES S-box output — Hamming weight | 9 |

The two S-box modes take a **key byte hypothesis** and label each trace by `S-box(data_byte ⊕ key)`, so a correct hypothesis produces a sharper SNR peak. Requires a file with per-trace data bytes.

Accumulator memory is `2 × classes × samples × 8` bytes; the dialog estimates it up front and asks before allocating anything large.
