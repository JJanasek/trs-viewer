# Cross-Correlation Matrix

**SCA → Cross-Correlation…**

Computes the M×M normalised Pearson correlation matrix `C[i,j] = Corr(sᵢ, sⱼ)`, or a rectangular search×ref matrix for template matching.

| Method | Description |
|---|---|
| **Baseline** | Streaming rank-1 outer-product updates |
| **Dual Matrix** | Gram `G = AᵀA / M` → eigen-reconstruction |
| **MP-Cleaned** | Dual Matrix + zeroes eigenvalues ≤ λ₊ (Marchenko-Pastur noise edge) |
| **Two-Window** | Rectangular search×ref cross-correlation for template matching |

A **stride** parameter subsamples before computing (`M = ⌈samples / stride⌉`) to reduce memory and computation time.

![Cross-correlation heatmap](../heatmap.gif)
