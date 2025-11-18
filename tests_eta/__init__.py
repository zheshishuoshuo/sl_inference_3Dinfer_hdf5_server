from __future__ import annotations

"""
Injection–recovery tests for (alpha_sps, mu_h, mu_gamma).

This subpackage provides a small pipeline:

1. Generate or load 1000-lens mock samples for a given TestMode.
2. Tabulate 2D dark-matter kernels on the external DM grid.
3. Compute 1D / 2D likelihood slices over eta = (alpha_sps, mu_h, mu_gamma).
4. Plot the resulting slices for quick visual inspection.
"""

