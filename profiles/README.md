# Profiles

Benchmark utilities for the `pspace` decomposition operators. These scripts do **not**
check numerical correctness; they are meant to characterise runtime performance as the
number of variables and the basis size grow.

## `profile_vector_decomposition.py`

Profiles the rank-1 (`CoordinateSystem.decompose`) operator across numerical, symbolic,
and analytic inner-product modes.

```bash
python3 profiles/profile_vector_decomposition.py \
    --trials 3 \
    --num-coords 4 \
    --max-degree 2 \
    --output-dir profiles/output/vector
```

## `profile_matrix_decomposition.py`

Profiles the rank-2 (`CoordinateSystem.decompose_matrix`) operator across the same set
of modes, optionally with dense or symmetric assembly toggles.

```bash
python3 profiles/profile_matrix_decomposition.py \
    --trials 3 \
    --num-coords 4 \
    --max-degree 2 \
    --output-dir profiles/output/matrix
```

## Shared options

| Flag | Meaning |
| --- | --- |
| `--trials` | Number of repetitions per combination (default `3`). |
| `--num-coords` | How many coordinates/variables to include. |
| `--max-degree` | Maximum monomial degree per coordinate. |
| `--seed` | Reproducibility seed (default `2025`). |

## `profile_single_mode.py`

Profiles a single inner-product mode (vector or matrix) for a chosen configuration:

```bash
python3 profiles/profile_single_mode.py \
    --mode analytic \
    --rank vector \
    --basis tensor \
    --num-coords 4 \
    --max-degree 3 \
    --trials 10 \
    --sparse true \
    --symmetric true \
    --seed 2025
```

This script prints the random instance (coordinates, basis size, polynomial terms) along with individual trial timings and aggregate statistics (best, worst, mean). Re-running with the same `--seed` lets you benchmark multiple modes against the exact same problem instance for apples-to-apples comparisons.
