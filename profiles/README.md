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
