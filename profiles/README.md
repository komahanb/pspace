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

## Parallel examples

The profilers understand the parallel coordination layer added to `pspace` and can exercise MPI and shared-memory policies directly from the CLI. A few sample invocations:

- Launch a vector profile with MPI ranks distributing the basis and threads handling shared work on each rank:

  ```bash
  mpiexec -n 4 python profiles/profile_single_mode.py \
      --mode numerical \
      --rank vector \
      --basis tensor \
      --parallel true \
      --shared threads \
      --shared-workers 8 \
      --trials 5
  ```

- Profile matrix assembly while mirroring the result to a CUDA device via CuPy:

  ```bash
  python profiles/profile_single_mode.py \
      --mode numerical \
      --rank matrix \
      --parallel false \
      --shared cupy \
      --shared-device cuda:0 \
      --symmetric true \
      --trials 3
  ```

- Run a matrix-mode benchmark with MPI ranks and OpenMP-style shared threading:

  ```bash
  mpiexec -n 2 python profiles/profile_single_mode.py \
      --mode numerical \
      --rank matrix \
      --basis total \
      --parallel true \
      --shared openmp \
      --shared-workers 4 \
      --max-degree 2 \
      --trials 3
  ```

Adjust the MPI launch parameters, shared workers, or CUDA device identifiers as appropriate for your hardware. Each run reports the active policy composition so you can confirm the configuration that was exercised.
