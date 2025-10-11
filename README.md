![alt text](https://github.com/komahanb/pspace/blob/master/sgm.png)

# Probabilistic Space

Module for uncertainty quantification and optimization under uncertainty applications

## Modes
1. Stochastic Projection (Galerkin)
2. Stochastic Collocation

## Basis Constrution
1. Tensor product space
2. Complete polynomial space

## Distributions
1. Gaussian/Normal
2. Exponential
3. Uniform

## Quadratures
1. Gauss--Hermite
2. Gauss--Laguerre
3. Gauss--Legendre

## Orthogonal and Orthonormal Polynomials
1. Hermite
2. Laguerre
3. Legendre

---

Komahan Boopathy

## Development Environment

Linux/macOS:

```bash
scripts/setup_env.sh
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

Override the interpreter with `PYTHON_BIN=/path/to/python scripts/setup_env.sh` or `-PythonBin C:\Path\To\python.exe` when invoking the Windows script.

Both scripts install `requirements.txt`. GPU acceleration relies on `cupy` (install the CUDA-matched wheel manually if you need the parallel CuPy backend).

Core Python dependencies:
- numpy
- scipy
- sympy
- matplotlib
- mpi4py

Optional additions:
- pytest (to run the test suite)
- cupy-cudaXX (enable `CudaCupyParallelPolicy` on NVIDIA GPUs)
- stacs/tacs (needed for the aeroelastic examples; install from their respective projects)

OpenMP-style shared-memory policies run on Python threads; no extra Python package is required, but building the C++ extension with OpenMP support and configuring `mpicxx` with `-fopenmp` may improve performance.
