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

## Build & Test

All builds are surfaced through the top-level `Makefile` and the helper script `build.sh`.

- `make` – release build (real scalars) of the native library and staged artifacts in `lib/`
- `make debug` – debug build (adds `-O0 -g`)
- `make complex` – release build that enables complex scalars by defining `USE_COMPLEX`
- `make complex_debug` – debug build with complex scalars
- `make interface` – rebuild the Cython extension against the current native artifacts
- `make complex_interface` – rebuild the Cython extension in complex mode (`PSPACE_COMPLEX=1`)

After a build you can execute the Python test suite:

```bash
# Real-valued build
make && make interface
python -m pytest -q

# Complex-valued build
make complex && make complex_interface
python -m pytest -q
```

The sparsity plotting tests are skipped automatically when Matplotlib
cannot be imported; the remaining tests run to completion in both modes.

---

Komahan Boopathy
