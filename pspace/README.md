# PSPACE Core Modules

The `pspace` package implements the contract that all mirrors (numeric, analytic,
symbolic, profiling, parallel, …) share. Key modules:

- `core.py` — shared enums (`CoordinateType`, `DistributionType`, `BasisFunctionType`,
  `InnerProductMode`) plus the `PolyFunction`/`OrthoPolyFunction` data containers.
- `interface.py` — abstract base classes for coordinate systems and mixins that define
  the decomposition/reconstruction API.
- `numeric.py` — reference implementation that evaluates integrals numerically and
  owns the coordinate hierarchy (`CoordinateFactory`, `NumericCoordinateSystem`,
  `StateEquation`, etc.).
- `symbolic.py` / `analytic.py` — mirrors that perform the same operations using
  SymPy expressions or closed-form rules.
- `sparsity.py`, `parallel.py`, `profile.py`, `validate.py`, `verify.py` — decorators
  adding cross-cutting concerns (sparsity gating, policy-based parallelism, timing,
  validation metadata, verification helpers).
- `orthonormal.py` / `plotter.py` — alternative views of the coordinate system (orthonormal
  basis, plotting utilities).
- `orthogonal_polynomials.py`, `stochastic_utils.py` — shared math utilities used by
  multiple mirrors.

The `PSPACE.pyx`/`PSPACE.pxd` pair hosts the legacy Cython bindings; they depend on the
same modules listed above.

> When adding a new mirror or decorator, implement the `CoordinateSystem` interface,
> keep method signatures identical, and update this file so future contributors can
> discover the entry point.
