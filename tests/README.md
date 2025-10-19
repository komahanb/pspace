# Test Suite Overview

The automated tests exercise each mirror and decorator through focused unit tests:

- `test_numeric_coordinate_system.py` — numerical inner products, sparsity toggles,
  and mirror delegation checks for `NumericCoordinateSystem`.
- `test_symbolic_coordinate_system.py` / `test_analytic_coordinate_system.py` — mirror
  parity between symbolic/analytic projections and the numeric baseline.
- `test_parallel_coordinate_system.py` — policy composition, delegation order, and
  optional backends (`SharedMemoryParallelPolicy`, `MPIParallelPolicy`, etc.).
- `test_sparsity_coordinate_system.py` — runtime sparsity overrides and mask
  correctness.
- `test_orthonormal_coordinate_system.py` — orthonormal mirror round-trip coverage.

Shared fixtures live under `tests/utils/`. All tests rely on `pytest`; run them via
`pytest -q` from the repository root.
