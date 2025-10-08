# Validations

Stress and robustness evaluations for `pspace`. These scripts run heavier
randomised trials via `pspace.validate.CoordinateSystem` to expose conditioning
issues, sparsity handling problems, and overall performance envelopes.

Example usage:

```bash
pytest validations/validate_vector_decomposition.py::test_stress_tensor_sparse_full -q
```

## Scripts

- `validate_vector_decomposition.py` – Sparse vs full vector decompositions across
  random high-dimensional systems, reporting timing ratios.
- `validate_matrix_decomposition.py` – Analogous matrix assembly stress tests.

Both scripts print timing summaries and assert that sparse and dense assemblies
agree within tolerance.
