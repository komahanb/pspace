# Verifications

Mathematical and functional consistency checks for `pspace`.  These scripts use
`pspace.verify.CoordinateSystem` and focus on validating that decompositions,
reconstructions, and change-of-basis operations behave as expected.

Run an individual verification with:

```bash
pytest verifications/verify_change_of_basis_identity.py -q
```

## Available verifications

- `verify_change_of_basis_identity.py` – Checks the α↔ψ change-of-basis consistency.
- `verify_commutativity_multiply_x.py` – Ensures multiplication operators commute across bases.
- `verify_vector_decomposition.py` / `verify_matrix_decomposition.py` – Randomised decompositions comparing numerical vs symbolic vs analytic modes.
- `verify_reconstruction.py` / `verify_reconstruction_mixed.py` – Reconstruction consistency in 1D and mixed distributions.
- `verify_adjoint_function_gradient.py` – Dual reconstruction for gradients.
- Utilities live in `verify_utils.py` for randomized coordinate systems and polynomials.
