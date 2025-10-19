# Demos

Small, executable scripts that illustrate typical workflows:

- `2d_orthonormal_reconstruction.py` — builds a two-axis coordinate system, decomposes
  a polynomial, and visualises the reconstructed function on a grid.
- `demo_matrix_decomposition.py` — assembles tensor-product quadrature for two axes and
  prints the resulting matrix coefficients.
- `demo_state_equation.py` — shows how to assemble and solve the state equation wrapper
  (`StateEquation`) for a simple PDE-like operator and right-hand side.
- `test_basis_transform.py` — round-trip sanity check for the basis transform
  (monomials ↔ orthonormal); marked as a manual demo, not part of CI.

Run any script with `python demos/<name>.py`. The demos assume local execution
without MPI; they only depend on the Python package itself.
