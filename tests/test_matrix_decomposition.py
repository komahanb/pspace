import numpy as np
from collections import Counter

def test_decompose_matrix_basic(space):
    """
    space : your stochastic space object, with:
        - .basis   {i: Counter}
        - .getNumBasisFunctions()
        - .evaluateBasisDegreesY()
        - .build_quadrature()
        - .decompose_matrix()
    """

    # Polynomial: f(y) = 3 + 3*y0 + 3*y0^2*y1
    from yourmodule import PolyFunction  # wherever you put the class

    f_eval = PolyFunction([
        (3, Counter({})),             # constant
        (3, Counter({0:1})),          # linear
        (3, Counter({0:2, 1:1}))      # mixed quadratic
    ])

    # Dense (full) assembly
    A_full = space.decompose_matrix(f_eval, sparse=False, symmetric=True)

    # Sparse (polynomial_sparsity_mask) assembly
    A_sparse = space.decompose_matrix(f_eval, sparse=True, symmetric=True)

    # Compare
    diff = np.max(np.abs(A_full - A_sparse))
    print("max diff =", diff)
    print("full matrix:\n", A_full)
    print("sparse matrix:\n", A_sparse)

    # Assert equality within tolerance
    assert np.allclose(A_full, A_sparse, atol=1e-10)
