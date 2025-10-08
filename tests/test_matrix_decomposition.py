#=====================================================================#
# Randomized matrix decomposition tests in randomly chosen
# N-dimensional coordinate system
#---------------------------------------------------------------------#
# Combinations:
#---------------------------------------------------------------------#
#     - numerical vs symbolic
#     - sparse vs full decomposition
#---------------------------------------------------------------------#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

# python module imports
import pytest
import random

# core module imports
import numpy as np

from pspace.core import BasisFunctionType, InnerProductMode

# local module imports
from .test_utils import (
    random_polynomial,
    get_coordinate_system_type,
)

#=====================================================================#
# Sparsity-aware and sparsity-unware matrix decomposition tests
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_tensor_basis_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Matrix TENSOR Basis (sparse vs full assembly)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                    max_deg = 3, max_coords = 3)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_matrix_sparse_full(polynomial_function,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

@pytest.mark.parametrize("trial", range(5))
def test_randomized_total_basis_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Matrix TOTAL Basis (sparse vs full assembly)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg = 3, max_coords = 3)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_matrix_sparse_full(polynomial_function,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

#=====================================================================#
# Sparse numerical versus sparse analytical matrix decomposition tests
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_tensor_matrix_numerical_symbolic(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Matrix TENSOR Basis (numerical vs symbolic)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                    max_deg=3, max_coords=3)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_matrix_numerical_symbolic(
                    polynomial_function,
                    tol=1e-6,
                    verbose=True)
    assert ok


@pytest.mark.parametrize("trial", range(5))
def test_randomized_total_matrix_numerical_symbolic(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Matrix TOTAL Basis (numerical vs symbolic)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg=3, max_coords=3)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_matrix_numerical_symbolic(
                    polynomial_function,
                    tol=1e-6,
                    verbose=True)
    assert ok

#=====================================================================#
# Dense (sparsity-unaware) numerical vs symbolic checks
#=====================================================================#

@pytest.mark.parametrize("basis_type", [
    BasisFunctionType.TENSOR_DEGREE,
    BasisFunctionType.TOTAL_DEGREE,
])
def test_dense_numerical_symbolic_matrix(basis_type):
    random.seed(6789)

    cs = get_coordinate_system_type(basis_type, max_deg=2, max_coords=2)
    polynomial_function = random_polynomial(cs, max_deg=2, max_cross_terms=1)

    A_num = cs.decompose_matrix(
        polynomial_function,
        sparse=False,
        symmetric=True,
        mode=InnerProductMode.NUMERICAL,
    )
    A_sym = cs.decompose_matrix(
        polynomial_function,
        sparse=False,
        symmetric=True,
        mode=InnerProductMode.SYMBOLIC,
    )

    assert np.allclose(A_num, A_sym, atol=1e-8), "dense matrix numeric vs analytic mismatch"
