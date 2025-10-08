#=====================================================================#
# Randomized vector decomposition tests in randomly chosen
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
from pspace.core import BasisFunctionType, InnerProductMode

# local module imports
from .verify_utils import (
    random_polynomial,
    get_coordinate_system_type,
)

import numpy as np

#=====================================================================#
# Symbolic and numerical vector decomposition tests
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_tensor_numerical_symbolic(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Tensor Degree Basis (numerical vs symbolic)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_symbolic(polynomial_function,
                                                          sparse = True,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

@pytest.mark.parametrize("trial", range(5))
def test_randomized_total_numerical_symbolic(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Total Degree Basis (numerical vs symbolic)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_symbolic(polynomial_function,
                                                          sparse = True,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

#=====================================================================#
# Sparsity-aware and sparsity-unware vector decomposition tests
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_tensor_basis_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Tensor Degree Basis (sparse vs full assembly)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                    max_deg = 3, max_coords = 3)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(polynomial_function,
                                                             tol=1e-6,
                                                             verbose=True)
    assert ok

#=====================================================================#
# Tests 2 B: Total Degree Basis (sparse vs full assembly)
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_total_basis_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Total Degree Basis (sparse vs full assembly)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg = 3, max_coords = 3)

    polynomial_function = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(polynomial_function,
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
def test_dense_numerical_symbolic_vector(basis_type):
    random.seed(12345)

    cs = get_coordinate_system_type(basis_type, max_deg=2, max_coords=2)
    polynomial_function = random_polynomial(cs, max_deg=2, max_cross_terms=1)

    coeffs_num = cs.decompose(
        polynomial_function,
        sparse=False,
        mode=InnerProductMode.NUMERICAL,
    )
    coeffs_sym = cs.decompose(
        polynomial_function,
        sparse=False,
        mode=InnerProductMode.SYMBOLIC,
    )

    for k in coeffs_num:
        assert np.isclose(
            coeffs_num[k],
            float(coeffs_sym[k]),
            atol=1e-8,
        ), f"dense numeric vs analytic mismatch at basis {k}"


@pytest.mark.parametrize("basis_type", [
    BasisFunctionType.TENSOR_DEGREE,
    BasisFunctionType.TOTAL_DEGREE,
])
def test_dense_analytic_vector(basis_type):
    random.seed(4242)

    cs = get_coordinate_system_type(basis_type, max_deg=2, max_coords=2)
    polynomial_function = random_polynomial(cs, max_deg=2, max_cross_terms=1)

    coeffs_num = cs.decompose(
        polynomial_function,
        sparse=False,
        mode=InnerProductMode.NUMERICAL,
    )
    coeffs_sym = cs.decompose(
        polynomial_function,
        sparse=False,
        mode=InnerProductMode.SYMBOLIC,
    )
    coeffs_ana = cs.decompose(
        polynomial_function,
        sparse=False,
        mode=InnerProductMode.ANALYTIC,
    )

    for k in coeffs_num:
        assert np.isclose(coeffs_num[k], coeffs_ana[k], atol=1e-10)
        assert np.isclose(coeffs_sym[k], coeffs_ana[k], atol=1e-10)
