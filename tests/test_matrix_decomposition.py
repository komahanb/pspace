#=====================================================================#
# Randomized matrix decomposition tests in randomly chosen
# N-dimensional coordinate system
#---------------------------------------------------------------------#
# Combinations:
#---------------------------------------------------------------------#
#     - numerical vs symbolic (to be implemented)
#     - sparse vs full decomposition
#---------------------------------------------------------------------#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

# python module imports
import pytest
import random

# core module imports
from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType,
                         PolyFunction)

# local module imports
from .test_utils import (random_coordinate,
                         random_polynomial,
                         get_coordinate_system_type)

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

#=====================================================================#
# Tests 2 B: Total Degree Basis (sparse vs full assembly)
#=====================================================================#

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
