#=====================================================================#
# Stress tests: vector decomposition in high-dim / high-degree setups
#---------------------------------------------------------------------#
# Goals:
#   - probe quadrature sufficiency (.max_degrees logic)
#   - probe sparsity detection (.degrees masks)
#   - catch conditioning errors in Normal/Exponential distributions
#---------------------------------------------------------------------#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
import random

from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType,
                         PolyFunction)

from .test_utils import (random_coordinate,
                         random_polynomial,
                         get_coordinate_system_type)

from .test_utils import ENABLE_ANALYTIC_TESTS

#=====================================================================#
# Parameters for stress regime
#=====================================================================#

MAX_DEG               = 8       # push polynomial/basis degree
MAX_COORD             = 5       # up to 5D coordinates
TRIALS                = 3       # fewer trials (stress is heavier)
TOL                   = 1e-6

#=====================================================================#
# Sparse vs full stress tests
#=====================================================================#

@pytest.mark.parametrize("trial", range(TRIALS))
def test_stress_tensor_sparse_full(trial):
    random.seed(3000 + trial)
    print(f"\n=== STRESS Trial {trial} : Tensor Basis (sparse vs full) ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                    max_deg=MAX_DEG,
                                    max_coords=MAX_COORD)

    polynomial_function = random_polynomial(cs, max_deg=MAX_DEG)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(polynomial_function,
                                                             tol=TOL,
                                                             verbose=False)
    assert ok


@pytest.mark.parametrize("trial", range(TRIALS))
def test_stress_total_sparse_full(trial):
    random.seed(4000 + trial)
    print(f"\n=== STRESS Trial {trial} : Total Basis (sparse vs full) ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg=MAX_DEG,
                                    max_coords=MAX_COORD)

    polynomial_function = random_polynomial(cs, max_deg=MAX_DEG)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(polynomial_function,
                                                             tol=TOL,
                                                             verbose=False)
    assert ok

#=====================================================================#
# Numerical vs symbolic stress tests
#=====================================================================#
@pytest.mark.skipif(not ENABLE_ANALYTIC_TESTS, reason="Analytic vs numerical disabled")
@pytest.mark.parametrize("trial", range(TRIALS))
def test_stress_tensor_numerical_symbolic(trial):
    random.seed(1000 + trial)  # offset to avoid overlap with regular tests
    print(f"\n=== STRESS Trial {trial} : Tensor Basis (deg≤{MAX_DEG}, dim≤{MAX_COORD}) ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                    max_deg=MAX_DEG,
                                    max_coords=MAX_COORD)

    polynomial_function = random_polynomial(cs, max_deg=MAX_DEG)

    ok, diffs = cs.check_decomposition_numerical_symbolic(polynomial_function,
                                                          sparse=True,
                                                          tol=TOL,
                                                          verbose=False)
    assert ok

@pytest.mark.skipif(not ENABLE_ANALYTIC_TESTS, reason="Analytic vs numerical disabled")
@pytest.mark.parametrize("trial", range(TRIALS))
def test_stress_total_numerical_symbolic(trial):
    random.seed(2000 + trial)
    print(f"\n=== STRESS Trial {trial} : Total Basis (deg≤{MAX_DEG}, dim≤{MAX_COORD}) ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg=MAX_DEG,
                                    max_coords=MAX_COORD)

    polynomial_function = random_polynomial(cs, max_deg=MAX_DEG)

    ok, diffs = cs.check_decomposition_numerical_symbolic(polynomial_function,
                                                          sparse=True,
                                                          tol=TOL,
                                                          verbose=False)
    assert ok
