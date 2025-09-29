#=====================================================================#
# Stress tests: vector decomposition in high-dim / high-degree setups
#---------------------------------------------------------------------#
# Goals:
#   - probe quadrature sufficiency (.max_degrees logic)
#   - probe sparsity detection (.degrees masks)
#   - catch conditioning errors in Normal/Exponential distributions
#   - collect timing ratios for sparse vs full assembly
#---------------------------------------------------------------------#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
import random
import numpy as np
from timeit import default_timer as timer

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

MAX_DEG               = 5       # push polynomial/basis degree
MAX_COORD             = 5       # up to 5D coordinates
TRIALS                = 3       # fewer trials (stress is heavier)
TOL                   = 1e-6

# Timing collectors
tensor_timings = []
total_timings  = []

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

    polynomial_function = random_polynomial(cs, max_deg=MAX_DEG, max_cross_terms=MAX_DEG)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(polynomial_function,
                                                             tol=TOL,
                                                             verbose=False)
    assert ok

    # Timing comparison
    start_sparse = timer(); cs.decompose(polynomial_function, sparse=True);  elapsed_sparse = timer() - start_sparse
    start_full   = timer(); cs.decompose(polynomial_function, sparse=False); elapsed_full   = timer() - start_full
    tensor_timings.append(elapsed_full / elapsed_sparse)


@pytest.mark.parametrize("trial", range(TRIALS))
def test_stress_total_sparse_full(trial):
    random.seed(4000 + trial)
    print(f"\n=== STRESS Trial {trial} : Total Basis (sparse vs full) ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg=MAX_DEG,
                                    max_coords=MAX_COORD)

    polynomial_function = random_polynomial(cs, max_deg=MAX_DEG, max_cross_terms=MAX_DEG)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(polynomial_function,
                                                             tol=TOL,
                                                             verbose=False)
    assert ok

    # Timing comparison
    start_sparse = timer(); cs.decompose(polynomial_function, sparse=True);  elapsed_sparse = timer() - start_sparse
    start_full   = timer(); cs.decompose(polynomial_function, sparse=False); elapsed_full   = timer() - start_full
    total_timings.append(elapsed_full / elapsed_sparse)


#=====================================================================#
# Timing summaries
#=====================================================================#

@pytest.mark.parametrize("finalize", [True])
def test_tensor_timing_summary(finalize):
    if not tensor_timings:
        pytest.skip("No tensor timings recorded")

    ratios = np.array(tensor_timings)
    print("\n=== Timing Summary: Tensor Basis Vector Decomposition ===")
    print(f"Trials      : {len(ratios)}")
    print(f"Mean ratio  : {ratios.mean():.3f}")
    print(f"Std. dev    : {ratios.std():.3f}")
    print(f"Min ratio   : {ratios.min():.3f}")
    print(f"Max ratio   : {ratios.max():.3f}")


@pytest.mark.parametrize("finalize", [True])
def test_total_timing_summary(finalize):
    if not total_timings:
        pytest.skip("No total timings recorded")

    ratios = np.array(total_timings)
    print("\n=== Timing Summary: Total Basis Vector Decomposition ===")
    print(f"Trials      : {len(ratios)}")
    print(f"Mean ratio  : {ratios.mean():.3f}")
    print(f"Std. dev    : {ratios.std():.3f}")
    print(f"Min ratio   : {ratios.min():.3f}")
    print(f"Max ratio   : {ratios.max():.3f}")
