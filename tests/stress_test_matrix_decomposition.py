#=====================================================================#
# Stress tests for randomized matrix decomposition in randomly chosen
# N-dimensional coordinate systems
#---------------------------------------------------------------------#
# Focus:
#     - Sparse vs Full assembly of matrix decomposition
#     - Timing summaries across multiple trials
#---------------------------------------------------------------------#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
import random
import numpy as np

# core imports
from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType,
                         PolyFunction)

# local imports
from .test_utils import (random_coordinate,
                         random_polynomial,
                         get_coordinate_system_type)


#=====================================================================#
# Helper: store timing ratios for summary
#=====================================================================#
tensor_timings = []
total_timings  = []


#=====================================================================#
# Stress Test A: Tensor Degree Basis
#=====================================================================#

@pytest.mark.parametrize("trial", range(20))
def test_stress_tensor_matrix_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Stress Trial {trial} : Matrix TENSOR Basis (sparse vs full)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                    max_deg=4,
                                    max_coords=4)

    polynomial_function = random_polynomial(cs, max_deg=3, max_terms=5)

    ok, diffs = cs.check_decomposition_matrix_sparse_full(polynomial_function,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

    # record timing ratio if available
    from timeit import default_timer as timer
    start_sparse = timer(); cs.decompose_matrix(polynomial_function, sparse=True); elapsed_sparse = timer() - start_sparse
    start_full   = timer(); cs.decompose_matrix(polynomial_function, sparse=False); elapsed_full   = timer() - start_full
    tensor_timings.append(elapsed_full / elapsed_sparse)


@pytest.mark.parametrize("finalize", [True])
def test_tensor_timing_summary(finalize):
    if not tensor_timings:
        pytest.skip("No tensor timings recorded")

    ratios = np.array(tensor_timings)
    print("\n=== Timing Summary: Tensor Basis Matrix Decomposition ===")
    print(f"Trials      : {len(ratios)}")
    print(f"Mean ratio  : {ratios.mean():.3f}")
    print(f"Std. dev    : {ratios.std():.3f}")
    print(f"Min ratio   : {ratios.min():.3f}")
    print(f"Max ratio   : {ratios.max():.3f}")


#=====================================================================#
# Stress Test B: Total Degree Basis
#=====================================================================#

@pytest.mark.parametrize("trial", range(20))
def test_stress_total_matrix_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Stress Trial {trial} : Matrix TOTAL Basis (sparse vs full)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                    max_deg=4,
                                    max_coords=4)

    polynomial_function = random_polynomial(cs, max_deg=3, max_terms=5)

    ok, diffs = cs.check_decomposition_matrix_sparse_full(polynomial_function,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

    # record timing ratio if available
    from timeit import default_timer as timer
    start_sparse = timer(); cs.decompose_matrix(polynomial_function, sparse=True); elapsed_sparse = timer() - start_sparse
    start_full   = timer(); cs.decompose_matrix(polynomial_function, sparse=False); elapsed_full   = timer() - start_full
    total_timings.append(elapsed_full / elapsed_sparse)


@pytest.mark.parametrize("finalize", [True])
def test_total_timing_summary(finalize):
    if not total_timings:
        pytest.skip("No total timings recorded")

    ratios = np.array(total_timings)
    print("\n=== Timing Summary: Total Basis Matrix Decomposition ===")
    print(f"Trials      : {len(ratios)}")
    print(f"Mean ratio  : {ratios.mean():.3f}")
    print(f"Std. dev    : {ratios.std():.3f}")
    print(f"Min ratio   : {ratios.min():.3f}")
    print(f"Max ratio   : {ratios.max():.3f}")
