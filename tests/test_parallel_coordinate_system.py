from __future__ import annotations

from collections import Counter

import numpy as np

from pspace.core import (
    BasisFunctionType,
    CoordinateFactory,
    CoordinateSystem as NumericCoordinateSystem,
    InnerProductMode,
    PolyFunction,
)
from pspace.parallel import (
    ParallelCoordinateSystem,
    ParallelPolicy,
    DistributedBasisParallel,
    DistributedCoordinateParallel,
)
from pspace.sparsity import CoordinateSystem as SparsityCoordinateSystem


def build_numeric_coordinate_system(basis_type: BasisFunctionType) -> NumericCoordinateSystem:
    factory = CoordinateFactory()
    cs = NumericCoordinateSystem(basis_type)

    uniform = factory.createUniformCoordinate(
        factory.newCoordinateID(),
        "y0",
        dict(a=-1.0, b=1.0),
        max_monomial_dof=3,
    )
    cs.addCoordinateAxis(uniform)

    normal = factory.createNormalCoordinate(
        factory.newCoordinateID(),
        "y1",
        dict(mu=0.0, sigma=1.0),
        max_monomial_dof=2,
    )
    cs.addCoordinateAxis(normal)

    cs.initialize()
    return cs


def make_polynomial(cs: NumericCoordinateSystem) -> PolyFunction:
    terms = [
        (1.0, Counter()),
        (-0.6, Counter({0: 1})),
        (0.4, Counter({0: 2})),
        (0.2, Counter({1: 1})),
    ]
    return PolyFunction(terms, coordinates=cs.coordinates)


class CountingPolicy(ParallelPolicy):
    def __init__(self) -> None:
        self.vector_calls = 0
        self.matrix_calls = 0

    def execute_vector(self, coordinate_system, function, sparse, mode, analytic, thunk):
        self.vector_calls += 1
        return thunk()

    def execute_matrix(self, coordinate_system, function, sparse, symmetric, mode, analytic, thunk):
        self.matrix_calls += 1
        return thunk()


def test_parallel_policy_invocation():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    policy = CountingPolicy()
    wrapper = ParallelCoordinateSystem(base, policy=policy)

    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix = wrapper.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert policy.vector_calls == 1
    assert policy.matrix_calls == 1

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix_direct = base.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    for k in coeffs:
        assert np.isclose(coeffs[k], coeffs_direct[k])
    assert np.allclose(matrix, matrix_direct)


def test_parallel_policy_switching():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    wrapper = ParallelCoordinateSystem(base, policy=DistributedBasisParallel(chunks=2))
    coeffs_basis = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)

    wrapper.configure_policy(DistributedCoordinateParallel())
    coeffs_coord = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs_basis.keys()) == set(coeffs_coord.keys())
    for k in coeffs_basis:
        assert np.isclose(coeffs_basis[k], coeffs_coord[k])


def test_parallel_composed_with_sparsity():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    sparse = SparsityCoordinateSystem(base, enabled=True)
    parallel = ParallelCoordinateSystem(sparse, policy=DistributedBasisParallel())

    poly = make_polynomial(base)
    coeffs_parallel = parallel.decompose(poly, sparse=None, mode=InnerProductMode.NUMERICAL)

    # Disable sparsity after wrapping to ensure delegation reaches through
    parallel.configure_sparsity(False)
    coeffs_dense = parallel.decompose(poly, sparse=None, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs_parallel.keys()) == set(coeffs_dense.keys())
    for k in coeffs_parallel:
        assert np.isclose(coeffs_parallel[k], coeffs_dense[k])
