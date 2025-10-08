from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pspace.core import (
    BasisFunctionType,
    CoordinateFactory,
    CoordinateSystem as NumericCoordinateSystem,
    InnerProductMode,
    PolyFunction,
)
from pspace.symbolic import CoordinateSystem as SymbolicCoordinateSystem


def build_numeric_coordinate_system(basis_type: BasisFunctionType) -> NumericCoordinateSystem:
    factory = CoordinateFactory()
    cs = NumericCoordinateSystem(basis_type)

    uniform = factory.createUniformCoordinate(
        factory.newCoordinateID(),
        "y0",
        dict(a=-1.0, b=1.0),
        max_monomial_dof=4,
    )
    cs.addCoordinateAxis(uniform)

    exponential = factory.createExponentialCoordinate(
        factory.newCoordinateID(),
        "y1",
        dict(mu=0.0, beta=1.0),
        max_monomial_dof=3,
    )
    cs.addCoordinateAxis(exponential)

    cs.initialize()
    return cs


def build_small_numeric_coordinate_system(basis_type: BasisFunctionType) -> NumericCoordinateSystem:
    factory = CoordinateFactory()
    cs = NumericCoordinateSystem(basis_type)

    uniform = factory.createUniformCoordinate(
        factory.newCoordinateID(),
        "y0",
        dict(a=-1.0, b=1.0),
        max_monomial_dof=2,
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


def make_test_polynomial(cs: NumericCoordinateSystem) -> PolyFunction:
    terms = [
        (1.0, Counter()),
        (-2.5, Counter({0: 1})),
        (0.75, Counter({1: 1})),
        (1.2, Counter({0: 2})),
        (-0.6, Counter({0: 1, 1: 1})),
    ]
    return PolyFunction(terms, coordinates=cs.coordinates)


def test_symbolic_vector_matches_numeric():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_test_polynomial(cs)

    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    coeffs_numeric = cs.decompose(
        poly,
        sparse=False,
        mode=InnerProductMode.NUMERICAL,
    )
    coeffs_symbolic = symbolic.decompose(
        poly,
        sparse=False,
        mode=InnerProductMode.SYMBOLIC,
    )

    assert set(coeffs_numeric.keys()) == set(coeffs_symbolic.keys())
    for k in coeffs_numeric:
        assert np.isclose(coeffs_numeric[k], coeffs_symbolic[k], atol=1e-8)


def test_symbolic_matrix_matches_numeric():
    cs = build_numeric_coordinate_system(BasisFunctionType.TOTAL_DEGREE)
    poly = make_test_polynomial(cs)

    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TOTAL_DEGREE, numeric=cs)

    matrix_numeric = cs.decompose_matrix(
        poly,
        sparse=False,
        symmetric=False,
        mode=InnerProductMode.NUMERICAL,
    )
    matrix_symbolic = symbolic.decompose_matrix(
        poly,
        sparse=False,
        symmetric=False,
        mode=InnerProductMode.SYMBOLIC,
    )

    assert matrix_numeric.shape == matrix_symbolic.shape
    assert np.allclose(matrix_numeric, matrix_symbolic, atol=1e-8)


def test_symbolic_rejects_non_symbolic_mode():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_test_polynomial(cs)

    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    with pytest.raises(ValueError):
        symbolic.decompose(poly, mode=InnerProductMode.NUMERICAL)

    with pytest.raises(ValueError):
        symbolic.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)


def test_symbolic_sparse_matches_dense_vector():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_test_polynomial(cs)

    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    symbolic.configure_sparsity(True)
    coeffs_sparse = symbolic.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.SYMBOLIC,
    )
    symbolic.configure_sparsity(False)
    coeffs_dense = symbolic.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.SYMBOLIC,
    )
    symbolic.configure_sparsity(True)

    assert set(coeffs_sparse.keys()) == set(coeffs_dense.keys())
    for k in coeffs_sparse:
        assert np.isclose(coeffs_sparse[k], coeffs_dense[k], atol=1e-8)


def test_symbolic_sparse_matches_dense_matrix():
    cs = build_small_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_test_polynomial(cs)

    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    symbolic.configure_sparsity(True)
    matrix_sparse = symbolic.decompose_matrix(
        poly,
        sparse=None,
        symmetric=False,
        mode=InnerProductMode.SYMBOLIC,
    )
    symbolic.configure_sparsity(False)
    matrix_dense = symbolic.decompose_matrix(
        poly,
        sparse=None,
        symmetric=False,
        mode=InnerProductMode.SYMBOLIC,
    )
    symbolic.configure_sparsity(True)

    assert np.allclose(matrix_sparse, matrix_dense, atol=1e-8)


def test_symbolic_configure_sparsity_toggle():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)
    assert symbolic.sparsity_enabled is True
    symbolic.configure_sparsity(False)
    assert symbolic.sparsity_enabled is False
    symbolic.configure_sparsity(True)
    assert symbolic.sparsity_enabled is True
