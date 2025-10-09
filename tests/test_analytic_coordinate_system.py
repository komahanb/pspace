from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pspace.numeric import (
    BasisFunctionType,
    NumericNumericCoordinateSystem,
    InnerProductMode,
    PolyFunction,
)
from pspace.analytic import AnalyticNumericCoordinateSystem
from tests.utils.factories import build_numeric_coordinate_system

DEFAULT_COORDS = [
    ("uniform", dict(a=-0.5, b=1.5), 7),
    ("normal", dict(mu=1.0, sigma=1.0), 6),
]

SMALL_COORDS = [
    ("uniform", dict(a=-0.5, b=0.5), 2),
    ("normal", dict(mu=0.0, sigma=1.0), 2),
]


def make_test_polynomial(cs: NumericNumericCoordinateSystem) -> PolyFunction:
    # f(y) = 2 + 3*y0 + 1.5*y0^2 - 0.75*y1 + 0.25*y0*y1
    terms = [
        (2.0, Counter()),
        (3.0, Counter({0: 1})),
        (1.5, Counter({0: 2})),
        (-0.75, Counter({1: 1})),
        (0.25, Counter({0: 1, 1: 1})),
    ]
    poly = PolyFunction(terms, coordinates=cs.coordinates)
    return poly


def test_analytic_vector_matches_numeric():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, DEFAULT_COORDS)
    poly = make_test_polynomial(cs)

    analytic = AnalyticNumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    coeffs_numeric = cs.decompose(
        poly,
        sparse=False,
        mode=InnerProductMode.NUMERICAL,
    )
    coeffs_analytic = analytic.decompose(
        poly,
        sparse=False,
        mode=InnerProductMode.ANALYTIC,
    )

    assert set(coeffs_numeric.keys()) == set(coeffs_analytic.keys())
    for k in coeffs_numeric:
        assert np.isclose(coeffs_numeric[k], coeffs_analytic[k], atol=1e-8)


def test_analytic_matrix_matches_numeric():
    cs = build_numeric_coordinate_system(BasisFunctionType.TOTAL_DEGREE, DEFAULT_COORDS)
    poly = make_test_polynomial(cs)

    analytic = AnalyticNumericCoordinateSystem(BasisFunctionType.TOTAL_DEGREE, numeric=cs)

    matrix_numeric = cs.decompose_matrix(
        poly,
        sparse=False,
        symmetric=False,
        mode=InnerProductMode.NUMERICAL,
    )
    matrix_analytic = analytic.decompose_matrix(
        poly,
        sparse=False,
        symmetric=False,
        mode=InnerProductMode.ANALYTIC,
    )

    assert matrix_numeric.shape == matrix_analytic.shape
    assert np.allclose(matrix_numeric, matrix_analytic, atol=1e-8)


def test_analytic_rejects_non_analytic_mode():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_test_polynomial(cs)

    analytic = AnalyticNumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    with pytest.raises(ValueError):
        analytic.decompose(poly, mode=InnerProductMode.NUMERICAL)

    with pytest.raises(ValueError):
        analytic.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)


def test_analytic_sparse_matches_dense_vector():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, SMALL_COORDS)
    poly = make_test_polynomial(cs)

    analytic = AnalyticNumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    analytic.configure_sparsity(True)
    coeffs_sparse = analytic.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.ANALYTIC,
    )
    analytic.configure_sparsity(False)
    coeffs_dense = analytic.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.ANALYTIC,
    )
    analytic.configure_sparsity(True)

    assert set(coeffs_sparse.keys()) == set(coeffs_dense.keys())
    for k in coeffs_sparse:
        assert np.isclose(coeffs_sparse[k], coeffs_dense[k], atol=1e-8)


def test_analytic_sparse_matches_dense_matrix():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, SMALL_COORDS)
    poly = make_test_polynomial(cs)

    analytic = AnalyticNumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)

    analytic.configure_sparsity(True)
    matrix_sparse = analytic.decompose_matrix(
        poly,
        sparse=None,
        symmetric=False,
        mode=InnerProductMode.ANALYTIC,
    )
    analytic.configure_sparsity(False)
    matrix_dense = analytic.decompose_matrix(
        poly,
        sparse=None,
        symmetric=False,
        mode=InnerProductMode.ANALYTIC,
    )
    analytic.configure_sparsity(True)

    assert np.allclose(matrix_sparse, matrix_dense, atol=1e-8)


def test_analytic_configure_sparsity_toggle():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, DEFAULT_COORDS)
    analytic = AnalyticNumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, numeric=cs)
    assert analytic.sparsity_enabled is True
    analytic.configure_sparsity(False)
    assert analytic.sparsity_enabled is False
    analytic.configure_sparsity(True)
    assert analytic.sparsity_enabled is True
