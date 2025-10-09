from __future__ import annotations

from collections import Counter

import numpy as np

from pspace.core import (
    BasisFunctionType,
    CoordinateSystem as NumericCoordinateSystem,
    InnerProductMode,
    PolyFunction,
)
from pspace.symbolic import CoordinateSystem as SymbolicCoordinateSystem
from pspace.analytic import CoordinateSystem as AnalyticCoordinateSystem
from tests.utils.factories import build_numeric_coordinate_system

DEFAULT_COORDS = [
    ("uniform", dict(a=-0.75, b=0.75), 4),
    ("normal", dict(mu=0.5, sigma=0.75), 3),
]

def make_test_polynomial(cs: NumericCoordinateSystem) -> PolyFunction:
    terms = [
        (1.0, Counter()),
        (0.8, Counter({0: 1})),
        (-1.1, Counter({1: 1})),
        (0.5, Counter({0: 2})),
        (0.3, Counter({0: 1, 1: 1})),
    ]
    return PolyFunction(terms, coordinates=cs.coordinates)


def make_axis_aligned_polynomial(cs: NumericCoordinateSystem) -> PolyFunction:
    terms = [
        (1.0, Counter()),
        (0.7, Counter({0: 1})),
        (-0.2, Counter({0: 2})),
    ]
    return PolyFunction(terms, coordinates=cs.coordinates)


def test_numeric_default_mode_matches_explicit_numerical():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, DEFAULT_COORDS)
    poly = make_test_polynomial(cs)

    coeffs_default = cs.decompose(
        poly,
        sparse=True,
    )
    coeffs_explicit = cs.decompose(
        poly,
        sparse=True,
        mode=InnerProductMode.NUMERICAL,
    )

    assert set(coeffs_default.keys()) == set(coeffs_explicit.keys())
    for k in coeffs_default:
        assert np.isclose(coeffs_default[k], coeffs_explicit[k], atol=1e-12)


def test_numeric_sparse_and_dense_match_for_vector():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, DEFAULT_COORDS)
    poly = make_test_polynomial(cs)

    cs.configure_sparsity(True)
    coeffs_sparse = cs.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.NUMERICAL,
    )

    cs.configure_sparsity(False)
    coeffs_dense = cs.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.NUMERICAL,
    )
    cs.configure_sparsity(True)

    assert set(coeffs_sparse.keys()) == set(coeffs_dense.keys())
    for k in coeffs_sparse:
        assert np.isclose(coeffs_sparse[k], coeffs_dense[k], atol=1e-8)


def test_numeric_delegates_symbolic_and_analytic_modes():
    cs = build_numeric_coordinate_system(BasisFunctionType.TOTAL_DEGREE, DEFAULT_COORDS)
    poly = make_test_polynomial(cs)

    symbolic = SymbolicCoordinateSystem(BasisFunctionType.TOTAL_DEGREE, numeric=cs)
    analytic = AnalyticCoordinateSystem(BasisFunctionType.TOTAL_DEGREE, numeric=cs)

    coeffs_symbolic_core = cs.decompose(
        poly,
        sparse=False,
        mode=InnerProductMode.SYMBOLIC,
    )
    coeffs_symbolic_direct = symbolic.decompose(
        poly,
        sparse=False,
        mode=InnerProductMode.SYMBOLIC,
    )
    assert coeffs_symbolic_core == coeffs_symbolic_direct

    matrix_analytic_core = cs.decompose_matrix(
        poly,
        sparse=False,
        symmetric=False,
        mode=InnerProductMode.ANALYTIC,
    )
    matrix_analytic_direct = analytic.decompose_matrix(
        poly,
        sparse=False,
        symmetric=False,
        mode=InnerProductMode.ANALYTIC,
    )
    assert np.allclose(matrix_analytic_core, matrix_analytic_direct, atol=1e-8)


def test_tensor_sparsity_mask_excludes_unused_axes():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, DEFAULT_COORDS)
    poly = make_axis_aligned_polynomial(cs)
    cs.configure_sparsity(True)
    mask = cs.polynomial_vector_sparsity_mask(poly.degrees)
    assert mask  # ensure not empty
    for basis_id in mask:
        entry = cs.basis[basis_id]
        assert entry.get(1, 0) == 0  # axis y1 has zero degree in polynomial


def test_total_degree_sparsity_mask_excludes_unused_axes():
    cs = build_numeric_coordinate_system(BasisFunctionType.TOTAL_DEGREE, DEFAULT_COORDS)
    poly = make_axis_aligned_polynomial(cs)
    cs.configure_sparsity(True)
    mask = cs.polynomial_vector_sparsity_mask(poly.degrees)
    assert mask
    for basis_id in mask:
        entry = cs.basis[basis_id]
        assert entry.get(1, 0) == 0


def test_numeric_configure_sparsity_toggle():
    cs = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, DEFAULT_COORDS)
    assert cs.sparsity_enabled is True
    cs.configure_sparsity(False)
    assert cs.sparsity_enabled is False
    cs.configure_sparsity(True)
    assert cs.sparsity_enabled is True
