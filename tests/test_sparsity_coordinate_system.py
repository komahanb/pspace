from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pspace.core import BasisFunctionType, InnerProductMode
from pspace.sparsity import SparsityCoordinateSystem
from tests.utils.factories import build_numeric_coordinate_system, make_polynomial


def nonzero_terms(poly, tol: float = 1e-12):
    return [(coeff, degs) for coeff, degs in poly.terms if abs(coeff) > tol]


def test_sparsity_wrapper_matches_base_results():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    wrapper = SparsityCoordinateSystem(base, enabled=True)

    coeffs_sparse = wrapper.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.NUMERICAL,
    )

    wrapper.configure_sparsity(False)
    coeffs_dense = wrapper.decompose(
        poly,
        sparse=None,
        mode=InnerProductMode.NUMERICAL,
    )

    assert set(coeffs_sparse.keys()) == set(coeffs_dense.keys())
    for k in coeffs_sparse:
        assert np.isclose(coeffs_sparse[k], coeffs_dense[k], atol=1e-8)


def test_sparsity_wrapper_respects_per_call_override():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)
    wrapper = SparsityCoordinateSystem(base, enabled=False)

    coeffs_dense = wrapper.decompose(poly, sparse=None, mode=InnerProductMode.NUMERICAL)
    coeffs_force_sparse = wrapper.decompose(poly, sparse=True, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs_dense.keys()) == set(coeffs_force_sparse.keys())
    for k in coeffs_dense:
        assert np.isclose(coeffs_dense[k], coeffs_force_sparse[k], atol=1e-8)


def test_sparsity_wrapper_matrix_paths():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)
    wrapper = SparsityCoordinateSystem(base, enabled=True)

    matrix_sparse = wrapper.decompose_matrix(
        poly,
        sparse=None,
        symmetric=True,
        mode=InnerProductMode.NUMERICAL,
    )
    wrapper.configure_sparsity(False)
    matrix_dense = wrapper.decompose_matrix(
        poly,
        sparse=None,
        symmetric=True,
        mode=InnerProductMode.NUMERICAL,
    )

    assert np.allclose(matrix_sparse, matrix_dense, atol=1e-8)


def test_sparsity_wrapper_reconstruct_delegation():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)
    wrapper = SparsityCoordinateSystem(base, enabled=True)

    recon_sparse = wrapper.reconstruct(poly, sparse=None)
    wrapper.configure_sparsity(False)
    recon_dense = wrapper.reconstruct(poly, sparse=None)

    sparse_terms = nonzero_terms(recon_sparse)
    dense_terms = nonzero_terms(recon_dense)

    assert len(sparse_terms) == len(dense_terms)
    for (coeff_s, deg_s), (coeff_d, deg_d) in zip(sparse_terms, dense_terms):
        assert deg_s == deg_d
        assert abs(coeff_s - coeff_d) < 1e-10
