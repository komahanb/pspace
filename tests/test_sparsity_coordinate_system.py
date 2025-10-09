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
        (1.2, Counter({0: 1})),
        (-0.8, Counter({0: 2})),
    ]
    return PolyFunction(terms, coordinates=cs.coordinates)


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

    assert len(recon_sparse.terms) == len(recon_dense.terms)
    for (coeff_s, deg_s), (coeff_d, deg_d) in zip(recon_sparse.terms, recon_dense.terms):
        assert deg_s == deg_d
        assert abs(coeff_s - coeff_d) < 1e-10
