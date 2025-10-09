from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pspace.numeric import (
    BasisFunctionType,
    PolyFunction,
)
from pspace.orthonormal import OrthonormalCoordinateSystem
from tests.utils.factories import (
    build_numeric_coordinate_system,
    build_orthonormal_coordinate_system,
    make_polynomial,
    make_orthopoly,
)

COORDS_DEFAULT = [
    ("uniform", dict(a=-1.0, b=1.0), 3),
    ("uniform", dict(a=-0.5, b=0.5), 2),
]


def test_orthonormal_decompose_matches_numeric():
    numeric = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE, COORDS_DEFAULT)
    ortho = build_orthonormal_coordinate_system(BasisFunctionType.TENSOR_DEGREE, COORDS_DEFAULT)

    poly = make_polynomial(numeric)
    ortho_poly = ortho.to_orthopoly(poly)

    coeffs_numeric = numeric.decompose(poly, sparse=True, mode=None)
    coeffs_ortho = ortho.decompose(ortho_poly, sparse=True, mode=None)

    assert coeffs_numeric.keys() == coeffs_ortho.keys()
    for k in coeffs_numeric:
        assert np.isclose(coeffs_numeric[k], coeffs_ortho[k])


def test_orthonormal_reconstruct_roundtrip():
    cs = build_orthonormal_coordinate_system(BasisFunctionType.TENSOR_DEGREE, COORDS_DEFAULT)
    ortho_poly = make_orthopoly(cs)

    reconstructed = cs.reconstruct(ortho_poly, sparse=True)

    pts = [
        {coord_id: 0.0 for coord_id in cs.coordinates},
        {coord_id: 0.5 for coord_id in cs.coordinates},
        {coord_id: -0.25 for coord_id in cs.coordinates},
    ]
    for pt in pts:
        f_val = float(ortho_poly(pt))
        f_rec = float(reconstructed(pt))
        assert np.isclose(f_val, f_rec, atol=1e-9)
