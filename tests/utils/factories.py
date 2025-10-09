from __future__ import annotations

from collections import Counter
from typing import Iterable

from pspace.numeric import (
    BasisFunctionType,
    CoordinateFactory,
    NumericCoordinateSystem,
    PolyFunction,
    OrthoPolyFunction,
)
from pspace.orthonormal import OrthonormalCoordinateSystem


def build_numeric_coordinate_system(
    basis_type: BasisFunctionType,
    coordinates: Iterable[tuple[str, dict, int]] | None = None,
) -> NumericCoordinateSystem:
    """Construct a numeric coordinate system with standard coordinates.

    Parameters
    ----------
    basis_type : BasisFunctionType
        Basis construction strategy.
    coordinates : Iterable of (dist_name, params, max_degree)
        Optional specification of coordinates. When omitted, a default
        uniform + normal pair is created.
    """
    factory = CoordinateFactory()
    cs = NumericCoordinateSystem(basis_type)

    coords = list(coordinates) if coordinates is not None else [
        ("uniform", dict(a=-1.0, b=1.0), 3),
        ("normal", dict(mu=0.0, sigma=1.0), 2),
    ]

    for name, params, degree in coords:
        coord_id = factory.newCoordinateID()
        if name == "uniform":
            coord = factory.createUniformCoordinate(coord_id, f"y{coord_id}", params, degree)
        elif name == "normal":
            coord = factory.createNormalCoordinate(coord_id, f"y{coord_id}", params, degree)
        elif name == "exponential":
            coord = factory.createExponentialCoordinate(coord_id, f"y{coord_id}", params, degree)
        else:
            raise ValueError(f"Unsupported coordinate distribution '{name}'")
        cs.addCoordinateAxis(coord)

    cs.initialize()
    return cs


def make_polynomial(cs: NumericCoordinateSystem) -> PolyFunction:
    """Create a simple polynomial convenient for tests."""
    terms = [
        (1.0, Counter()),
        (-0.6, Counter({0: 1})),
        (0.4, Counter({0: 2})),
        (0.2, Counter({1: 1})),
    ]
    return PolyFunction(terms, coordinates=cs.coordinates)


def build_orthonormal_coordinate_system(
    basis_type: BasisFunctionType,
    coordinates: Iterable[tuple[str, dict, int]] | None = None,
) -> OrthonormalCoordinateSystem:
    cs = OrthonormalCoordinateSystem(basis_type)
    factory = CoordinateFactory()

    coords = list(coordinates) if coordinates is not None else [
        ("uniform", dict(a=-1.0, b=1.0), 3),
        ("normal", dict(mu=0.0, sigma=1.0), 2),
    ]

    for name, params, degree in coords:
        coord_id = factory.newCoordinateID()
        if name == "uniform":
            coord = factory.createUniformCoordinate(coord_id, f"y{coord_id}", params, degree)
        elif name == "normal":
            coord = factory.createNormalCoordinate(coord_id, f"y{coord_id}", params, degree)
        elif name == "exponential":
            coord = factory.createExponentialCoordinate(coord_id, f"y{coord_id}", params, degree)
        else:
            raise ValueError(f"Unsupported coordinate distribution '{name}'")
        cs.addCoordinateAxis(coord)

    cs.initialize()
    return cs


def make_orthopoly(cs: OrthonormalCoordinateSystem) -> OrthoPolyFunction:
    coeffs = [
        (1.0, Counter()),
        (0.5, Counter({0: 1})),
        (-0.3, Counter({1: 1})),
    ]
    return OrthoPolyFunction(coeffs, cs.coordinates)
