from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np

from pspace.numeric import (
    BasisFunctionType,
    CoordinateFactory,
    NumericNumericCoordinateSystem,
    PolyFunction,
)
from pspace.plotter import NumericCoordinateSystem as PlottingNumericCoordinateSystem


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_coordinate_systems(
    basis_type: BasisFunctionType,
    num_coords: int,
    max_degree: int,
    generator: np.random.Generator,
) -> Tuple[NumericNumericCoordinateSystem, PlottingNumericCoordinateSystem]:
    cf = CoordinateFactory()
    numeric = NumericNumericCoordinateSystem(basis_type)

    for idx in range(num_coords):
        coord_id = cf.newCoordinateID()
        name = f"y{idx}"
        deg = int(generator.integers(1, max_degree + 1))
        choice = generator.choice(["normal", "uniform", "exponential"])

        if choice == "normal":
            mu = float(generator.uniform(-2.0, 2.0))
            sigma = float(generator.uniform(0.5, 2.0))
            coord = cf.createNormalCoordinate(coord_id, name, dict(mu=mu, sigma=sigma), deg)
        elif choice == "uniform":
            a = float(generator.uniform(-3.0, 1.0))
            b = float(a + generator.uniform(1.0, 4.0))
            coord = cf.createUniformCoordinate(coord_id, name, dict(a=a, b=b), deg)
        else:
            mu = float(generator.uniform(0.0, 2.0))
            beta = float(generator.uniform(0.5, 2.5))
            coord = cf.createExponentialCoordinate(coord_id, name, dict(mu=mu, beta=beta), deg)

        numeric.addCoordinateAxis(coord)

    numeric.initialize()

    plotting = PlottingNumericCoordinateSystem(basis_type, verbose=False)
    for coord in numeric.coordinates.values():
        plotting.addCoordinateAxis(coord)
    plotting.initialize()

    return numeric, plotting


def random_polynomial(
    numeric: NumericNumericCoordinateSystem,
    generator: np.random.Generator,
    max_degree: int,
    max_terms: int = 6,
) -> PolyFunction:
    coord_ids = list(numeric.coordinates.keys())
    terms: List[tuple[float, Counter]] = []
    terms.append((float(generator.uniform(-2.0, 2.0)), Counter()))

    for _ in range(max_terms - 1):
        coeff = float(generator.uniform(-2.0, 2.0))
        degs = Counter()
        for cid in coord_ids:
            cap = numeric.coordinates[cid].degree
            deg = int(generator.integers(0, min(max_degree, cap) + 1))
            if deg:
                degs[cid] = deg
        if degs:
            terms.append((coeff, degs))

    return PolyFunction(terms, coordinates=numeric.coordinates)
