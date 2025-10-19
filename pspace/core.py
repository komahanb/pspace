from __future__ import annotations

import math
from collections import Counter
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


__all__ = [
    "CoordinateType",
    "DistributionType",
    "BasisFunctionType",
    "InnerProductMode",
    "PolyFunction",
    "OrthoPolyFunction",
]


class CoordinateType(Enum):
    """High level grouping for coordinate axes."""

    PROBABILISTIC = 1
    SPATIAL = 2
    TEMPORAL = 3


class DistributionType(Enum):
    """Supported marginal distributions per coordinate axis."""

    NORMAL = 0
    UNIFORM = 1
    EXPONENTIAL = 2
    POISSON = 3
    BINORMAL = 4


class BasisFunctionType(Enum):
    """Basis construction options for coordinate systems."""

    TENSOR_DEGREE = 0
    TOTAL_DEGREE = 1
    ADAPTIVE_DEGREE = 2


class InnerProductMode(Enum):
    """Available evaluation backends for inner products."""

    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    ANALYTIC = "analytic"


class PolyFunction:
    """Polynomial represented in the physical (monomial) basis."""

    def __init__(self, terms: Iterable[tuple[float, Counter]], coordinates: Mapping[int, Any] | None = None):
        """
        Parameters
        ----------
        terms:
            Iterable of (coefficient, Counter(axis -> degree)) pairs.
        coordinates:
            Optional coordinate dictionary providing `phi_y` evaluations.
        """

        self._terms: list[tuple[float, Counter]] = []
        self._degrees: list[Counter] = []
        self._max_degrees: Counter = Counter()
        self._coords = coordinates

        for term in terms:
            if isinstance(term, tuple) and len(term) == 2 and isinstance(term[1], Counter):
                coeff, degs = term
                self._terms.append((coeff, degs))
                self._degrees.append(degs)
                for axis, deg in degs.items():
                    self._max_degrees[axis] = max(self._max_degrees.get(axis, 0), deg)
            else:
                raise TypeError(f"Invalid term format: {term!r}")

    @property
    def terms(self) -> Sequence[tuple[float, Counter]]:
        return self._terms

    @property
    def degrees(self) -> Sequence[Counter]:
        """Degree structure per monomial."""
        return self._degrees

    @property
    def max_degrees(self) -> Counter:
        """Maximum degree encountered per axis."""
        return self._max_degrees

    @property
    def coordinates(self):
        return self._coords

    def bind_coordinates(self, coordinates: Mapping[int, Any]) -> None:
        """Attach coordinate dictionary used for physical basis evaluation."""
        self._coords = coordinates

    def __call__(self, y_by_axis: Mapping[int, float]) -> float:
        total = 0.0
        coords = self._coords
        for coeff, degs in self._terms:
            mon = coeff
            for axis, deg in degs.items():
                yval = y_by_axis[axis]
                if coords is not None and axis in coords:
                    mon *= coords[axis].phi_y(yval, deg)
                else:
                    mon *= yval**deg
            total += mon
        return total

    def __repr__(self) -> str:
        return f"PolyFunction({self._terms})"


class OrthoPolyFunction:
    """
    Polynomial expressed in orthonormal coordinates (Legendre, Hermite, etc.).
    """

    def __init__(self, terms: Sequence[tuple[float, Counter]], coordinates: Mapping[int, Any]):
        self._terms = terms
        self._coords = coordinates

    def __call__(self, y_by_axis: Mapping[int, float]) -> float:
        total = 0.0
        for coeff, degs in self._terms:
            term_val = coeff
            for axis, deg in degs.items():
                term_val *= self._coords[axis].psi_y(y_by_axis[axis], deg)
            total += term_val
        return total

    def coeffs(self) -> Mapping[tuple[tuple[int, int], ...], float]:
        """Return coefficients keyed by sorted degree tuples."""
        return {tuple(sorted(degs.items())): coeff for coeff, degs in self._terms}

    def toPolyFunction(self) -> PolyFunction:
        """
        Expand into the monomial basis using change-of-basis matrices.
        Currently supports Legendre coordinates.
        """
        from numpy.polynomial import legendre as npleg
        from numpy.polynomial import polynomial as nppoly

        monomial_terms: dict[tuple[tuple[int, int], ...], float] = {}

        for coeff, degs in self._terms:
            expansions: list[tuple[int, np.ndarray]] = []
            for axis, deg in degs.items():
                coord = self._coords[axis]
                if coord.distribution.name == "UNIFORM":
                    Pn = npleg.Legendre.basis(deg)
                    poly = Pn.convert(kind=nppoly.Polynomial)
                    coeffs_power = np.array(poly.coef, dtype=float)
                    scale = math.sqrt((2 * deg + 1) / 2.0)
                    coeffs_power *= scale
                    expansions.append((axis, coeffs_power))
                else:
                    raise NotImplementedError("toPolyFunction only supports Legendre for now")

            def recurse(idx: int, running_coeff: float, running_degs: Counter) -> None:
                if idx == len(expansions):
                    key = tuple(sorted(running_degs.items()))
                    monomial_terms[key] = monomial_terms.get(key, 0.0) + coeff * running_coeff
                    return
                axis, coeffs_power = expansions[idx]
                for power, value in enumerate(coeffs_power):
                    if abs(value) < 1e-15:
                        continue
                    recurse(idx + 1, running_coeff * value, running_degs + Counter({axis: power}))

            recurse(0, 1.0, Counter())

        terms = [
            (float(value), Counter(dict(key)))
            for key, value in monomial_terms.items()
            if abs(value) > 1e-15
        ]
        return PolyFunction(terms, coordinates=self._coords)
