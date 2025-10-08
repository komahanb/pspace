from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import hermite_e as herm_e
from numpy.polynomial import legendre as leg
from numpy.polynomial import laguerre as lag

from .interface import CoordinateSystem as CoordinateSystemInterface
from .core import (
    CoordinateSystem as NumericCoordinateSystem,
    DistributionType,
    InnerProductMode,
    PolyFunction,
)


# --------------------------------------------------------------------------- #
# Helper utilities for closed-form integration
# --------------------------------------------------------------------------- #

def _shift_legendre_power_to_unit(power_x: np.ndarray) -> np.ndarray:
    base = np.array([-1.0, 2.0], dtype=float)  # represents (2z - 1)
    result = np.array([0.0], dtype=float)
    for m, c in enumerate(power_x):
        if abs(c) < 1e-15:
            continue
        term = poly.polypow(base, m)
        result = poly.polyadd(result, c * term)
    return result


def _trim_coeffs(array, tol: float = 1e-15):
    arr = np.array(array, dtype=float)
    if not arr.size:
        return tuple()
    while arr.size > 1 and abs(arr[-1]) < tol:
        arr = arr[:-1]
    return tuple(float(x) for x in arr)


@lru_cache(None)
def _phi_power_coeffs(sig, degree):
    if degree == 0:
        return (1.0,)
    dist = sig[0]
    if dist == "normal":
        mu, sigma = sig[1], sig[2]
        coeffs = [
            math.comb(degree, m) * (mu ** (degree - m)) * (sigma ** m)
            for m in range(degree + 1)
        ]
        return _trim_coeffs(coeffs)
    if dist == "uniform":
        a, b = sig[1], sig[2]
        span = b - a
        coeffs = [
            math.comb(degree, m) * (a ** (degree - m)) * (span ** m)
            for m in range(degree + 1)
        ]
        return _trim_coeffs(coeffs)
    if dist == "exponential":
        mu, beta = sig[1], sig[2]
        coeffs = [
            math.comb(degree, m) * (mu ** (degree - m)) * (beta ** m)
            for m in range(degree + 1)
        ]
        return _trim_coeffs(coeffs)
    raise NotImplementedError


@lru_cache(None)
def _psi_power_coeffs(sig, degree):
    if degree == 0:
        return (1.0,)
    dist = sig[0]
    if dist == "normal":
        coeffs = [0.0] * (degree + 1)
        coeffs[degree] = 1.0
        power = herm_e.herme2poly(coeffs)
        norm = math.sqrt(math.factorial(degree))
        return _trim_coeffs(np.asarray(power, dtype=float) / norm)
    if dist == "uniform":
        coeffs = [0.0] * (degree + 1)
        coeffs[degree] = 1.0
        power_x = leg.leg2poly(coeffs)
        power_z = _shift_legendre_power_to_unit(power_x)
        factor = math.sqrt(2 * degree + 1)
        return _trim_coeffs(power_z * factor)
    if dist == "exponential":
        coeffs = [0.0] * (degree + 1)
        coeffs[degree] = 1.0
        power = lag.lag2poly(coeffs)
        return _trim_coeffs(power)
    raise NotImplementedError


@lru_cache(None)
def _gaussian_moment(m: int) -> float:
    if m % 2 == 1:
        return 0.0
    n = m // 2
    return math.factorial(m) / (2 ** n * math.factorial(n))


def _integrate_power_series(sig, coeffs) -> float:
    dist = sig[0]
    total = 0.0
    coeffs = np.asarray(coeffs, dtype=float)
    if dist == "normal":
        for m, c in enumerate(coeffs):
            if abs(c) < 1e-15:
                continue
            total += c * _gaussian_moment(m)
        return float(total)
    if dist == "uniform":
        for m, c in enumerate(coeffs):
            if abs(c) < 1e-15:
                continue
            total += c / (m + 1)
        return float(total)
    if dist == "exponential":
        for m, c in enumerate(coeffs):
            if abs(c) < 1e-15:
                continue
            total += c * math.factorial(m)
        return float(total)
    raise NotImplementedError


@lru_cache(None)
def _vector_axis_integral(sig, phi_deg, psi_deg):
    phi = np.asarray(_phi_power_coeffs(sig, phi_deg), dtype=float)
    psi = np.asarray(_psi_power_coeffs(sig, psi_deg), dtype=float)
    product = poly.polymul(phi, psi)
    return float(_integrate_power_series(sig, product))


@lru_cache(None)
def _matrix_axis_integral(sig, phi_deg, psi_i_deg, psi_j_deg):
    phi = np.asarray(_phi_power_coeffs(sig, phi_deg), dtype=float)
    psi_i = np.asarray(_psi_power_coeffs(sig, psi_i_deg), dtype=float)
    psi_j = np.asarray(_psi_power_coeffs(sig, psi_j_deg), dtype=float)
    tmp = poly.polymul(phi, psi_i)
    product = poly.polymul(tmp, psi_j)
    return float(_integrate_power_series(sig, product))


def _coord_signature(coord):
    dist = coord.distribution
    if dist == DistributionType.NORMAL:
        mu = float(coord.dist_coords["mu"])
        sigma = float(coord.dist_coords["sigma"])
        return ("normal", mu, sigma)
    if dist == DistributionType.UNIFORM:
        a = float(coord.dist_coords["a"])
        b = float(coord.dist_coords["b"])
        return ("uniform", a, b)
    if dist == DistributionType.EXPONENTIAL:
        mu = float(coord.dist_coords["mu"])
        beta = float(coord.dist_coords["beta"])
        return ("exponential", mu, beta)
    raise NotImplementedError(
        f"Closed-form integration not implemented for distribution {dist}"
    )


# --------------------------------------------------------------------------- #
# Analytic inner-product operators
# --------------------------------------------------------------------------- #

class AnalyticVectorInnerProductOperator:
    """Rank-1 closed-form inner products."""

    def __init__(self, coordinate_system: "CoordinateSystem") -> None:
        self.cs = coordinate_system

    def compute(self, function: PolyFunction, sparse: bool = True) -> dict[int, float]:
        cs = self.cs
        function.bind_coordinates(cs.coordinates)

        if sparse:
            mask = cs.polynomial_vector_sparsity_mask(function.degrees)
        else:
            mask = cs.basis.keys()

        coeffs = {k: 0.0 for k in mask}
        basis = cs.basis
        coord_items = list(cs.coordinates.items())

        for coeff, degs in function.terms:
            base_coeff = float(coeff)
            if abs(base_coeff) < 1e-15:
                continue
            for k in mask:
                psi_degs = basis[k]
                val = base_coeff
                for cid, coord in coord_items:
                    sig = _coord_signature(coord)
                    phi_deg = degs.get(cid, 0)
                    psi_deg = psi_degs.get(cid, 0)
                    val *= _vector_axis_integral(sig, phi_deg, psi_deg)
                    if abs(val) < 1e-18:
                        break
                if abs(val) >= 1e-15:
                    coeffs[k] += val

        if sparse:
            for k in cs.basis:
                coeffs.setdefault(k, 0.0)

        return coeffs


class AnalyticMatrixInnerProductOperator:
    """Rank-2 closed-form inner products."""

    def __init__(self, coordinate_system: "CoordinateSystem") -> None:
        self.cs = coordinate_system

    def compute(
        self,
        function: PolyFunction,
        sparse: bool = False,
        symmetric: bool = True,
    ) -> np.ndarray:
        cs = self.cs
        function.bind_coordinates(cs.coordinates)

        nbasis = cs.getNumBasisFunctions()
        matrix = np.zeros((nbasis, nbasis))

        if sparse:
            mask = cs.polynomial_sparsity_mask(function.degrees, symmetric=symmetric)
        else:
            if symmetric:
                mask = {(i, j) for i in cs.basis for j in cs.basis if i <= j}
            else:
                mask = {(i, j) for i in cs.basis for j in cs.basis}

        basis = cs.basis
        coord_items = list(cs.coordinates.items())

        for coeff, degs in function.terms:
            base_coeff = float(coeff)
            if abs(base_coeff) < 1e-15:
                continue
            for i, j in mask:
                psi_i = basis[i]
                psi_j = basis[j]
                val = base_coeff
                for cid, coord in coord_items:
                    sig = _coord_signature(coord)
                    phi_deg = degs.get(cid, 0)
                    deg_i = psi_i.get(cid, 0)
                    deg_j = psi_j.get(cid, 0)
                    val *= _matrix_axis_integral(sig, phi_deg, deg_i, deg_j)
                    if abs(val) < 1e-18:
                        break
                if abs(val) >= 1e-15:
                    matrix[i, j] += val
                    if symmetric and i != j:
                        matrix[j, i] += val

        return matrix


# --------------------------------------------------------------------------- #
# Analytic Coordinate System mirror
# --------------------------------------------------------------------------- #

class CoordinateSystem(CoordinateSystemInterface):
    """
    Analytic mirror of the CoordinateSystem contract. Reuses the numeric core
    for structural information while evaluating inner products with closed-form
    formulas.
    """

    def __init__(
        self,
        basis_type,
        verbose: bool = False,
        numeric: NumericCoordinateSystem | None = None,
    ) -> None:
        super().__init__(basis_type, verbose=verbose)
        self.numeric = numeric or NumericCoordinateSystem(basis_type, verbose=verbose)
        self._vector_analytic = AnalyticVectorInnerProductOperator(self)
        self._matrix_analytic = AnalyticMatrixInnerProductOperator(self)

    # ------------------------------------------------------------------ #
    # Shared state                                                       #
    # ------------------------------------------------------------------ #
    @property
    def coordinates(self):
        return self.numeric.coordinates

    @property
    def basis(self):
        return self.numeric.basis

    @property
    def sparsity_enabled(self) -> bool:
        return self.numeric.sparsity_enabled

    def configure_sparsity(self, enabled: bool) -> None:
        self.numeric.configure_sparsity(enabled)

    # ------------------------------------------------------------------ #
    # Coordinate management                                              #
    # ------------------------------------------------------------------ #
    def addCoordinateAxis(self, coordinate: Any) -> None:
        self.numeric.addCoordinateAxis(coordinate)

    def initialize(self) -> None:
        self.numeric.initialize()

    def getNumBasisFunctions(self) -> int:
        return self.numeric.getNumBasisFunctions()

    def getNumCoordinateAxes(self) -> int:
        return self.numeric.getNumCoordinateAxes()

    def getMonomialDegreeCoordinates(self) -> Mapping[int, int]:
        return self.numeric.getMonomialDegreeCoordinates()

    # ------------------------------------------------------------------ #
    # Basis evaluation / quadrature                                      #
    # ------------------------------------------------------------------ #
    def evaluate_basis(self, yscalar: float, degree: int) -> Any:
        return self.numeric.evaluate_basis(yscalar, degree)

    def print_quadrature(self, qmap: Mapping[int, Mapping[str, Any]]) -> None:
        self.numeric.print_quadrature(qmap)

    def build_quadrature(self, degrees: Counter) -> Mapping[int, Mapping[str, Any]]:
        return self.numeric.build_quadrature(degrees)

    def evaluateBasisDegreesY(self, y_by_cid: Mapping[int, float], degrees_counter: Counter) -> Any:
        return self.numeric.evaluateBasisDegreesY(y_by_cid, degrees_counter)

    def evaluateBasisIndexY(self, y_by_cid: Mapping[int, float], basis_id: int) -> Any:
        return self.numeric.evaluateBasisIndexY(y_by_cid, basis_id)

    # ------------------------------------------------------------------ #
    # Sparsity helpers                                                   #
    # ------------------------------------------------------------------ #
    def sparse_vector(self, dmapi: Counter, dmapf: Counter) -> bool:
        return self.numeric.sparse_vector(dmapi, dmapf)

    def monomial_vector_sparsity_mask(self, f_deg: Counter) -> Sequence[int]:
        return self.numeric.monomial_vector_sparsity_mask(f_deg)

    def polynomial_vector_sparsity_mask(self, f_degrees: Sequence[Counter]) -> Sequence[int]:
        return self.numeric.polynomial_vector_sparsity_mask(f_degrees)

    def monomial_sparsity_mask(self, f_deg: Counter, symmetric: bool = False) -> Sequence[tuple[int, int]]:
        return self.numeric.monomial_sparsity_mask(f_deg, symmetric=symmetric)

    def polynomial_sparsity_mask(self, f_degrees: Sequence[Counter], symmetric: bool = False) -> Sequence[tuple[int, int]]:
        return self.numeric.polynomial_sparsity_mask(f_degrees, symmetric=symmetric)

    # ------------------------------------------------------------------ #
    # Inner products                                                     #
    # ------------------------------------------------------------------ #
    def inner_product(self, f_eval: Any, g_eval: Any, f_deg: Counter | None = None, g_deg: Counter | None = None) -> float:
        return self.numeric.inner_product(f_eval, g_eval, f_deg=f_deg, g_deg=g_deg)

    def inner_product_basis(
        self,
        i_id: int,
        j_id: int,
        f_eval: Any = None,
        f_deg: Counter | None = None,
    ) -> float:
        return self.numeric.inner_product_basis(i_id, j_id, f_eval=f_eval, f_deg=f_deg)

    # ------------------------------------------------------------------ #
    # Decomposition / Reconstruction                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_mode(mode: InnerProductMode | str | bool | None) -> InnerProductMode:
        if mode is None:
            return InnerProductMode.ANALYTIC
        if isinstance(mode, bool):
            return InnerProductMode.ANALYTIC if mode else InnerProductMode.NUMERICAL
        if isinstance(mode, InnerProductMode):
            return mode
        if isinstance(mode, str):
            return InnerProductMode(mode.lower())
        raise ValueError(f"Unsupported inner product mode for analytic backend: {mode!r}")

    def decompose(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> dict[int, float]:
        if sparse is None:
            sparse = self.numeric.sparsity_enabled

        normalized = self._normalize_mode(mode)
        if normalized is not InnerProductMode.ANALYTIC:
            raise ValueError("Analytic CoordinateSystem supports only analytic decomposition.")
        return self._vector_analytic.compute(function, sparse=sparse)

    def decompose_matrix(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        if sparse is None:
            sparse = self.numeric.sparsity_enabled

        normalized = self._normalize_mode(mode)
        if normalized is not InnerProductMode.ANALYTIC:
            raise ValueError("Analytic CoordinateSystem supports only analytic decomposition.")
        return self._matrix_analytic.compute(function, sparse=sparse, symmetric=symmetric)

    def decompose_matrix_analytic(self, function: PolyFunction, sparse: bool | None = None, symmetric: bool = True) -> np.ndarray:
        if sparse is None:
            sparse = self.numeric.sparsity_enabled
        return self._matrix_analytic.compute(function, sparse=sparse, symmetric=symmetric)

    def reconstruct(
        self,
        function: PolyFunction,
        sparse: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
        precondition: bool = True,
        method: str = "cholesky",
        tol: float = 0.0,
    ) -> PolyFunction:
        return self.numeric.reconstruct(
            function,
            sparse=sparse,
            mode=mode,
            analytic=analytic,
            precondition=precondition,
            method=method,
            tol=tol,
        )

    # ------------------------------------------------------------------ #
    # Consistency checks                                                 #
    # ------------------------------------------------------------------ #
    def check_orthonormality(self) -> float:
        return self.numeric.check_orthonormality()

    def check_decomposition_numerical_symbolic(
        self,
        function: PolyFunction,
        sparse: bool = True,
        tol: float = 1e-10,
        verbose: bool = True,
    ):
        return self.numeric.check_decomposition_numerical_symbolic(function, sparse=sparse, tol=tol, verbose=verbose)

    def check_decomposition_numerical_sparse_full(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self.numeric.check_decomposition_numerical_sparse_full(function, tol=tol, verbose=verbose)

    def check_decomposition_matrix_sparse_full(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self.numeric.check_decomposition_matrix_sparse_full(function, tol=tol, verbose=verbose)

    def check_decomposition_matrix_numerical_symbolic(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self.numeric.check_decomposition_matrix_numerical_symbolic(function, tol=tol, verbose=verbose)
