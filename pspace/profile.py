from __future__ import annotations

import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .interface import CoordinateSystem as CoordinateSystemInterface
from .core import (
    CoordinateSystem as NumericCoordinateSystem,
    InnerProductMode,
    PolyFunction,
)


class CoordinateSystem(CoordinateSystemInterface):
    """
    Profiling-oriented CoordinateSystem.

    Mirrors the numerical CoordinateSystem API exactly while recording
    wall-clock timings for key operations. Timings are accessible via
    `last_timing(name)` where `name` matches the method identifier
    (e.g., "decompose", "decompose_matrix", "reconstruct").
    """

    def __init__(self, basis_type, verbose: bool = False):
        super().__init__(basis_type, verbose=verbose)
        self.numeric = NumericCoordinateSystem(basis_type, verbose=verbose)
        self._last_timings: dict[str, float] = {}

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
    # Timing utilities                                                    #
    # ------------------------------------------------------------------ #
    def _record_time(self, name: str, func):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        self._last_timings[name] = elapsed
        return result

    def last_timing(self, name: str) -> float | None:
        return self._last_timings.get(name)

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

    def monomial_sparsity_mask(self, f_deg: Counter, symmetric: bool = False):
        return self.numeric.monomial_sparsity_mask(f_deg, symmetric=symmetric)

    def polynomial_sparsity_mask(self, f_degrees: Sequence[Counter], symmetric: bool = False):
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
    # Decomposition / Reconstruction                                     #
    # ------------------------------------------------------------------ #
    def decompose(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ):
        return self._record_time(
            "decompose",
            lambda: self.numeric.decompose(function, sparse=sparse, mode=mode, analytic=analytic),
        )

    def decompose_matrix(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        return self._record_time(
            "decompose_matrix",
            lambda: self.numeric.decompose_matrix(
                function,
                sparse=sparse,
                symmetric=symmetric,
                mode=mode,
                analytic=analytic,
            ),
        )

    def decompose_matrix_analytic(self, function: PolyFunction, sparse: bool | None = None, symmetric: bool = True) -> np.ndarray:
        return self.decompose_matrix(function, sparse=sparse, symmetric=symmetric, mode=InnerProductMode.ANALYTIC)

    def reconstruct(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
        precondition: bool = True,
        method: str = "cholesky",
        tol: float = 0.0,
    ) -> PolyFunction:
        return self._record_time(
            "reconstruct",
            lambda: self.numeric.reconstruct(
                function,
                sparse=sparse,
                mode=mode,
                analytic=analytic,
                precondition=precondition,
                method=method,
                tol=tol,
            ),
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
        return self._record_time(
            "check_decomposition_numerical_symbolic",
            lambda: self.numeric.check_decomposition_numerical_symbolic(
                function, sparse=sparse, tol=tol, verbose=verbose
            ),
        )

    def check_decomposition_numerical_sparse_full(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self._record_time(
            "check_decomposition_numerical_sparse_full",
            lambda: self.numeric.check_decomposition_numerical_sparse_full(function, tol=tol, verbose=verbose),
        )

    def check_decomposition_matrix_sparse_full(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self._record_time(
            "check_decomposition_matrix_sparse_full",
            lambda: self.numeric.check_decomposition_matrix_sparse_full(function, tol=tol, verbose=verbose),
        )

    def check_decomposition_matrix_numerical_symbolic(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self._record_time(
            "check_decomposition_matrix_numerical_symbolic",
            lambda: self.numeric.check_decomposition_matrix_numerical_symbolic(function, tol=tol, verbose=verbose),
        )
