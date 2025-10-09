from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .numeric import InnerProductMode, PolyFunction
from .interface import CoordinateSystem


class SparsityCoordinateSystem(CoordinateSystem):
    """
    Sparsity-aware decorator for CoordinateSystem instances.

    Wraps any coordinate-system mirror and enforces sparse/dense execution
    according to the configured policy. By default, the wrapped instance's
    sparsity setting is mirrored; callers may override per-call via the
    ``sparse`` keyword argument (as usual) or globally with
    ``configure_sparsity``.
    """

    def __init__(
        self,
        base: CoordinateSystem,
        enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__(base.basis_construction, verbose=verbose)
        self._base = base
        self._enabled = bool(enabled)
        self._base.configure_sparsity(self._enabled)

    # ------------------------------------------------------------------ #
    # Shared state                                                       #
    # ------------------------------------------------------------------ #
    @property
    def coordinates(self) -> Mapping[int, Any]:
        return self._base.coordinates

    @property
    def basis(self) -> Mapping[int, Counter]:
        return self._base.basis

    @property
    def sparsity_enabled(self) -> bool:
        return self._enabled

    def configure_sparsity(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._base.configure_sparsity(enabled)

    # ------------------------------------------------------------------ #
    # Coordinate management                                              #
    # ------------------------------------------------------------------ #
    def addCoordinateAxis(self, coordinate: Any) -> None:
        self._base.addCoordinateAxis(coordinate)

    def initialize(self) -> None:
        self._base.initialize()

    def getNumBasisFunctions(self) -> int:
        return self._base.getNumBasisFunctions()

    def getNumCoordinateAxes(self) -> int:
        return self._base.getNumCoordinateAxes()

    def getMonomialDegreeCoordinates(self) -> Mapping[int, int]:
        return self._base.getMonomialDegreeCoordinates()

    # ------------------------------------------------------------------ #
    # Basis evaluation / quadrature                                      #
    # ------------------------------------------------------------------ #
    def evaluate_basis(self, yscalar: float, degree: int) -> Any:
        return self._base.evaluate_basis(yscalar, degree)

    def print_quadrature(self, qmap: Mapping[int, Mapping[str, Any]]) -> None:
        return self._base.print_quadrature(qmap)

    def build_quadrature(self, degrees: Counter) -> Mapping[int, Mapping[str, Any]]:
        return self._base.build_quadrature(degrees)

    def evaluateBasisDegreesY(self, y_by_cid: Mapping[int, float], degrees_counter: Counter) -> Any:
        return self._base.evaluateBasisDegreesY(y_by_cid, degrees_counter)

    def evaluateBasisIndexY(self, y_by_cid: Mapping[int, float], basis_id: int) -> Any:
        return self._base.evaluateBasisIndexY(y_by_cid, basis_id)

    # ------------------------------------------------------------------ #
    # Sparsity helpers                                                   #
    # ------------------------------------------------------------------ #
    def sparse_vector(self, dmapi: Counter, dmapf: Counter) -> bool:
        return self._base.sparse_vector(dmapi, dmapf)

    def monomial_vector_sparsity_mask(self, f_deg: Counter) -> Sequence[int]:
        return self._base.monomial_vector_sparsity_mask(f_deg)

    def polynomial_vector_sparsity_mask(self, f_degrees: Sequence[Counter]) -> Sequence[int]:
        return self._base.polynomial_vector_sparsity_mask(f_degrees)

    def monomial_sparsity_mask(self, f_deg: Counter, symmetric: bool = False) -> Sequence[tuple[int, int]]:
        return self._base.monomial_sparsity_mask(f_deg, symmetric=symmetric)

    def polynomial_sparsity_mask(self, f_degrees: Sequence[Counter], symmetric: bool = False) -> Sequence[tuple[int, int]]:
        return self._base.polynomial_sparsity_mask(f_degrees, symmetric=symmetric)

    # ------------------------------------------------------------------ #
    # Decomposition / Reconstruction                                    #
    # ------------------------------------------------------------------ #
    def _resolve_sparse(self, sparse: bool | None) -> bool | None:
        if sparse is None:
            return self._enabled
        return sparse

    def decompose(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> Dict[int, Any]:
        return self._base.decompose(
            function,
            sparse=self._resolve_sparse(sparse),
            mode=mode,
            analytic=analytic,
        )

    def decompose_matrix(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        return self._base.decompose_matrix(
            function,
            sparse=self._resolve_sparse(sparse),
            symmetric=symmetric,
            mode=mode,
            analytic=analytic,
        )

    def decompose_matrix_analytic(self, function: PolyFunction, sparse: bool | None = None, symmetric: bool = True) -> np.ndarray:
        return self._base.decompose_matrix_analytic(
            function,
            sparse=self._resolve_sparse(sparse),
            symmetric=symmetric,
        )

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
        return self._base.reconstruct(
            function,
            sparse=self._resolve_sparse(sparse),
            mode=mode,
            analytic=analytic,
            precondition=precondition,
            method=method,
            tol=tol,
        )

    # ------------------------------------------------------------------ #
    # Inner products                                                     #
    # ------------------------------------------------------------------ #
    def inner_product(self, f_eval: Any, g_eval: Any, f_deg: Counter | None = None, g_deg: Counter | None = None) -> float:
        return self._base.inner_product(f_eval, g_eval, f_deg=f_deg, g_deg=g_deg)

    def inner_product_basis(
        self,
        i_id: int,
        j_id: int,
        f_eval: Any = None,
        f_deg: Counter | None = None,
    ) -> float:
        return self._base.inner_product_basis(i_id, j_id, f_eval=f_eval, f_deg=f_deg)

    # ------------------------------------------------------------------ #
    # Consistency checks                                                 #
    # ------------------------------------------------------------------ #
    def check_orthonormality(self) -> float:
        return self._base.check_orthonormality()

    def check_decomposition_numerical_symbolic(
        self,
        function: PolyFunction,
        sparse: bool = True,
        tol: float = 1e-10,
        verbose: bool = True,
    ):
        return self._base.check_decomposition_numerical_symbolic(
            function,
            sparse=sparse,
            tol=tol,
            verbose=verbose,
        )

    def check_decomposition_numerical_sparse_full(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self._base.check_decomposition_numerical_sparse_full(function, tol=tol, verbose=verbose)

    def check_decomposition_matrix_sparse_full(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self._base.check_decomposition_matrix_sparse_full(function, tol=tol, verbose=verbose)

    def check_decomposition_matrix_numerical_symbolic(
        self,
        function: PolyFunction,
        tol: float = 1e-12,
        verbose: bool = True,
    ):
        return self._base.check_decomposition_matrix_numerical_symbolic(function, tol=tol, verbose=verbose)

CoordinateSystem = SparsityCoordinateSystem

