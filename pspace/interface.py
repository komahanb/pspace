from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .core import InnerProductMode, PolyFunction


class CoordinateSystem(ABC):
    """Abstract interface mirrored by numeric, plotting, and profiling layers."""

    def __init__(self, basis_type, verbose: bool = False) -> None:
        self.basis_construction = basis_type
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Coordinate management
    # ------------------------------------------------------------------
    @abstractmethod
    def addCoordinateAxis(self, coordinate: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def getNumBasisFunctions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def getNumCoordinateAxes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def getMonomialDegreeCoordinates(self) -> Mapping[int, int]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Basis evaluation / quadrature
    # ------------------------------------------------------------------
    def evaluate_basis(self, yscalar: float, degree: int) -> Any:
        raise NotImplementedError

    def print_quadrature(self, qmap: Mapping[int, Mapping[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_quadrature(self, degrees: Counter) -> Mapping[int, Mapping[str, Any]]:
        raise NotImplementedError

    def evaluateBasisDegreesY(self, y_by_cid: Mapping[int, float], degrees_counter: Counter) -> Any:
        raise NotImplementedError

    def evaluateBasisIndexY(self, y_by_cid: Mapping[int, float], basis_id: int) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sparsity helpers
    # ------------------------------------------------------------------
    def sparse_vector(self, dmapi: Counter, dmapf: Counter) -> bool:
        raise NotImplementedError

    def monomial_vector_sparsity_mask(self, f_deg: Counter) -> Iterable[int]:
        raise NotImplementedError

    def polynomial_vector_sparsity_mask(self, f_degrees: Sequence[Counter]) -> Iterable[int]:
        raise NotImplementedError

    def monomial_sparsity_mask(self, f_deg: Counter, symmetric: bool = False) -> Iterable[tuple[int, int]]:
        raise NotImplementedError

    def polynomial_sparsity_mask(self, f_degrees: Sequence[Counter], symmetric: bool = False) -> Iterable[tuple[int, int]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------
    def inner_product(self, f_eval: Any, g_eval: Any, f_deg: Counter | None = None, g_deg: Counter | None = None) -> float:
        raise NotImplementedError

    def inner_product_basis(
        self,
        i_id: int,
        j_id: int,
        f_eval: Any = None,
        f_deg: Counter | None = None,
    ) -> float:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Decomposition / Reconstruction
    # ------------------------------------------------------------------
    @abstractmethod
    def decompose(
        self,
        function: "PolyFunction",
        sparse: bool = True,
        mode: "InnerProductMode" | str | None = None,
        analytic: bool = False,
    ) -> Dict[int, Any]:
        raise NotImplementedError

    @abstractmethod
    def decompose_matrix(
        self,
        function: "PolyFunction",
        sparse: bool = False,
        symmetric: bool = True,
        mode: "InnerProductMode" | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    def decompose_matrix_analytic(self, function: "PolyFunction", sparse: bool = False, symmetric: bool = True) -> np.ndarray:
        raise NotImplementedError

    def reconstruct(
        self,
        function: "PolyFunction",
        sparse: bool = True,
        mode: "InnerProductMode" | str | None = None,
        analytic: bool = False,
        precondition: bool = True,
        method: str = "cholesky",
        tol: float = 0.0,
    ) -> "PolyFunction":
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def check_orthonormality(self) -> float:
        raise NotImplementedError

    def check_decomposition_numerical_symbolic(
        self,
        function: "PolyFunction",
        sparse: bool = True,
        tol: float = 1e-10,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError

    def check_decomposition_numerical_sparse_full(
        self,
        function: "PolyFunction",
        tol: float = 1e-12,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError

    def check_decomposition_matrix_sparse_full(
        self,
        function: "PolyFunction",
        tol: float = 1e-12,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError

    def check_decomposition_matrix_numerical_symbolic(
        self,
        function: "PolyFunction",
        tol: float = 1e-12,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError
