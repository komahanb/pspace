from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .core import InnerProductMode, PolyFunction, OrthoPolyFunction


class CoordinateSystem(ABC):
    """Abstract interface mirrored by numeric, plotting, and profiling layers."""

    def __init__(self, basis_type, verbose: bool = False) -> None:
        self.basis_construction = basis_type
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Coordinate management
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def coordinates(self) -> Mapping[int, Any]:
        raise NotImplementedError

    @property
    @abstractmethod
    def basis(self) -> Mapping[int, Counter]:
        raise NotImplementedError

    @property
    @abstractmethod
    def sparsity_enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def configure_sparsity(self, enabled: bool) -> None:
        raise NotImplementedError

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
    @abstractmethod
    def evaluate_basis(self, yscalar: float, degree: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def print_quadrature(self, qmap: Mapping[int, Mapping[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_quadrature(self, degrees: Counter) -> Mapping[int, Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def evaluateBasisDegreesY(self, y_by_cid: Mapping[int, float], degrees_counter: Counter) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluateBasisIndexY(self, y_by_cid: Mapping[int, float], basis_id: int) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sparsity helpers
    # ------------------------------------------------------------------
    @abstractmethod
    def sparse_vector(self, dmapi: Counter, dmapf: Counter) -> bool:
        raise NotImplementedError

    @abstractmethod
    def monomial_vector_sparsity_mask(self, f_deg: Counter) -> Iterable[int]:
        raise NotImplementedError

    @abstractmethod
    def polynomial_vector_sparsity_mask(self, f_degrees: Sequence[Counter]) -> Iterable[int]:
        raise NotImplementedError

    @abstractmethod
    def monomial_sparsity_mask(self, f_deg: Counter, symmetric: bool = False) -> Iterable[tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def polynomial_sparsity_mask(self, f_degrees: Sequence[Counter], symmetric: bool = False) -> Iterable[tuple[int, int]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sparsity helpers (default implementations)
    # ------------------------------------------------------------------
    def vector_mask(
        self,
        function: "PolyFunction",
        sparse: bool,
        *,
        sort: bool = False,
    ) -> list[int]:
        self._ensure_function_bound(function)
        if sparse:
            mask_iter = self.polynomial_vector_sparsity_mask(function.degrees)
        else:
            mask_iter = self.basis.keys()
        indices = [int(idx) for idx in mask_iter]
        if sort:
            indices.sort()
        return indices

    def matrix_mask(
        self,
        function: "PolyFunction",
        sparse: bool,
        symmetric: bool,
        *,
        sort: bool = False,
    ) -> list[tuple[int, int]]:
        self._ensure_function_bound(function)
        if sparse:
            mask_iter = list(self.polynomial_sparsity_mask(function.degrees, symmetric=symmetric))
        else:
            basis_ids = list(self.basis.keys())
            if symmetric:
                mask_iter = [(i, j) for ii, i in enumerate(basis_ids) for j in basis_ids[ii:]]
            else:
                mask_iter = [(i, j) for i in basis_ids for j in basis_ids]
        pairs = [(int(i), int(j)) for i, j in mask_iter]
        if sort:
            pairs.sort()
        return pairs

    def compute_vector_coefficients(
        self,
        function: "PolyFunction",
        indices: Sequence[int],
        *,
        bind_coordinates: bool = True,
    ) -> Dict[int, float]:
        if bind_coordinates:
            self._ensure_function_bound(function)
        coeffs: Dict[int, float] = {}
        max_degrees = getattr(function, "max_degrees", None)
        for idx in indices:
            psi = self.basis[idx]

            def psi_eval(y, basis_deg=psi):
                return self.evaluateBasisDegreesY(y, basis_deg)

            coeff = self.inner_product(
                function,
                psi_eval,
                f_deg=max_degrees,
                g_deg=psi,
            )
            coeffs[int(idx)] = float(coeff)
        return coeffs

    def compute_matrix_entries(
        self,
        function: Any,
        pairs: Sequence[tuple[int, int]],
        symmetric: bool,
        *,
        bind_coordinates: bool = True,
    ) -> Dict[tuple[int, int], float]:
        if bind_coordinates:
            self._ensure_function_bound(function)
        max_degrees = getattr(function, "max_degrees", None)
        partial: Dict[tuple[int, int], float] = {}
        for i, j in pairs:
            value = self.inner_product_basis(
                i,
                j,
                f_eval=function,
                f_deg=max_degrees,
            )
            sval = float(value)
            partial[(int(i), int(j))] = sval
            if symmetric and i != j:
                partial[(int(j), int(i))] = sval
        return partial

    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------
    @abstractmethod
    def inner_product(self, f_eval: Any, g_eval: Any, f_deg: Counter | None = None, g_deg: Counter | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
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
        sparse: bool | None = None,
        mode: "InnerProductMode" | str | None = None,
        analytic: bool = False,
    ) -> Dict[int, Any]:
        raise NotImplementedError

    @abstractmethod
    def decompose_matrix(
        self,
        function: "PolyFunction",
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: "InnerProductMode" | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decompose_matrix_analytic(self, function: "PolyFunction", sparse: bool = False, symmetric: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
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
    @abstractmethod
    def check_orthonormality(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def check_decomposition_numerical_symbolic(
        self,
        function: "PolyFunction",
        sparse: bool = True,
        tol: float = 1e-10,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError

    @abstractmethod
    def check_decomposition_numerical_sparse_full(
        self,
        function: "PolyFunction",
        tol: float = 1e-12,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError

    @abstractmethod
    def check_decomposition_matrix_sparse_full(
        self,
        function: "PolyFunction",
        tol: float = 1e-12,
        verbose: bool = True,
    ) -> tuple[bool, Dict[Any, tuple[float, float, float]]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _ensure_function_bound(self, function: Any) -> None:
        if hasattr(function, "bind_coordinates"):
            coords = getattr(function, "_coords", None)
            if coords is not self.coordinates:
                function.bind_coordinates(self.coordinates)


class MonomialCoordinateSystemMixin(ABC):
    """
    Coordinate-system mixin for monomial (non-orthonormal) polynomial spaces.
    """

    @abstractmethod
    def decompose(
        self,
        function: "PolyFunction",
        sparse: bool | None = True,
        mode: "InnerProductMode | str | None" = None,
        analytic: bool = False,
        **kwargs: Any,
    ) -> Dict[int, float]:
        raise NotImplementedError

    @abstractmethod
    def decompose_matrix(
        self,
        function: "PolyFunction",
        sparse: bool | None = False,
        symmetric: bool = True,
        mode: "InnerProductMode | str | None" = None,
        analytic: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(
        self,
        function: "PolyFunction",
        sparse: bool | None = True,
        mode: "InnerProductMode | str | None" = None,
        analytic: bool = False,
        precondition: bool = True,
        **kwargs: Any,
    ) -> "PolyFunction":
        raise NotImplementedError


class OrthonormalCoordinateSystemMixin(ABC):
    """
    Coordinate-system mixin for orthonormal polynomial spaces.
    """

    @abstractmethod
    def decompose(
        self,
        function: "OrthoPolyFunction",
        sparse: bool | None = True,
        mode: "InnerProductMode | str | None" = None,
        analytic: bool = False,
        **kwargs: Any,
    ) -> Dict[int, float]:
        raise NotImplementedError

    @abstractmethod
    def decompose_matrix(
        self,
        function: "OrthoPolyFunction",
        sparse: bool | None = False,
        symmetric: bool = True,
        mode: "InnerProductMode | str | None" = None,
        analytic: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(
        self,
        function: "OrthoPolyFunction",
        sparse: bool | None = True,
        mode: "InnerProductMode | str | None" = None,
        analytic: bool = False,
        precondition: bool = True,
        **kwargs: Any,
    ) -> "OrthoPolyFunction":
        raise NotImplementedError
