from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .interface import CoordinateSystem, OrthonormalCoordinateSystemMixin
from .numeric import NumericCoordinateSystem, OrthoPolyFunction, PolyFunction, InnerProductMode


class OrthonormalCoordinateSystem(OrthonormalCoordinateSystemMixin, CoordinateSystem):
    """Coordinate system operating in the orthonormal polynomial basis."""

    def __init__(self, basis_type, verbose: bool = False, numeric: NumericCoordinateSystem | None = None) -> None:
        super().__init__(basis_type, verbose=verbose)
        self.numeric = numeric or NumericCoordinateSystem(basis_type, verbose=verbose)

    # ------------------------------------------------------------------ #
    # Shared state delegation                                             #
    # ------------------------------------------------------------------ #
    @property
    def coordinates(self) -> Mapping[int, Any]:
        return self.numeric.coordinates

    @property
    def basis(self) -> Mapping[int, Any]:
        return self.numeric.basis

    @property
    def sparsity_enabled(self) -> bool:
        return self.numeric.sparsity_enabled

    def configure_sparsity(self, enabled: bool) -> None:
        self.numeric.configure_sparsity(enabled)

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

    def evaluate_basis(self, yscalar: float, degree: int) -> Any:
        return self.numeric.evaluate_basis(yscalar, degree)

    def print_quadrature(self, qmap: Mapping[int, Mapping[str, Any]]) -> None:
        self.numeric.print_quadrature(qmap)

    def build_quadrature(self, degrees: Mapping[int, int] | Sequence[int]) -> Mapping[int, Mapping[str, Any]]:
        return self.numeric.build_quadrature(degrees)  # type: ignore[arg-type]

    def evaluateBasisDegreesY(self, y_by_cid: Mapping[int, float], degrees_counter: Mapping[int, int]) -> Any:
        return self.numeric.evaluateBasisDegreesY(y_by_cid, degrees_counter)

    def evaluateBasisIndexY(self, y_by_cid: Mapping[int, float], basis_id: int) -> Any:
        return self.numeric.evaluateBasisIndexY(y_by_cid, basis_id)

    def sparse_vector(self, dmapi: Mapping[int, int], dmapf: Mapping[int, int]) -> bool:
        return self.numeric.sparse_vector(dmapi, dmapf)

    def monomial_vector_sparsity_mask(self, f_deg: Mapping[int, int]) -> Sequence[int]:
        return self.numeric.monomial_vector_sparsity_mask(f_deg)  # type: ignore[arg-type]

    def polynomial_vector_sparsity_mask(self, f_degrees: Sequence[Mapping[int, int]]) -> Sequence[int]:
        return self.numeric.polynomial_vector_sparsity_mask(f_degrees)  # type: ignore[arg-type]

    def monomial_sparsity_mask(self, f_deg: Mapping[int, int], symmetric: bool = False) -> Sequence[tuple[int, int]]:
        return self.numeric.monomial_sparsity_mask(f_deg, symmetric=symmetric)  # type: ignore[arg-type]

    def polynomial_sparsity_mask(self, f_degrees: Sequence[Mapping[int, int]], symmetric: bool = False) -> Sequence[tuple[int, int]]:
        return self.numeric.polynomial_sparsity_mask(f_degrees, symmetric=symmetric)  # type: ignore[arg-type]

    def inner_product(
        self,
        f_eval: Any,
        g_eval: Any,
        f_deg: Mapping[int, int] | None = None,
        g_deg: Mapping[int, int] | None = None,
    ) -> float:
        return self.numeric.inner_product(f_eval, g_eval, f_deg=f_deg, g_deg=g_deg)  # type: ignore[arg-type]

    def inner_product_basis(
        self,
        i_id: int,
        j_id: int,
        f_eval: Any = None,
        f_deg: Mapping[int, int] | None = None,
    ) -> float:
        return self.numeric.inner_product_basis(i_id, j_id, f_eval=f_eval, f_deg=f_deg)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Conversions                                                        #
    # ------------------------------------------------------------------ #
    def to_orthopoly(
        self,
        poly: PolyFunction,
        *,
        sparse: bool | None = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> OrthoPolyFunction:
        coeffs = self.numeric.decompose(poly, sparse=sparse, mode=mode, analytic=analytic)
        terms = [
            (float(coeffs.get(idx, 0.0)), self.numeric.basis[idx])
            for idx in self.numeric.basis
        ]
        return OrthoPolyFunction(terms, self.numeric.coordinates)

    def from_orthopoly(self, ortho: OrthoPolyFunction) -> PolyFunction:
        return ortho.toPolyFunction()

    # ------------------------------------------------------------------ #
    # Orthonormal-space API                                             #
    # ------------------------------------------------------------------ #
    def decompose(
        self,
        function: OrthoPolyFunction,
        sparse: bool | None = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
        **kwargs,
    ) -> Dict[int, float]:
        basis_lookup = {
            tuple(sorted(counter.items())): idx
            for idx, counter in self.numeric.basis.items()
        }
        coeffs: Dict[int, float] = {}
        for coeff, degs in function._terms:  # type: ignore[attr-defined]
            key = tuple(sorted(degs.items()))
            idx = basis_lookup.get(key)
            if idx is None:
                continue
            coeffs[int(idx)] = float(coeff)

        if sparse:
            for idx in self.numeric.basis:
                coeffs.setdefault(idx, 0.0)

        return coeffs

    def decompose_matrix(
        self,
        function: OrthoPolyFunction,
        sparse: bool | None = False,
        symmetric: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
        **kwargs,
    ):
        poly = self.from_orthopoly(function)
        return self.numeric.decompose_matrix(
            poly,
            sparse=sparse,
            symmetric=symmetric,
            mode=mode,
            analytic=analytic,
            **kwargs,
        )

    def reconstruct(
        self,
        function: OrthoPolyFunction,
        sparse: bool | None = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
        precondition: bool = True,
        **kwargs,
    ) -> OrthoPolyFunction:
        # Reconstruction in orthonormal coordinates is the identity mapping.
        return OrthoPolyFunction(list(function._terms), self.numeric.coordinates)  # type: ignore[attr-defined]

    def decompose_matrix_analytic(
        self,
        function: OrthoPolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
    ):
        poly = self.from_orthopoly(function)
        return self.numeric.decompose_matrix_analytic(poly, sparse=sparse, symmetric=symmetric)

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


__all__ = ["OrthonormalCoordinateSystem"]
