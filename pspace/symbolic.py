from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .interface import CoordinateAxis, CoordinateSystem, MonomialCoordinateSystemMixin
from .core import DistributionType, InnerProductMode, PolyFunction
from .numeric import NumericCoordinateSystem
from .distributions import SYMBOLIC_DISTRIBUTIONS, SymbolicDistribution


class SymbolicCoordinateAxis(CoordinateAxis):
    def __init__(
        self,
        *,
        coord_id: int,
        name: str,
        coord_type: Any,
        degree: int,
        distribution: SymbolicDistribution,
    ) -> None:
        super().__init__(
            coord_id     = coord_id,
            name         = name,
            coord_type   = coord_type,
            distribution = distribution.kind,
            degree       = degree,
            dist_coords  = distribution.params,
        )
        self._distribution = distribution
        self.symbol        = sp.Symbol(self.name)

    def domain(self):
        return self._distribution.domain()

    def weight(self) -> sp.Expr:
        return self._distribution.weight(self.symbol)

    def physical_to_standard(self, yscalar: Any) -> sp.Expr:
        return self._distribution.physical_to_standard(sp.sympify(yscalar))

    def quadrature_to_physical(self, xscalar: Any) -> sp.Expr:
        return self._distribution.quadrature_to_physical(sp.sympify(xscalar))

    def standard_to_physical(self, zscalar: Any) -> sp.Expr:
        return self._distribution.standard_to_physical(sp.sympify(zscalar))

    def psi_z(self, zscalar: Any, degree: int) -> sp.Expr:
        return self._distribution.psi_z(sp.sympify(zscalar), degree)

    def gaussian_quadrature(self, degree: int):
        return self._distribution.gaussian_quadrature(degree)


class SymbolicVectorInnerProductOperator:
    """Rank-1 inner products evaluated via symbolic integration."""

    def __init__(self, coordinate_system: "CoordinateSystem") -> None:
        self.cs = coordinate_system

    def compute(self, function: PolyFunction, sparse: bool = True) -> dict[int, float]:
        cs = self.cs
        coords = cs.coordinates
        function.bind_coordinates(coords)

        symbols = {cid: coord.symbol for cid, coord in coords.items()}
        weight  = sp.Integer(1)
        for cid, coord in coords.items():
            weight *= coord.weight()
        f_expr = function(symbols)

        if sparse:
            mask = cs.polynomial_vector_sparsity_mask(function.degrees)
        else:
            mask = cs.basis.keys()

        coeffs: dict[int, float] = {}
        for k in mask:
            psi_expr = sp.Integer(1)
            for cid, deg in cs.basis[k].items():
                coord = coords[cid]
                z = coord.physical_to_standard(symbols[cid])
                psi_expr *= coord.psi_z(z, deg)

            integrand = sp.simplify(f_expr * psi_expr * weight)
            val = integrand
            for cid, coord in coords.items():
                y = symbols[cid]
                a, b = coord.domain()
                val = sp.integrate(val, (y, a, b))

            coeffs[k] = float(sp.simplify(val))

        if sparse:
            for k in cs.basis:
                coeffs.setdefault(k, 0.0)

        return coeffs


class SymbolicMatrixInnerProductOperator:
    """Rank-2 inner products evaluated via symbolic integration."""

    def __init__(self, coordinate_system: "CoordinateSystem") -> None:
        self.cs = coordinate_system

    def compute(
        self,
        function: PolyFunction,
        sparse: bool = False,
        symmetric: bool = True,
    ) -> np.ndarray:
        cs = self.cs
        coords = cs.coordinates

        if hasattr(function, "bind_coordinates"):
            function.bind_coordinates(coords)

        nbasis = cs.getNumBasisFunctions()
        matrix = np.zeros((nbasis, nbasis))

        if sparse:
            mask = cs.polynomial_sparsity_mask(function.degrees, symmetric=symmetric)
        else:
            if symmetric:
                mask = {(i, j) for i in cs.basis for j in cs.basis if i <= j}
            else:
                mask = {(i, j) for i in cs.basis for j in cs.basis}

        symbols = {cid: coord.symbol for cid, coord in coords.items()}
        weight  = sp.Integer(1)
        for cid, coord in coords.items():
            weight *= coord.weight()

        f_expr = sp.Integer(0)
        for coeff, degs in function.terms:
            term = sp.sympify(coeff)
            for cid, deg in degs.items():
                term *= symbols[cid] ** deg
            f_expr += term

        psi_cache: dict[int, sp.Expr] = {}

        def psi_expr(index: int) -> sp.Expr:
            if index not in psi_cache:
                expr = sp.Integer(1)
                for cid, deg in cs.basis[index].items():
                    coord = coords[cid]
                    z     = coord.physical_to_standard(symbols[cid])
                    expr *= coord.psi_z(z, deg)
                psi_cache[index] = sp.simplify(expr)
            return psi_cache[index]

        for i, j in mask:
            integrand = sp.simplify(f_expr * psi_expr(i) * psi_expr(j) * weight)
            val = integrand
            for cid, coord in coords.items():
                y = symbols[cid]
                a, b = coord.domain()
                val = sp.integrate(val, (y, a, b))

            entry = float(sp.simplify(val))
            matrix[i, j] = entry
            if symmetric and i != j:
                matrix[j, i] = entry

        return matrix


class SymbolicCoordinateSystem(CoordinateSystem, MonomialCoordinateSystemMixin):
    """
    Symbolic mirror of the CoordinateSystem contract. Reuses the numeric core for
    structural information while evaluating decompositions with SymPy.
    """

    def __init__(
        self,
        basis_type,
        verbose: bool = False,
        numeric: NumericCoordinateSystem | None = None,
    ) -> None:
        super().__init__(basis_type, verbose=verbose)
        self.numeric = numeric or NumericCoordinateSystem(basis_type, verbose=verbose)
        self._vector_symbolic = SymbolicVectorInnerProductOperator(self)
        self._matrix_symbolic = SymbolicMatrixInnerProductOperator(self)
        self._symbolic_coords: dict[int, SymbolicCoordinateAxis] | None = None

    # ------------------------------------------------------------------ #
    # Shared state                                                       #
    # ------------------------------------------------------------------ #
    @property
    def coordinates(self):
        if self._symbolic_coords is None or len(self._symbolic_coords) != len(self.numeric.coordinates):
            self._symbolic_coords = {
                cid: SymbolicCoordinateAxis(
                    coord_id     = coord.id,
                    name         = coord.name,
                    coord_type   = coord.type,
                    degree       = coord.degree,
                    distribution = SYMBOLIC_DISTRIBUTIONS[coord.distribution](**coord.dist_coords),
                )
                for cid, coord in self.numeric.coordinates.items()
            }
        return self._symbolic_coords

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
        self._symbolic_coords = None

    def initialize(self) -> None:
        self.numeric.initialize()
        self._symbolic_coords = None

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
    def decompose(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        mode: InnerProductMode | None = None,
        analytic: bool = False,
    ) -> dict[int, float]:
        if sparse is None:
            sparse = self.numeric.sparsity_enabled

        if mode is None:
            mode = InnerProductMode.SYMBOLIC
        elif not isinstance(mode, InnerProductMode):
            raise TypeError(f"mode must be an InnerProductMode or None, got {mode!r}")

        if mode is not InnerProductMode.SYMBOLIC:
            raise ValueError("Symbolic CoordinateSystem supports only symbolic decomposition.")
        return self._vector_symbolic.compute(function, sparse=sparse)

    def decompose_matrix(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: InnerProductMode | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        if sparse is None:
            sparse = self.numeric.sparsity_enabled

        if mode is None:
            mode = InnerProductMode.SYMBOLIC
        elif not isinstance(mode, InnerProductMode):
            raise TypeError(f"mode must be an InnerProductMode or None, got {mode!r}")

        if mode is not InnerProductMode.SYMBOLIC:
            raise ValueError("Symbolic CoordinateSystem supports only symbolic decomposition.")
        return self._matrix_symbolic.compute(function, sparse=sparse, symmetric=symmetric)

    def decompose_matrix_analytic(self, function: PolyFunction, sparse: bool | None = None, symmetric: bool = True) -> np.ndarray:
        if sparse is None:
            sparse = self.numeric.sparsity_enabled
        return self._matrix_symbolic.compute(function, sparse=sparse, symmetric=symmetric)

    def reconstruct(
        self,
        function: PolyFunction,
        sparse: bool = True,
        mode: InnerProductMode | None = None,
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

CoordinateSystem = SymbolicCoordinateSystem
