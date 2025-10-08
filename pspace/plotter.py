from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .interface import CoordinateSystem as CoordinateSystemInterface
from .core import (
    CoordinateSystem as NumericCoordinateSystem,
    InnerProductMode,
    PolyFunction,
)


class CoordinateSystem(CoordinateSystemInterface):
    """
    Visualization-oriented coordinate system. Delegates all numerical work to
    the underlying numeric CoordinateSystem and records Matplotlib figures for
    consumer inspection.
    """

    def __init__(self, basis_type, verbose: bool = False, autoplot: bool = True):
        super().__init__(basis_type, verbose=verbose)
        self.numeric = NumericCoordinateSystem(basis_type, verbose=verbose)
        self.autoplot = autoplot
        self.last_figure: plt.Figure | None = None

    # expose shared state
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
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    def _update_last_figure(self, fig: plt.Figure | None) -> None:
        if self.autoplot:
            self.last_figure = fig
        else:
            self.last_figure = None if fig is None else fig

    def get_last_figure(self) -> plt.Figure | None:
        return self.last_figure

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
        qmap = self.numeric.build_quadrature(degrees)
        fig = None
        if self.autoplot:
            fig, ax = plt.subplots()
            ys = [node["Y"] for node in qmap.values()]
            coords = list(ys[0].keys()) if ys else []
            if len(coords) == 1:
                xs = [float(val[coords[0]]) for val in ys]
                weights = [float(node["W"]) for node in qmap.values()]
                ax.stem(xs, weights, use_line_collection=True)
                ax.set_xlabel("y")
                ax.set_ylabel("weight")
            elif len(coords) >= 2:
                x_values = [float(val[coords[0]]) for val in ys]
                y_values = [float(val[coords[1]]) for val in ys]
                sc = ax.scatter(
                    x_values,
                    y_values,
                    s=60,
                    c=[float(node["W"]) for node in qmap.values()],
                )
                ax.set_xlabel(f"y{coords[0]}")
                ax.set_ylabel(f"y{coords[1]}")
                fig.colorbar(sc, ax=ax, label="weight")
            ax.set_title("Quadrature nodes")
            fig.tight_layout()
        self._update_last_figure(fig)
        return qmap

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
        coeffs = self.numeric.decompose(function, sparse=sparse, mode=mode, analytic=analytic)
        fig = None
        if self.autoplot:
            fig, ax = plt.subplots()
            bars = sorted(coeffs.items())
            indices = [idx for idx, _ in bars]
            values = [float(val) for _, val in bars]
            ax.bar(indices, values)
            ax.set_title("Coefficient magnitudes")
            ax.set_xlabel("Basis index")
            ax.set_ylabel("Coefficient")
            fig.tight_layout()
        self._update_last_figure(fig)
        return coeffs

    def decompose_matrix(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        matrix = self.numeric.decompose_matrix(
            function, sparse=sparse, symmetric=symmetric, mode=mode, analytic=analytic
        )
        fig = None
        if self.autoplot:
            fig, ax = plt.subplots()
            im = ax.imshow(matrix, cmap="viridis")
            ax.set_title("Matrix decomposition")
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
        self._update_last_figure(fig)
        return matrix

    def decompose_matrix_analytic(self, function: PolyFunction, sparse: bool | None = None, symmetric: bool = True) -> np.ndarray:
        return self.decompose_matrix(function, sparse=sparse, symmetric=symmetric, mode=InnerProductMode.SYMBOLIC)

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
        recon = self.numeric.reconstruct(
            function,
            sparse=sparse,
            mode=mode,
            analytic=analytic,
            precondition=precondition,
            method=method,
            tol=tol,
        )
        fig = None
        if self.autoplot:
            fig, ax = plt.subplots()
            coeffs = [term[0] for term in recon.terms]
            ax.bar(range(len(coeffs)), coeffs)
            ax.set_title("Reconstructed φ coefficients")
            ax.set_xlabel("Term")
            ax.set_ylabel("Coefficient")
            fig.tight_layout()
        self._update_last_figure(fig)
        return recon

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
