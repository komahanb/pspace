from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from .core import InnerProductMode, PolyFunction
from .interface import CoordinateSystem as CoordinateSystemInterface


class ParallelPolicy(ABC):
    """
    Strategy object describing how to execute vector/matrix assembly in parallel.

    Policies may coordinate threads, processes, or distributed workers. The base
    implementation simply calls the provided thunk, but concrete policies can
    intercept the call, distribute work, and aggregate results.
    """

    name: str = "sequential"

    def setup(self, coordinate_system: CoordinateSystemInterface) -> None:  # pragma: no cover - hook
        """Opportunity for a policy to inspect the wrapped coordinate system."""

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        return thunk()

    def execute_matrix(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        symmetric: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], np.ndarray],
    ) -> np.ndarray:
        return thunk()


class DistributedBasisParallel(ParallelPolicy):
    """
    Proof-of-concept policy that *pretends* to split work by basis blocks.

    In a real implementation, this would coordinate rank/worker assignment. Here
    we simply record metadata to make comparison/testing easier.
    """

    name = "distributed_basis"

    def __init__(self, chunks: int = 4) -> None:
        self.chunks = max(1, chunks)
        self.last_basis_partitions: list[tuple[int, int]] = []

    def _plan(self, coordinate_system: CoordinateSystemInterface) -> None:
        nbasis = coordinate_system.getNumBasisFunctions()
        size = max(1, nbasis // self.chunks)
        partitions = []
        start = 0
        while start < nbasis:
            end = min(nbasis, start + size)
            partitions.append((start, end))
            start = end
        self.last_basis_partitions = partitions

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        self._plan(coordinate_system)
        return thunk()

    def execute_matrix(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        symmetric: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], np.ndarray],
    ) -> np.ndarray:
        self._plan(coordinate_system)
        return thunk()


class DistributedCoordinateParallel(ParallelPolicy):
    """
    Proof-of-concept policy that *pretends* to split work by coordinates/axes.

    Each axis is treated as an independent partition. Real implementations could
    fan out over MPI ranks or thread pools accordingly.
    """

    name = "distributed_coordinate"

    def __init__(self) -> None:
        self.last_coordinate_partitions: list[int] = []

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        self.last_coordinate_partitions = list(coordinate_system.coordinates.keys())
        return thunk()

    def execute_matrix(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        symmetric: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], np.ndarray],
    ) -> np.ndarray:
        self.last_coordinate_partitions = list(coordinate_system.coordinates.keys())
        return thunk()


class ParallelCoordinateSystem(CoordinateSystemInterface):
    """
    Decorator that routes decomposition/reconstruction through a parallel policy.

    Any CoordinateSystem instance can be wrapped; policies can be swapped at
    runtime to compare execution strategies.
    """

    def __init__(
        self,
        base: CoordinateSystemInterface,
        policy: ParallelPolicy | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(base.basis_construction, verbose=verbose)
        self._base = base
        self._policy = policy or ParallelPolicy()
        self._policy.setup(self._base)

    # ------------------------------------------------------------------ #
    # Shared state                                                       #
    # ------------------------------------------------------------------ #
    @property
    def policy(self) -> ParallelPolicy:
        return self._policy

    def configure_policy(self, policy: ParallelPolicy) -> None:
        self._policy = policy
        self._policy.setup(self._base)

    @property
    def coordinates(self):
        return self._base.coordinates

    @property
    def basis(self):
        return self._base.basis

    @property
    def sparsity_enabled(self) -> bool:
        return getattr(self._base, "sparsity_enabled", True)

    def configure_sparsity(self, enabled: bool) -> None:
        if hasattr(self._base, "configure_sparsity"):
            self._base.configure_sparsity(enabled)

    # ------------------------------------------------------------------ #
    # Delegate remaining interface                                       #
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
    # Parallelised operations                                            #
    # ------------------------------------------------------------------ #
    def decompose(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> Dict[int, Any]:
        sflag = self.sparsity_enabled if sparse is None else sparse

        def thunk():
            return self._base.decompose(function, sparse=sflag, mode=mode, analytic=analytic)

        return self._policy.execute_vector(self._base, function, sflag, mode, analytic, thunk)

    def decompose_matrix(
        self,
        function: PolyFunction,
        sparse: bool | None = None,
        symmetric: bool = True,
        mode: InnerProductMode | str | None = None,
        analytic: bool = False,
    ) -> np.ndarray:
        sflag = self.sparsity_enabled if sparse is None else sparse

        def thunk():
            return self._base.decompose_matrix(
                function,
                sparse=sflag,
                symmetric=symmetric,
                mode=mode,
                analytic=analytic,
            )

        return self._policy.execute_matrix(self._base, function, sflag, symmetric, mode, analytic, thunk)

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
        sflag = self.sparsity_enabled if sparse is None else sparse
        return self._base.reconstruct(
            function,
            sparse=sflag,
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
        return self._base.check_orthonormality()

    def check_decomposition_numerical_symbolic(
        self,
        function: PolyFunction,
        sparse: bool = True,
        tol: float = 1e-10,
        verbose: bool = True,
    ):
        return self._base.check_decomposition_numerical_symbolic(function, sparse=sparse, tol=tol, verbose=verbose)

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
