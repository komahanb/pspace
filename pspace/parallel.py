from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import math
import os
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

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
    category: str = "generic"

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

    # ------------------------------------------------------------------ #
    # Composition helpers                                               #
    # ------------------------------------------------------------------ #
    def compose(self, *others: "ParallelPolicy | Sequence[ParallelPolicy]") -> "ParallelPolicy":
        """
        Create a composite policy with this policy followed by ``others``.

        Examples
        --------
        >>> mpi = MPIParallelPolicy(world_size=4)
        >>> shared = SharedMemoryParallelPolicy(backend="openmp", workers=8)
        >>> hybrid = mpi.compose(shared)
        """
        return compose_parallel_policies(self, *others)

    def __or__(self, other: "ParallelPolicy | Sequence[ParallelPolicy]") -> "ParallelPolicy":
        """Shorthand for ``self.compose(other)``."""
        return self.compose(other)

    def describe(self) -> Dict[str, Any]:
        """Return a machine-readable description of the policy."""
        return {"name": self.name, "category": self.category}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        params = []
        for key, value in vars(self).items():
            if key.startswith("_"):
                continue
            params.append(f"{key}={value!r}")
        params_str = ", ".join(params)
        return f"{self.__class__.__name__}({params_str})"


class DistributedBasisParallel(ParallelPolicy):
    """
    Proof-of-concept policy that *pretends* to split work by basis blocks.

    In a real implementation, this would coordinate rank/worker assignment. Here
    we simply record metadata to make comparison/testing easier.
    """

    name = "distributed_basis"
    category = "distributed"

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
    category = "distributed"

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


class CompositeParallelPolicy(ParallelPolicy):
    """Compose multiple policies so each can reflect work before execution."""

    name = "composite"

    def __init__(self, policies: Iterable[ParallelPolicy]) -> None:
        flattened: list[ParallelPolicy] = []
        for policy in policies:
            if isinstance(policy, CompositeParallelPolicy):
                flattened.extend(policy.policies)
            elif isinstance(policy, ParallelPolicy):
                flattened.append(policy)
            else:  # pragma: no cover - defensive
                raise TypeError(f"Unsupported policy type: {type(policy)!r}")
        if not flattened:
            raise ValueError("CompositeParallelPolicy requires at least one policy.")
        self.policies: tuple[ParallelPolicy, ...] = tuple(flattened)
        self.category = "+".join(sorted({policy.category for policy in self.policies}))

    @property
    def name(self) -> str:
        return "+".join(policy.name for policy in self.policies)

    def setup(self, coordinate_system: CoordinateSystemInterface) -> None:
        for policy in self.policies:
            policy.setup(coordinate_system)

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        def invoke(idx: int) -> Dict[int, Any]:
            if idx >= len(self.policies):
                return thunk()
            policy = self.policies[idx]
            return policy.execute_vector(
                coordinate_system,
                function,
                sparse,
                mode,
                analytic,
                lambda: invoke(idx + 1),
            )

        return invoke(0)

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
        def invoke(idx: int) -> np.ndarray:
            if idx >= len(self.policies):
                return thunk()
            policy = self.policies[idx]
            return policy.execute_matrix(
                coordinate_system,
                function,
                sparse,
                symmetric,
                mode,
                analytic,
                lambda: invoke(idx + 1),
            )

        return invoke(0)


class MPIParallelPolicy(ParallelPolicy):
    """
    Reflection describing an MPI-based distributed execution strategy.

    This policy records a notional partition of the basis functions across MPI
    ranks so higher-level components can reason about workload decomposition.
    """

    name = "distributed.mpi"
    category = "distributed"

    def __init__(
        self,
        world_size: int | None = None,
        ranks: Sequence[int] | None = None,
        communicator: str = "MPI_COMM_WORLD",
    ) -> None:
        self.communicator = communicator
        self.world_size = max(1, world_size) if world_size else None
        self.ranks = list(ranks) if ranks else None
        self.last_plan: Dict[str, Any] | None = None

    def setup(self, coordinate_system: CoordinateSystemInterface) -> None:
        nbasis = coordinate_system.getNumBasisFunctions()
        if not self.world_size:
            if self.ranks:
                self.world_size = len(self.ranks)
            else:
                # Default to min(nbasis, cpu count) to keep plans realistic.
                default_world = min(nbasis or 1, os.cpu_count() or 1)
                self.world_size = max(1, default_world)
        if not self.ranks:
            self.ranks = list(range(self.world_size))

    def _partition(self, total_items: int) -> list[Dict[str, Any]]:
        if not total_items:
            return []
        assert self.world_size is not None  # for type checkers
        ranks = self.ranks or list(range(self.world_size))
        chunk = max(1, math.ceil(total_items / len(ranks)))
        partitions: list[Dict[str, Any]] = []
        start = 0
        for rank in ranks:
            end = min(total_items, start + chunk)
            partitions.append({"rank": rank, "start": start, "stop": end})
            start = end
            if start >= total_items:
                break
        return partitions

    def _record_plan(
        self,
        operation: str,
        coordinate_system: CoordinateSystemInterface,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        symmetric: bool | None = None,
    ) -> None:
        nbasis = coordinate_system.getNumBasisFunctions()
        self.last_plan = {
            "operation": operation,
            "sparse": bool(sparse),
            "mode": str(mode) if mode is not None else None,
            "analytic": bool(analytic),
            "symmetric": symmetric,
            "partitions": self._partition(nbasis),
            "communicator": self.communicator,
            "world_size": self.world_size,
        }

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        self._record_plan("vector", coordinate_system, sparse, mode, analytic)
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
        self._record_plan("matrix", coordinate_system, sparse, mode, analytic, symmetric=symmetric)
        return thunk()


class SharedMemoryParallelPolicy(ParallelPolicy):
    """
    Reflection representing shared-memory parallelism (OpenMP, CUDA, CuPy, etc.).

    The policy keeps track of the backend, workers/devices, and the scheduling
    strategy used to cover the basis functions inside one shared-memory node.
    """

    name = "shared"
    category = "shared"

    def __init__(
        self,
        backend: str = "threadpool",
        workers: int | None = None,
        device: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.backend = backend
        self.workers = max(1, workers) if workers else None
        self.device = device
        self.chunk_size = chunk_size
        self.last_schedule: Dict[str, Any] | None = None

    def setup(self, coordinate_system: CoordinateSystemInterface) -> None:
        if self.workers is None and self.backend.lower() not in {"cuda", "cupy"}:
            self.workers = max(1, os.cpu_count() or 1)
        if self.backend.lower() in {"cuda", "cupy"} and self.device is None:
            self.device = "cuda:0"

    def _schedule(self, total_items: int) -> list[Dict[str, Any]]:
        if total_items == 0:
            return []
        workers = self.workers or 1
        chunk = self.chunk_size or max(1, math.ceil(total_items / workers))
        schedule: list[Dict[str, Any]] = []
        start = 0
        for wid in range(workers):
            end = min(total_items, start + chunk)
            schedule.append({"worker": wid, "start": start, "stop": end})
            start = end
            if start >= total_items:
                break
        return schedule

    def _record_schedule(
        self,
        operation: str,
        coordinate_system: CoordinateSystemInterface,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        symmetric: bool | None = None,
    ) -> None:
        nbasis = coordinate_system.getNumBasisFunctions()
        self.last_schedule = {
            "operation": operation,
            "backend": self.backend,
            "device": self.device,
            "workers": self.workers,
            "chunk_size": self.chunk_size,
            "sparse": bool(sparse),
            "mode": str(mode) if mode is not None else None,
            "analytic": bool(analytic),
            "symmetric": symmetric,
            "schedule": self._schedule(nbasis),
        }

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        self._record_schedule("vector", coordinate_system, sparse, mode, analytic)
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
        self._record_schedule("matrix", coordinate_system, sparse, mode, analytic, symmetric=symmetric)
        return thunk()


class HybridParallelPolicy(CompositeParallelPolicy):
    """
    Convenience policy that composes distributed and shared-memory reflections.

    By default the distributed policy wraps the shared one, so MPI partitions
    dispatch into shared-memory workers on each rank.
    """

    name = "hybrid"

    def __init__(
        self,
        distributed: ParallelPolicy,
        shared: ParallelPolicy,
        *additional: ParallelPolicy,
    ) -> None:
        if distributed.category != "distributed":
            raise ValueError("HybridParallelPolicy expects a distributed policy as the first argument.")
        if shared.category != "shared":
            raise ValueError("HybridParallelPolicy expects a shared-memory policy as the second argument.")
        super().__init__([distributed, shared, *additional])
        self.distributed = distributed
        self.shared = shared


def _flatten_policy_sequence(items: Iterable["ParallelPolicy | Sequence[ParallelPolicy] | None"]) -> Iterable[ParallelPolicy]:
    for item in items:
        if item is None:
            continue
        if isinstance(item, ParallelPolicy):
            yield item
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            yield from _flatten_policy_sequence(item)
        else:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported policy input: {item!r}")


def compose_parallel_policies(*policies: "ParallelPolicy | Sequence[ParallelPolicy] | None") -> ParallelPolicy:
    """Normalize and compose policy inputs."""
    flat = list(_flatten_policy_sequence(policies))
    if not flat:
        raise ValueError("compose_parallel_policies requires at least one policy.")
    if len(flat) == 1:
        return flat[0]
    return CompositeParallelPolicy(flat)


def _ensure_parallel_policy(policy: "ParallelPolicy | Sequence[ParallelPolicy] | None") -> ParallelPolicy:
    if policy is None:
        return ParallelPolicy()
    if isinstance(policy, ParallelPolicy):
        return policy
    if isinstance(policy, Sequence) and not isinstance(policy, (str, bytes)):
        flat = list(_flatten_policy_sequence([policy]))
        if not flat:
            return ParallelPolicy()
        if len(flat) == 1:
            return flat[0]
        return CompositeParallelPolicy(flat)
    raise TypeError(f"Unsupported policy argument: {policy!r}")


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
        self._policy = _ensure_parallel_policy(policy)
        self._policy.setup(self._base)

    # ------------------------------------------------------------------ #
    # Shared state                                                       #
    # ------------------------------------------------------------------ #
    @property
    def policy(self) -> ParallelPolicy:
        return self._policy

    def configure_policy(self, policy: ParallelPolicy | Sequence[ParallelPolicy] | None) -> None:
        self._policy = _ensure_parallel_policy(policy)
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
