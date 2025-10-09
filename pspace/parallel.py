from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

from .core import InnerProductMode, PolyFunction
from .interface import CoordinateSystem as CoordinateSystemInterface
from .stochastic_utils import sum_degrees_union_matrix, sum_degrees_union_vector


def _vector_mask(cs: CoordinateSystemInterface, function: PolyFunction, sparse: bool) -> list[int]:
    if sparse:
        mask = cs.polynomial_vector_sparsity_mask(function.degrees)
    else:
        mask = cs.basis.keys()
    return sorted(int(idx) for idx in mask)


def _matrix_mask(
    cs: CoordinateSystemInterface,
    function: PolyFunction,
    sparse: bool,
    symmetric: bool,
) -> list[tuple[int, int]]:
    if sparse:
        mask = cs.polynomial_sparsity_mask(function.degrees, symmetric=symmetric)
    else:
        basis_ids = sorted(cs.basis.keys())
        if symmetric:
            mask = {(i, j) for ii, i in enumerate(basis_ids) for j in basis_ids[ii:]}
        else:
            mask = {(i, j) for i in basis_ids for j in basis_ids}
    return sorted((int(i), int(j)) for i, j in mask)


def _compute_vector_chunk(
    cs: CoordinateSystemInterface,
    function: PolyFunction,
    indices: Sequence[int],
) -> Dict[int, float]:
    coeffs: Dict[int, float] = {}
    for k in indices:
        psi_k = cs.basis[k]
        need = sum_degrees_union_vector(function.max_degrees, psi_k)
        qmap = cs.build_quadrature(need)
        s = 0.0
        for q in qmap.values():
            y = q["Y"]
            s += function(y) * cs.evaluateBasisDegreesY(y, psi_k) * q["W"]
        coeffs[int(k)] = float(s)
    return coeffs


def _compute_matrix_chunk(
    cs: CoordinateSystemInterface,
    function: PolyFunction,
    pairs: Sequence[tuple[int, int]],
    symmetric: bool,
) -> Dict[tuple[int, int], float]:
    qcache: Dict[tuple[tuple[int, int], ...], Mapping[int, Mapping[str, Any]]] = {}
    partial: Dict[tuple[int, int], float] = {}
    for i, j in pairs:
        psi_i, psi_j = cs.basis[i], cs.basis[j]
        need = sum_degrees_union_matrix(function.max_degrees, psi_i, psi_j)
        key = tuple(sorted(need.items()))
        qmap = qcache.get(key)
        if qmap is None:
            qmap = cs.build_quadrature(need)
            qcache[key] = qmap
        s = 0.0
        for q in qmap.values():
            y = q["Y"]
            s += (
                function(y)
                * cs.evaluateBasisDegreesY(y, psi_i)
                * cs.evaluateBasisDegreesY(y, psi_j)
                * q["W"]
            )
        sval = float(s)
        partial[(int(i), int(j))] = sval
        if symmetric and i != j:
            partial[(int(j), int(i))] = sval
    return partial


def _partition_list(sequence: Sequence[Any], parts: int) -> list[list[Any]]:
    if not sequence:
        return []
    parts = max(1, parts)
    size = max(1, math.ceil(len(sequence) / parts))
    return [list(sequence[idx : idx + size]) for idx in range(0, len(sequence), size)]


def _ensure_sparse_fill(
    coeffs: Dict[int, float],
    mask: Sequence[int],
    basis_keys: Iterable[int] | None = None,
) -> Dict[int, float]:
    for idx in mask:
        coeffs.setdefault(int(idx), 0.0)
    if basis_keys is not None:
        for idx in basis_keys:
            coeffs.setdefault(int(idx), 0.0)
    return coeffs


def _normalize_mode(mode: InnerProductMode | str | bool | None) -> InnerProductMode:
    if mode is None:
        return InnerProductMode.NUMERICAL
    if isinstance(mode, bool):
        return InnerProductMode.SYMBOLIC if mode else InnerProductMode.NUMERICAL
    if isinstance(mode, InnerProductMode):
        return mode
    if isinstance(mode, str):
        return InnerProductMode(mode.lower())
    raise ValueError(f"Unsupported inner product mode: {mode}")


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
        total_items: int | None = None,
    ) -> None:
        items = total_items if total_items is not None else coordinate_system.getNumBasisFunctions()
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
            "schedule": self._schedule(items),
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
        normalized = _normalize_mode(mode)
        if analytic or normalized is not InnerProductMode.NUMERICAL:
            self._record_schedule("vector", coordinate_system, sparse, mode, analytic)
            return thunk()

        if self.backend.lower() in {"cuda", "cupy"}:
            # GPU-backed shared memory is handled by the CUDA policy.
            self._record_schedule("vector", coordinate_system, sparse, mode, analytic)
            return thunk()

        coordinate_system_eval = coordinate_system
        function.bind_coordinates(coordinate_system_eval.coordinates)

        mask = _vector_mask(coordinate_system_eval, function, sparse)
        total_items = len(mask)

        self._record_schedule("vector", coordinate_system_eval, sparse, mode, analytic, total_items=total_items)

        workers = self.workers or 1
        if total_items == 0 or workers <= 1 or len(mask) <= 1:
            return thunk()

        partitions = [chunk for chunk in _partition_list(mask, workers) if chunk]
        if not partitions:
            return thunk()

        coeffs: Dict[int, float] = {}
        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            futures = [
                executor.submit(_compute_vector_chunk, coordinate_system_eval, function, chunk)
                for chunk in partitions
            ]
            for future in as_completed(futures):
                coeffs.update(future.result())

        coeffs = _ensure_sparse_fill(coeffs, mask, coordinate_system_eval.basis.keys())

        return coeffs

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
        normalized = _normalize_mode(mode)
        if analytic or normalized is not InnerProductMode.NUMERICAL:
            self._record_schedule("matrix", coordinate_system, sparse, mode, analytic, symmetric=symmetric)
            return thunk()

        if self.backend.lower() in {"cuda", "cupy"}:
            self._record_schedule("matrix", coordinate_system, sparse, mode, analytic, symmetric=symmetric)
            return thunk()

        coordinate_system_eval = coordinate_system
        function.bind_coordinates(coordinate_system_eval.coordinates)

        mask_pairs = _matrix_mask(coordinate_system_eval, function, sparse, symmetric)
        total_items = len(mask_pairs)
        self._record_schedule(
            "matrix",
            coordinate_system_eval,
            sparse,
            mode,
            analytic,
            symmetric=symmetric,
            total_items=total_items,
        )

        workers = self.workers or 1
        if total_items == 0 or workers <= 1:
            return thunk()

        partitions = [chunk for chunk in _partition_list(mask_pairs, workers) if chunk]
        if not partitions:
            return thunk()

        nbasis = coordinate_system_eval.getNumBasisFunctions()
        matrix = np.zeros((nbasis, nbasis))

        with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
            futures = [
                executor.submit(
                    _compute_matrix_chunk,
                    coordinate_system_eval,
                    function,
                    chunk,
                    symmetric,
                )
                for chunk in partitions
            ]
            for future in as_completed(futures):
                partial = future.result()
                for (i, j), value in partial.items():
                    matrix[int(i), int(j)] = value

        return matrix


class OpenMPParallelPolicy(SharedMemoryParallelPolicy):
    """
    Convenience policy configuring the shared-memory reflection for OpenMP-style workers.
    """

    name = "shared.openmp"

    def __init__(self, workers: int | None = None, chunk_size: int | None = None) -> None:
        super().__init__(backend="openmp", workers=workers, chunk_size=chunk_size)


class MPI4PyParallelPolicy(MPIParallelPolicy):
    """
    Distributed reflection backed by mpi4py for multi-process execution.
    """

    name = "distributed.mpi4py"

    def __init__(
        self,
        world_size: int | None = None,
        ranks: Sequence[int] | None = None,
        communicator: str = "COMM_WORLD",
        mpi_comm: Any | None = None,
    ) -> None:
        super().__init__(world_size=world_size, ranks=ranks, communicator=communicator)
        self._comm = mpi_comm
        self.rank: int | None = None

    def setup(self, coordinate_system: CoordinateSystemInterface) -> None:
        try:
            from mpi4py import MPI  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self._comm = None
            self.rank = 0
            self.world_size = 1
            self.ranks = [0]
            return

        if self._comm is None:
            comm_name = self.communicator
            if isinstance(comm_name, str) and hasattr(MPI, comm_name):
                self._comm = getattr(MPI, comm_name)
            else:
                self._comm = MPI.COMM_WORLD

        self.rank = self._comm.Get_rank()
        self.world_size = self._comm.Get_size()
        self.ranks = list(range(self.world_size))

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        normalized = _normalize_mode(mode)
        if analytic or normalized is not InnerProductMode.NUMERICAL:
            return super().execute_vector(coordinate_system, function, sparse, mode, analytic, thunk)
        if self._comm is None or self.world_size <= 1:
            return super().execute_vector(coordinate_system, function, sparse, mode, analytic, thunk)

        coordinate_system_eval = coordinate_system
        function.bind_coordinates(coordinate_system_eval.coordinates)

        mask = _vector_mask(coordinate_system_eval, function, sparse)
        partitions = _partition_list(mask, self.world_size)
        self.last_plan = {
            "operation": "vector",
            "sparse": bool(sparse),
            "mode": str(mode) if mode is not None else None,
            "analytic": bool(analytic),
            "symmetric": None,
            "partitions": partitions,
            "communicator": self.communicator,
            "world_size": self.world_size,
        }

        rank = self.rank or 0
        indices = partitions[rank] if rank < len(partitions) else []
        partial = _compute_vector_chunk(coordinate_system_eval, function, indices) if indices else {}
        gathered = self._comm.allgather(partial)
        coeffs: Dict[int, float] = {}
        for chunk in gathered:
            coeffs.update(chunk)

        coeffs = _ensure_sparse_fill(coeffs, mask, coordinate_system_eval.basis.keys())
        return coeffs

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
        normalized = _normalize_mode(mode)
        if analytic or normalized is not InnerProductMode.NUMERICAL:
            return super().execute_matrix(coordinate_system, function, sparse, symmetric, mode, analytic, thunk)
        if self._comm is None or self.world_size <= 1:
            return super().execute_matrix(coordinate_system, function, sparse, symmetric, mode, analytic, thunk)

        coordinate_system_eval = coordinate_system
        function.bind_coordinates(coordinate_system_eval.coordinates)

        mask_pairs = _matrix_mask(coordinate_system_eval, function, sparse, symmetric)
        partitions = _partition_list(mask_pairs, self.world_size)
        self.last_plan = {
            "operation": "matrix",
            "sparse": bool(sparse),
            "mode": str(mode) if mode is not None else None,
            "analytic": bool(analytic),
            "symmetric": symmetric,
            "partitions": partitions,
            "communicator": self.communicator,
            "world_size": self.world_size,
        }

        rank = self.rank or 0
        pairs = partitions[rank] if rank < len(partitions) else []
        partial = _compute_matrix_chunk(coordinate_system_eval, function, pairs, symmetric) if pairs else {}

        gathered = self._comm.allgather(partial)

        nbasis = coordinate_system_eval.getNumBasisFunctions()
        matrix = np.zeros((nbasis, nbasis))
        for chunk in gathered:
            for (i, j), value in chunk.items():
                matrix[int(i), int(j)] = value
        return matrix


class CudaCupyParallelPolicy(ParallelPolicy):
    """
    Shared-memory reflection that mirrors assembled data onto a CUDA device via CuPy.
    """

    name = "shared.cuda"
    category = "shared"

    def __init__(
        self,
        device: str | None = None,
        mirror_vector: bool = True,
        mirror_matrix: bool = True,
    ) -> None:
        self.device = device
        self.mirror_vector = mirror_vector
        self.mirror_matrix = mirror_matrix
        self._cupy: Any | None = None
        self.last_vector_gpu = None
        self.last_vector_basis_order: list[int] | None = None
        self.last_matrix_gpu = None

    def setup(self, coordinate_system: CoordinateSystemInterface) -> None:
        try:
            import cupy as cp  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self._cupy = None
            return

        self._cupy = cp
        if self.device is not None:
            cp.cuda.Device(self.device).use()

    def execute_vector(
        self,
        coordinate_system: CoordinateSystemInterface,
        function: PolyFunction,
        sparse: bool,
        mode: InnerProductMode | str | None,
        analytic: bool,
        thunk: Callable[[], Dict[int, Any]],
    ) -> Dict[int, Any]:
        coeffs = thunk()
        if not self.mirror_vector or self._cupy is None:
            self.last_vector_gpu = None
            self.last_vector_basis_order = None
            return coeffs

        basis_order = sorted(int(k) for k in coeffs)
        host_vec = np.array([coeffs[k] for k in basis_order], dtype=np.float64)
        self.last_vector_gpu = self._cupy.asarray(host_vec)
        self.last_vector_basis_order = basis_order
        return coeffs

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
        matrix = thunk()
        if not self.mirror_matrix or self._cupy is None:
            self.last_matrix_gpu = None
            return matrix

        self.last_matrix_gpu = self._cupy.asarray(matrix)
        return matrix


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
