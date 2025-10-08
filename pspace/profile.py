#=====================================================================#
# Profiling utilities for pspace decomposition operators
#=====================================================================#
"""
Object-oriented façade over `pspace.core` that mirrors the decomposition
APIs and provides timing helpers suitable for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from .core import (
    CoordinateSystem,
    InnerProductMode,
    PolyFunction,
)


@dataclass
class ProfileResult:
    """
    Container for profiling statistics.
    """

    elapsed: float
    mode: InnerProductMode
    sparse: bool
    symmetric: bool | None = None


class BaseProfiler:
    """
    Shared functionality for vector/matrix profilers.
    """

    def __init__(self, coordinate_system: CoordinateSystem):
        self.cs = coordinate_system

    @staticmethod
    def _ensure_mode(mode: InnerProductMode | str | None) -> InnerProductMode:
        if mode is None:
            return InnerProductMode.NUMERICAL
        if isinstance(mode, InnerProductMode):
            return mode
        if isinstance(mode, str):
            return InnerProductMode(mode.lower())
        raise ValueError(f"Unsupported mode: {mode!r}")

    @staticmethod
    def _time_callable(fn: Callable[[], object]) -> float:
        import time

        start = time.perf_counter()
        fn()
        return time.perf_counter() - start


class VectorProfiler(BaseProfiler):
    """
    Mirrors CoordinateSystem.decompose while collecting timings.
    """

    def profile(
        self,
        function: PolyFunction,
        mode: InnerProductMode | str | None = None,
        sparse: bool = True,
    ) -> ProfileResult:
        mode_enum = self._ensure_mode(mode)
        elapsed = self._time_callable(
            lambda: self.cs.decompose(function, sparse=sparse, mode=mode_enum)
        )
        return ProfileResult(
            elapsed=elapsed,
            mode=mode_enum,
            sparse=sparse,
            symmetric=None,
        )

    def compare_modes(
        self,
        function: PolyFunction,
        modes: Sequence[InnerProductMode | str | None],
        sparse: bool = True,
    ) -> Dict[InnerProductMode, ProfileResult]:
        results: Dict[InnerProductMode, ProfileResult] = {}
        for mode in modes:
            res = self.profile(function, mode=mode, sparse=sparse)
            results[res.mode] = res
        return results


class MatrixProfiler(BaseProfiler):
    """
    Mirrors CoordinateSystem.decompose_matrix.
    """

    def profile(
        self,
        function: PolyFunction,
        mode: InnerProductMode | str | None = None,
        sparse: bool = False,
        symmetric: bool = True,
    ) -> ProfileResult:
        mode_enum = self._ensure_mode(mode)
        elapsed = self._time_callable(
            lambda: self.cs.decompose_matrix(
                function,
                sparse=sparse,
                symmetric=symmetric,
                mode=mode_enum,
            )
        )
        return ProfileResult(
            elapsed=elapsed,
            mode=mode_enum,
            sparse=sparse,
            symmetric=symmetric,
        )

    def compare_modes(
        self,
        function: PolyFunction,
        modes: Sequence[InnerProductMode | str | None],
        sparse: bool = False,
        symmetric: bool = True,
    ) -> Dict[InnerProductMode, ProfileResult]:
        results: Dict[InnerProductMode, ProfileResult] = {}
        for mode in modes:
            res = self.profile(function, mode=mode, sparse=sparse, symmetric=symmetric)
            results[res.mode] = res
        return results


class CoordinateSystemProfiler:
    """
    Aggregate profiler exposing both vector and matrix benchmarking.
    """

    def __init__(self, coordinate_system: CoordinateSystem):
        self.cs = coordinate_system
        self.vector = VectorProfiler(coordinate_system)
        self.matrix = MatrixProfiler(coordinate_system)

    def benchmark(
        self,
        function: PolyFunction,
        rank: str = "vector",
        modes: Iterable[InnerProductMode | str | None] = (
            InnerProductMode.NUMERICAL,
            InnerProductMode.SYMBOLIC,
            InnerProductMode.ANALYTIC,
        ),
        sparse: bool = True,
        symmetric: bool = True,
    ) -> Dict[InnerProductMode, ProfileResult]:
        """
        Convenience wrapper to benchmark a list of modes for the chosen rank.
        """
        if rank not in {"vector", "matrix"}:
            raise ValueError("rank must be 'vector' or 'matrix'")
        if rank == "vector":
            return self.vector.compare_modes(function, list(modes), sparse=sparse)
        return self.matrix.compare_modes(
            function, list(modes), sparse=sparse, symmetric=symmetric
        )
