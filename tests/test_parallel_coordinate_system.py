from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pspace.numeric import BasisFunctionType, InnerProductMode
from pspace.parallel import (
    ParallelCoordinateSystem,
    ParallelPolicy,
    CompositeParallelPolicy,
    DistributedBasisParallel,
    DistributedCoordinateParallel,
    HybridParallelPolicy,
    MPIParallelPolicy,
    SharedMemoryParallelPolicy,
    OpenMPParallelPolicy,
    MPI4PyParallelPolicy,
    CudaCupyParallelPolicy,
    compose_parallel_policies,
)
from pspace.sparsity import SparsityCoordinateSystem
from tests.utils.factories import build_numeric_coordinate_system, make_polynomial


class CountingPolicy(ParallelPolicy):
    def __init__(self) -> None:
        self.vector_calls = 0
        self.matrix_calls = 0

    def execute_vector(self, coordinate_system, function, sparse, mode, analytic, thunk):
        self.vector_calls += 1
        return thunk()

    def execute_matrix(self, coordinate_system, function, sparse, symmetric, mode, analytic, thunk):
        self.matrix_calls += 1
        return thunk()


def test_parallel_policy_invocation():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    policy = CountingPolicy()
    wrapper = ParallelCoordinateSystem(base, policy=policy)

    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix = wrapper.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert policy.vector_calls == 1
    assert policy.matrix_calls == 1

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix_direct = base.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    for k in coeffs:
        assert np.isclose(coeffs[k], coeffs_direct[k])
    assert np.allclose(matrix, matrix_direct)


def test_parallel_policy_switching():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    wrapper = ParallelCoordinateSystem(base, policy=DistributedBasisParallel(chunks=2))
    coeffs_basis = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)

    wrapper.configure_policy(DistributedCoordinateParallel())
    coeffs_coord = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs_basis.keys()) == set(coeffs_coord.keys())
    for k in coeffs_basis:
        assert np.isclose(coeffs_basis[k], coeffs_coord[k])


def test_parallel_composed_with_sparsity():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    sparse = SparsityCoordinateSystem(base, enabled=True)
    parallel = ParallelCoordinateSystem(sparse, policy=DistributedBasisParallel())

    poly = make_polynomial(base)
    coeffs_parallel = parallel.decompose(poly, sparse=None, mode=InnerProductMode.NUMERICAL)

    # Disable sparsity after wrapping to ensure delegation reaches through
    parallel.configure_sparsity(False)
    coeffs_dense = parallel.decompose(poly, sparse=None, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs_parallel.keys()) == set(coeffs_dense.keys())
    for k in coeffs_parallel:
        assert np.isclose(coeffs_parallel[k], coeffs_dense[k])

def test_shared_memory_parallel_policy_threads():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    policy = SharedMemoryParallelPolicy(backend="threadpool", workers=2)
    wrapper = ParallelCoordinateSystem(base, policy=policy)

    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix = wrapper.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix_direct = base.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert policy.last_schedule is not None
    assert policy.last_schedule["operation"] == "matrix"
    assert len(policy.last_schedule["schedule"]) <= policy.workers

    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    for k in coeffs_direct:
        assert np.isclose(coeffs[k], coeffs_direct[k])
    assert np.allclose(matrix, matrix_direct)


def test_parallel_policy_composite_runtime():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    mpi = MPIParallelPolicy(world_size=3)
    shared = SharedMemoryParallelPolicy(backend="openmp", workers=4)
    hybrid = HybridParallelPolicy(mpi, shared)

    wrapper = ParallelCoordinateSystem(base, policy=hybrid)

    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix = wrapper.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert isinstance(wrapper.policy, CompositeParallelPolicy)
    assert hybrid.distributed.last_plan is not None
    assert hybrid.distributed.last_plan["operation"] == "matrix"
    assert hybrid.distributed.last_plan["world_size"] == 3
    assert len(hybrid.distributed.last_plan["partitions"]) <= 3

    assert hybrid.shared.last_schedule is not None
    assert hybrid.shared.last_schedule["operation"] == "matrix"
    assert hybrid.shared.last_schedule["backend"] == "openmp"
    assert hybrid.shared.last_schedule["workers"] == 4
    assert len(hybrid.shared.last_schedule["schedule"]) <= 4

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix_direct = base.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    assert np.allclose(matrix, matrix_direct)


def test_parallel_policy_sequence_and_operator_overload():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    mpi = MPIParallelPolicy(world_size=2)
    shared = SharedMemoryParallelPolicy(backend="cuda", workers=None, device="cuda:1")

    wrapper_list = ParallelCoordinateSystem(base, policy=[mpi, shared])
    policy_from_list = wrapper_list.policy
    assert isinstance(policy_from_list, CompositeParallelPolicy)
    assert policy_from_list.policies[0] is mpi
    assert policy_from_list.policies[1] is shared

    coeffs_list = wrapper_list.decompose(poly, mode=InnerProductMode.NUMERICAL)

    # Use operator composition
    mpi2 = MPIParallelPolicy(world_size=2)
    shared2 = SharedMemoryParallelPolicy(backend="threadpool", workers=2)
    composed = mpi2 | shared2

    assert isinstance(composed, CompositeParallelPolicy)
    assert composed.policies[0] is mpi2
    assert composed.policies[1] is shared2

    wrapper = ParallelCoordinateSystem(base, policy=composed)
    coeffs_or = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)

    manual = compose_parallel_policies(mpi2, [shared2])
    assert isinstance(manual, CompositeParallelPolicy)
    coeffs_manual = ParallelCoordinateSystem(base, policy=manual).decompose(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs_list.keys()) == set(coeffs_or.keys())
    assert set(coeffs_or.keys()) == set(coeffs_manual.keys())
    for k in coeffs_list:
        assert np.isclose(coeffs_list[k], coeffs_or[k])
        assert np.isclose(coeffs_or[k], coeffs_manual[k])


def test_openmp_parallel_policy_alias():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    policy = OpenMPParallelPolicy(workers=2)
    wrapper = ParallelCoordinateSystem(base, policy=policy)
    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix = wrapper.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix_direct = base.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    for k in coeffs_direct:
        assert np.isclose(coeffs[k], coeffs_direct[k])
    assert np.allclose(matrix, matrix_direct)


def test_mpi4py_parallel_policy_comm_self(monkeypatch):
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    try:
        from mpi4py import MPI  # type: ignore
    except ImportError:
        class DummyComm:
            def Get_size(self):
                return 1

            def Get_rank(self):
                return 0

            def allgather(self, value):
                return [value]

        def fake_setup(self, coordinate_system):
            self.world_size = self._comm.Get_size() if self._comm is not None else 1
            self.rank = self._comm.Get_rank() if self._comm is not None else 0
            self.ranks = list(range(self.world_size))

        monkeypatch.setattr(MPI4PyParallelPolicy, "setup", fake_setup, raising=False)

        policy = MPI4PyParallelPolicy(mpi_comm=DummyComm())
    else:
        policy = MPI4PyParallelPolicy(mpi_comm=MPI.COMM_SELF)

    wrapper = ParallelCoordinateSystem(base, policy=policy)

    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    for k in coeffs_direct:
        assert np.isclose(coeffs[k], coeffs_direct[k])

    assert policy.last_plan is not None
    assert policy.last_plan["world_size"] == 1


def test_cuda_cupy_parallel_policy_mirror():
    base = build_numeric_coordinate_system(BasisFunctionType.TENSOR_DEGREE)
    poly = make_polynomial(base)

    policy = CudaCupyParallelPolicy(mirror_vector=True, mirror_matrix=True)
    wrapper = ParallelCoordinateSystem(base, policy=policy)

    coeffs = wrapper.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix = wrapper.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    coeffs_direct = base.decompose(poly, mode=InnerProductMode.NUMERICAL)
    matrix_direct = base.decompose_matrix(poly, mode=InnerProductMode.NUMERICAL)

    assert set(coeffs.keys()) == set(coeffs_direct.keys())
    for k in coeffs_direct:
        assert np.isclose(coeffs[k], coeffs_direct[k])
    assert np.allclose(matrix, matrix_direct)

    if policy._cupy is None:
        assert policy.last_vector_gpu is None
        assert policy.last_matrix_gpu is None
    else:
        import cupy

        assert isinstance(policy.last_vector_gpu, cupy.ndarray)
        assert isinstance(policy.last_matrix_gpu, cupy.ndarray)
