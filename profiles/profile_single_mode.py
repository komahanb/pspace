from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from importlib import import_module

from pspace.numeric import BasisFunctionType, InnerProductMode
from pspace.profile import ProfileCoordinateSystem
from pspace.parallel import (
    ParallelCoordinateSystem,
    ParallelPolicy,
    MPIParallelPolicy,
    MPI4PyParallelPolicy,
    SharedMemoryParallelPolicy,
    OpenMPParallelPolicy,
    CudaCupyParallelPolicy,
    compose_parallel_policies,
)

try:  # pragma: no cover - runtime convenience
    from profiles.helpers import build_coordinate_system, random_polynomial, rng
except ImportError:  # pragma: no cover
    from helpers import build_coordinate_system, random_polynomial, rng


def str_to_bool(value: str) -> bool:
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    lowered = value.lower()
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got '{value}'")


def describe_backend(coordinate_system) -> str:
    cls = coordinate_system.__class__
    return f"{cls.__module__}.{cls.__name__}"


BACKEND_REGISTRY: dict[InnerProductMode, tuple[str, str]] = {
    InnerProductMode.NUMERICAL: ("pspace.numeric", "NumericCoordinateSystem"),
    InnerProductMode.SYMBOLIC: ("pspace.symbolic", "SymbolicNumericCoordinateSystem"),
    InnerProductMode.ANALYTIC: ("pspace.analytic", "AnalyticNumericCoordinateSystem"),
}


def build_backend_coordinate_system(
    mode: InnerProductMode,
    basis_type: BasisFunctionType,
    verbose: bool = False,
):
    module_name, class_name = BACKEND_REGISTRY.get(
        mode,
        BACKEND_REGISTRY[InnerProductMode.NUMERICAL],
    )
    try:
        module = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"Unable to import backend '{module_name}' required for mode '{mode.value}'. "
            "Ensure the optional dependencies for that backend are installed."
        ) from exc

    backend_cls = getattr(module, class_name)
    return backend_cls(basis_type, verbose=verbose)


def _walk_policies(policy: ParallelPolicy | None):
    if policy is None:
        return
    children = getattr(policy, "policies", None)
    if children:
        for child in children:
            yield from _walk_policies(child)
    else:
        yield policy


def summarize_policy(policy: ParallelPolicy | None) -> str:
    if policy is None:
        return "sequential"

    children = getattr(policy, "policies", None)
    if children:
        return " + ".join(summarize_policy(child) for child in children)

    details: list[str] = []
    if isinstance(policy, MPIParallelPolicy):
        world = getattr(policy, "world_size", None)
        communicator = getattr(policy, "communicator", None)
        if communicator:
            details.append(f"comm={communicator}")
        if world:
            details.append(f"world={world}")
    if isinstance(policy, SharedMemoryParallelPolicy):
        backend = getattr(policy, "backend", None)
        workers = getattr(policy, "workers", None)
        chunk = getattr(policy, "chunk_size", None)
        if backend:
            details.append(str(backend))
        if workers:
            details.append(f"workers={workers}")
        if chunk:
            details.append(f"chunk={chunk}")
    if isinstance(policy, CudaCupyParallelPolicy):
        device = getattr(policy, "device", None)
        if device:
            details.append(f"device={device}")

    name = getattr(policy, "name", policy.__class__.__name__)
    if details:
        return f"{name}({', '.join(details)})"
    return name


def collect_parallel_diagnostics(policy: ParallelPolicy | None) -> list[str]:
    notes: list[str] = []
    for component in _walk_policies(policy):
        if isinstance(component, MPIParallelPolicy):
            plan = getattr(component, "last_plan", None)
            if plan:
                partitions = plan.get("partitions") or []
                notes.append(
                    f"MPI plan: world={plan.get('world_size')}, comm={plan.get('communicator')}, partitions={len(partitions)}"
                )
        elif isinstance(component, SharedMemoryParallelPolicy):
            schedule = getattr(component, "last_schedule", None)
            if schedule:
                entries = schedule.get("schedule") or []
                notes.append(
                    f"Shared schedule ({schedule.get('backend')}): workers={schedule.get('workers')}, chunks={len(entries)}"
                )
        elif isinstance(component, CudaCupyParallelPolicy):
            if getattr(component, "last_vector_gpu", None) is not None:
                size = int(component.last_vector_gpu.size)
                notes.append(f"CuPy vector mirror: length={size}")
            if getattr(component, "last_matrix_gpu", None) is not None:
                shape = tuple(int(dim) for dim in component.last_matrix_gpu.shape)
                notes.append(f"CuPy matrix mirror: shape={shape}")
    return notes


def configure_parallel(
    profile_cs: ProfileCoordinateSystem,
    args: argparse.Namespace,
) -> ParallelPolicy | None:
    policies: list[ParallelPolicy] = []

    if getattr(args, "parallel", False):
        policies.append(
            MPI4PyParallelPolicy(communicator=getattr(args, "parallel_communicator", "COMM_WORLD"))
        )

    shared = getattr(args, "shared", "none") or "none"
    shared = shared.lower()
    if shared not in {"none", "threads", "threadpool", "openmp", "cupy", "cuda"}:
        raise argparse.ArgumentTypeError(f"Unsupported shared backend '{shared}'")

    if shared in {"threads", "threadpool"}:
        policies.append(
            SharedMemoryParallelPolicy(
                backend="threadpool",
                workers=getattr(args, "shared_workers", None),
                chunk_size=getattr(args, "shared_chunk_size", None),
            )
        )
    elif shared == "openmp":
        policies.append(
            OpenMPParallelPolicy(
                workers=getattr(args, "shared_workers", None),
                chunk_size=getattr(args, "shared_chunk_size", None),
            )
        )
    elif shared in {"cupy", "cuda"}:
        policies.append(
            CudaCupyParallelPolicy(
                device=getattr(args, "shared_device", None),
            )
        )

    if not policies:
        return None

    if len(policies) == 1:
        policy = policies[0]
    else:
        policy = compose_parallel_policies(policies)

    numeric = profile_cs.numeric
    if isinstance(numeric, ParallelCoordinateSystem):
        numeric.configure_policy(policy)
        return numeric.policy

    profile_cs.numeric = ParallelCoordinateSystem(
        numeric,
        policy=policy,
        verbose=getattr(profile_cs, "verbose", False),
    )
    return profile_cs.numeric.policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a single decomposition mode for the coordinate system."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[m.value for m in InnerProductMode],
        help="Inner-product mode to profile.",
    )
    parser.add_argument("--rank", type=str, choices=["vector", "matrix"], default="vector")
    parser.add_argument("--basis", type=str, choices=["tensor", "total"], default="tensor")
    parser.add_argument("--num-coords", type=int, default=3)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--sparse",
        type=str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Enable/disable sparsity masking (default: true).",
    )
    parser.add_argument(
        "--symmetric",
        type=str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Enforce symmetric assembly for matrices (default: true).",
    )
    parser.add_argument(
        "--parallel",
        type=str_to_bool,
        default=False,
        metavar="{true,false}",
        help="Enable MPI-driven distributed parallelism (default: false).",
    )
    parser.add_argument(
        "--parallel-communicator",
        type=str,
        default="COMM_WORLD",
        help="MPI communicator name to use when --parallel=true (default: COMM_WORLD).",
    )
    parser.add_argument(
        "--shared",
        type=str,
        default="none",
        choices=["none", "threads", "threadpool", "openmp", "cupy", "cuda"],
        help="Shared-memory accelerator to activate (default: none).",
    )
    parser.add_argument(
        "--shared-workers",
        type=int,
        default=None,
        help="Worker count for shared thread/OpenMP backends.",
    )
    parser.add_argument(
        "--shared-chunk-size",
        type=int,
        default=None,
        help="Chunk size per worker for shared thread/OpenMP backends.",
    )
    parser.add_argument(
        "--shared-device",
        type=str,
        default=None,
        help="CUDA device string for the CuPy backend (default: CuPy default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = rng(args.seed)

    basis_type = (
        BasisFunctionType.TENSOR_DEGREE
        if args.basis == "tensor"
        else BasisFunctionType.TOTAL_DEGREE
    )

    mode = InnerProductMode(args.mode)
    backend_cs = build_backend_coordinate_system(mode, basis_type, verbose=False)
    backend_label = describe_backend(backend_cs)

    base_cs = build_coordinate_system(
        basis_type=basis_type,
        num_coords=args.num_coords,
        max_degree=args.max_degree,
        generator=generator,
    )

    profile_cs = ProfileCoordinateSystem(basis_type, verbose=False, backend=backend_cs)
    for coord in base_cs.coordinates.values():
        profile_cs.addCoordinateAxis(coord)
    profile_cs.initialize()

    active_policy: ParallelPolicy | None = None
    if args.parallel or args.shared.lower() != "none":
        active_policy = configure_parallel(profile_cs, args)
    elif isinstance(profile_cs.numeric, ParallelCoordinateSystem):
        active_policy = profile_cs.numeric.policy

    polynomial = random_polynomial(
        profile_cs.numeric,
        generator,
        max_degree=args.max_degree,
    )

    nbasis = profile_cs.getNumBasisFunctions()
    coord_descriptions = []
    for coord in profile_cs.numeric.coordinates.values():
        dist_name = coord.distribution.name.lower()
        coord_descriptions.append(
            f"{coord.name}:{dist_name}(deg={coord.degree})"
        )
    coords_summary = ", ".join(coord_descriptions)
    term_summaries = []
    for coeff, degrees in polynomial.terms:
        if degrees:
            term_summaries.append(f"{coeff:+.3g} * {dict(degrees)}")
        else:
            term_summaries.append(f"{coeff:+.3g}")

    if isinstance(profile_cs.numeric, ParallelCoordinateSystem):
        active_policy = profile_cs.numeric.policy
    policy_summary = summarize_policy(active_policy)

    print(
        f"Profiling mode={mode.value}, rank={args.rank}, basis={basis_type.name}, "
        f"coords={args.num_coords}, max_degree={args.max_degree}, trials={args.trials}"
    )
    if args.rank == "vector":
        print(f"Sparse={args.sparse}, backend={backend_label}, parallel={policy_summary}\n")
    else:
        print(
            f"Sparse={args.sparse}, symmetric={args.symmetric}, "
            f"backend={backend_label}, parallel={policy_summary}\n"
        )

    print("Problem summary:")
    print(f"  seed       : {args.seed}")
    print(f"  nbasis     : {nbasis}")
    print(f"  backend    : {backend_label}")
    print(f"  parallel   : {policy_summary}")
    print(f"  coordinates: {coords_summary or 'none'}")
    print(f"  polynomial : {len(polynomial.terms)} terms")
    for idx, summary in enumerate(term_summaries, start=1):
        print(f"     term {idx:02d}: {summary}")
    print("")

    elapsed_samples: list[float] = []

    for trial in range(args.trials):
        start = time.perf_counter()

        if args.rank == "vector":
            profile_cs.decompose(
                polynomial,
                sparse=args.sparse,
                mode=mode,
            )
            elapsed = profile_cs.last_timing("decompose")
        else:
            profile_cs.decompose_matrix(
                polynomial,
                sparse=args.sparse,
                symmetric=args.symmetric,
                mode=mode,
            )
            elapsed = profile_cs.last_timing("decompose_matrix")

        if elapsed is None:
            elapsed = time.perf_counter() - start
        elapsed_samples.append(float(elapsed))
        print(f"Trial {trial+1}/{args.trials}: {elapsed_samples[-1]:.6f} s")

    total = sum(elapsed_samples)
    mean = total / len(elapsed_samples)
    best = min(elapsed_samples)
    worst = max(elapsed_samples)

    print("\nSummary:")
    print(f"  best : {best:.6f} s")
    print(f"  worst: {worst:.6f} s")
    print(f"  mean : {mean:.6f} s")

    if isinstance(profile_cs.numeric, ParallelCoordinateSystem):
        diagnostics = collect_parallel_diagnostics(profile_cs.numeric.policy)
        if diagnostics:
            print("\nParallel diagnostics:")
            for note in diagnostics:
                print(f"  - {note}")


if __name__ == "__main__":
    main()
