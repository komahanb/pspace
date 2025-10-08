from __future__ import annotations

import argparse
import time

from pspace.core import BasisFunctionType, InnerProductMode
from pspace.profile import CoordinateSystem as ProfileCoordinateSystem

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = rng(args.seed)

    basis_type = (
        BasisFunctionType.TENSOR_DEGREE
        if args.basis == "tensor"
        else BasisFunctionType.TOTAL_DEGREE
    )

    base_cs = build_coordinate_system(
        basis_type=basis_type,
        num_coords=args.num_coords,
        max_degree=args.max_degree,
        generator=generator,
    )

    profile_cs = ProfileCoordinateSystem(basis_type, verbose=False)
    for coord in base_cs.coordinates.values():
        profile_cs.addCoordinateAxis(coord)
    profile_cs.initialize()

    polynomial = random_polynomial(
        profile_cs.numeric,
        generator,
        max_degree=args.max_degree,
    )

    mode = InnerProductMode(args.mode)
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

    print(
        f"Profiling mode={mode.value}, rank={args.rank}, basis={basis_type.name}, "
        f"coords={args.num_coords}, max_degree={args.max_degree}, trials={args.trials}"
    )
    if args.rank == "vector":
        print(f"Sparse={args.sparse}\n")
    else:
        print(f"Sparse={args.sparse}, symmetric={args.symmetric}\n")

    print("Problem summary:")
    print(f"  seed       : {args.seed}")
    print(f"  nbasis     : {nbasis}")
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


if __name__ == "__main__":
    main()
