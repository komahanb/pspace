from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from pspace.numeric import BasisFunctionType, InnerProductMode
from pspace.profile import ProfileNumericCoordinateSystem

try:  # pragma: no cover - runtime convenience
    from profiles.helpers import build_coordinate_system, random_polynomial, rng
except ImportError:  # pragma: no cover
    from helpers import build_coordinate_system, random_polynomial, rng

DEFAULT_SEED = 2025

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(rows: Iterable[dict], path: str) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_vector_modes(
    cs: ProfileNumericCoordinateSystem,
    polynomial,
    modes: Sequence[InnerProductMode],
    sparse: bool,
    trials: int,
) -> list[dict]:
    results: list[dict] = []
    nbasis = cs.getNumBasisFunctions()
    for mode in modes:
        for trial in range(trials):
            coeffs = cs.decompose(polynomial, sparse=sparse, mode=mode)
            elapsed = cs.last_timing("decompose")
            results.append(
                {
                    "mode": mode.value,
                    "trial": trial,
                    "elapsed": float(elapsed) if elapsed is not None else np.nan,
                    "nbasis": nbasis,
                    "sparse": sparse,
                    "coeff_count": len(coeffs),
                }
            )
    return results


def summarize_speedups(rows: list[dict]) -> None:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        elapsed = row.get("elapsed")
        if elapsed is None or np.isnan(elapsed):
            continue
        basis = row.get("basis_type", "UNKNOWN")
        mode = row.get("mode", "unknown")
        grouped[basis][mode].append(float(elapsed))

    if not grouped:
        print("No timing data available to summarize.")
        return

    print("\nSpeedup summary (relative to numerical mode):")
    for basis, mode_map in grouped.items():
        numeric_values = mode_map.get(InnerProductMode.NUMERICAL.value)
        if not numeric_values:
            print(f"  {basis}: missing numerical baseline; skipping.")
            continue
        numeric_mean = float(np.mean(numeric_values))
        print(f"  Basis {basis}: numerical avg {numeric_mean:.4e} s")
        for mode, samples in mode_map.items():
            mode_mean = float(np.mean(samples))
            if np.isclose(mode_mean, 0.0):
                ratio = np.inf
            else:
                ratio = numeric_mean / mode_mean
            label = "baseline" if mode == InnerProductMode.NUMERICAL.value else f"{ratio:.2f}× faster"
            print(f"    - {mode:<10}: {mode_mean:.4e} s ({label})")
    print("")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile rank-1 decomposition timings for different inner-product modes."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-coords", type=int, default=3)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--dense", action="store_true", help="Disable sparsity masking.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiles/output/vector",
        help="Directory for CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    generator = rng(args.seed)
    results: list[dict] = []

    for basis_type in (BasisFunctionType.TENSOR_DEGREE, BasisFunctionType.TOTAL_DEGREE):
        base_cs = build_coordinate_system(
            basis_type,
            num_coords=args.num_coords,
            max_degree=args.max_degree,
            generator=generator,
        )

        profile_cs = ProfileNumericCoordinateSystem(basis_type, verbose=False)
        for coord in base_cs.coordinates.values():
            profile_cs.addCoordinateAxis(coord)
        profile_cs.initialize()

        polynomial = random_polynomial(
            profile_cs.numeric,
            generator,
            max_degree=args.max_degree,
        )

        modes = list(InnerProductMode)
        result_rows = profile_vector_modes(
            profile_cs,
            polynomial,
            modes=modes,
            sparse=not args.dense,
            trials=args.trials,
        )

        for row in result_rows:
            row.update(
                {
                    "basis_type": basis_type.name,
                    "num_coords": args.num_coords,
                    "max_degree": args.max_degree,
                }
            )
        results.extend(result_rows)

    csv_path = os.path.join(args.output_dir, "vector_timings.csv")
    write_csv(results, csv_path)
    print(f"Wrote vector timing profiles to {csv_path}")
    summarize_speedups(results)


if __name__ == "__main__":
    main()
