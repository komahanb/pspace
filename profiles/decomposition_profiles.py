#=====================================================================#
# Decomposition timing profiles
#=====================================================================#
"""
Benchmark the vector (rank-1) and matrix (rank-2) decomposition
operators across the three inner-product backends:
    - numerical tensor-product quadrature
    - symbolic (SymPy) integration
    - closed-form analytic integration

Results are written as a CSV table plus simple bar plots comparing the
average runtime of each backend per basis construction.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pspace.core import (
    BasisFunctionType,
    CoordinateFactory,
    CoordinateSystem,
    InnerProductMode,
    PolyFunction,
)

# Allow reproducible randomness
DEFAULT_SEED = 2025


@dataclass
class ProfileConfig:
    seed: int = DEFAULT_SEED
    max_degree: int = 2
    num_coords: int = 3
    trials: int = 3
    sparse: bool = True
    output_dir: str = "profiles/output"
    enable_plots: bool = True


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _build_coordinate_system(
    basis_type: BasisFunctionType,
    num_coords: int,
    max_degree: int,
    rng: np.random.Generator,
) -> CoordinateSystem:
    """
    Construct a CoordinateSystem with randomly sampled coordinates
    (Normal, Uniform, Exponential).
    """
    cf = CoordinateFactory()
    cs = CoordinateSystem(basis_type)

    for idx in range(num_coords):
        coord_id = cf.newCoordinateID()
        name = f"y{idx}"
        deg = int(rng.integers(1, max_degree + 1))
        dist_choice = rng.choice(["normal", "uniform", "exponential"])

        if dist_choice == "normal":
            mu = float(rng.uniform(-2.0, 2.0))
            sigma = float(rng.uniform(0.5, 2.0))
            coord = cf.createNormalCoordinate(
                coord_id, name, dict(mu=mu, sigma=sigma), deg
            )
        elif dist_choice == "uniform":
            a = float(rng.uniform(-3.0, 1.0))
            b = float(a + rng.uniform(1.0, 4.0))
            coord = cf.createUniformCoordinate(
                coord_id, name, dict(a=a, b=b), deg
            )
        else:
            mu = float(rng.uniform(0.0, 2.0))
            beta = float(rng.uniform(0.5, 2.5))
            coord = cf.createExponentialCoordinate(
                coord_id, name, dict(mu=mu, beta=beta), deg
            )

        cs.addCoordinateAxis(coord)

    cs.initialize()
    return cs


def _random_polynomial(
    cs: CoordinateSystem,
    rng: np.random.Generator,
    max_degree: int,
    max_terms: int = 6,
) -> PolyFunction:
    """
    Build a random polynomial whose degrees respect the coordinate limits.
    """
    coord_ids = list(cs.coordinates.keys())
    terms: List[tuple[float, Counter]] = []

    # constant term
    terms.append((float(rng.uniform(-2.0, 2.0)), Counter()))

    for _ in range(max_terms - 1):
        coeff = float(rng.uniform(-2.0, 2.0))
        degs = Counter()
        for cid in coord_ids:
            deg_cap = cs.coordinates[cid].degree
            deg = int(rng.integers(0, min(max_degree, deg_cap) + 1))
            if deg:
                degs[cid] = deg
        if degs:
            terms.append((coeff, degs))

    return PolyFunction(terms, coordinates=cs.coordinates)


def _time_call(func) -> float:
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def _profile_vector(
    cs: CoordinateSystem,
    poly: PolyFunction,
    mode: InnerProductMode,
    sparse: bool,
) -> float:
    return _time_call(lambda: cs.decompose(poly, sparse=sparse, mode=mode))


def _profile_matrix(
    cs: CoordinateSystem,
    poly: PolyFunction,
    mode: InnerProductMode,
    sparse: bool,
) -> float:
    return _time_call(
        lambda: cs.decompose_matrix(poly, sparse=sparse, symmetric=True, mode=mode)
    )


def run_profiles(cfg: ProfileConfig) -> List[dict]:
    rng = _rng(cfg.seed)
    results: List[dict] = []

    ranks = ["vector", "matrix"]
    modes: Sequence[InnerProductMode] = [
        InnerProductMode.NUMERICAL,
        InnerProductMode.SYMBOLIC,
        InnerProductMode.ANALYTIC,
    ]
    basis_types = [
        BasisFunctionType.TENSOR_DEGREE,
        BasisFunctionType.TOTAL_DEGREE,
    ]

    for basis in basis_types:
        cs = _build_coordinate_system(
            basis,
            num_coords=cfg.num_coords,
            max_degree=cfg.max_degree,
            rng=rng,
        )
        poly = _random_polynomial(cs, rng, max_degree=cfg.max_degree)
        nbasis = cs.getNumBasisFunctions()

        for rank in ranks:
            for mode in modes:
                for trial in range(cfg.trials):
                    if rank == "vector":
                        elapsed = _profile_vector(cs, poly, mode, cfg.sparse)
                    else:
                        elapsed = _profile_matrix(cs, poly, mode, cfg.sparse)

                    results.append(
                        {
                            "rank": rank,
                            "basis_type": basis.name,
                            "mode": mode.value,
                            "trial": trial,
                            "elapsed": elapsed,
                            "num_coords": cfg.num_coords,
                            "max_degree": cfg.max_degree,
                            "nbasis": nbasis,
                            "sparse": cfg.sparse,
                            "seed": cfg.seed,
                        }
                    )
    return results


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_results_csv(results: Iterable[dict], path: str) -> None:
    results = list(results)
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def plot_results(results: Sequence[dict], output_dir: str) -> None:
    if not results:
        return

    grouped = defaultdict(list)
    for row in results:
        key = (row["rank"], row["basis_type"], row["mode"])
        grouped[key].append(row["elapsed"])

    ranks = sorted({row["rank"] for row in results})
    basis_types = sorted({row["basis_type"] for row in results})
    modes = [mode.value for mode in InnerProductMode]

    xaxis = np.arange(len(basis_types))
    width = 0.25

    for rank in ranks:
        fig, ax = plt.subplots(figsize=(8, 4))
        for idx, mode in enumerate(modes):
            averages = []
            for basis in basis_types:
                samples = grouped.get((rank, basis, mode), [])
                avg = float(np.mean(samples)) if samples else np.nan
                averages.append(avg)
            offset = (idx - 1) * width
            ax.bar(xaxis + offset, averages, width=width, label=mode.capitalize())

        ax.set_xticks(xaxis)
        ax.set_xticklabels(basis_types)
        ax.set_ylabel("Elapsed time (s)")
        ax.set_title(f"{rank.capitalize()} decomposition timing")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{rank}_timings.png"))
        plt.close(fig)


def parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark decomposition operators across inner-product modes."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--num-coords", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense assembly (sparse=False) during profiling.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only CSV output).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiles/output",
        help="Directory where CSV and plots are stored.",
    )
    args = parser.parse_args()
    return ProfileConfig(
        seed=args.seed,
        max_degree=args.max_degree,
        num_coords=args.num_coords,
        trials=args.trials,
        sparse=not args.dense,
        output_dir=args.output_dir,
        enable_plots=not args.no_plots,
    )


def main() -> None:
    cfg = parse_args()
    _ensure_dir(cfg.output_dir)

    results = run_profiles(cfg)
    csv_path = os.path.join(cfg.output_dir, "decomposition_timings.csv")
    write_results_csv(results, csv_path)

    if cfg.enable_plots:
        plot_results(results, cfg.output_dir)

    print(f"Stored {len(results)} measurements in {csv_path}")
    if cfg.enable_plots:
        print(f"Plots saved under {cfg.output_dir}")


if __name__ == "__main__":
    main()
