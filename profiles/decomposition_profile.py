#=====================================================================#
# Decomposition timing profile script (uses pspace.profile)
#=====================================================================#

from __future__ import annotations

import argparse
import csv
import os
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
from pspace.profile import CoordinateSystemProfiler

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
    cf = CoordinateFactory()
    cs = CoordinateSystem(basis_type)

    for idx in range(num_coords):
        coord_id = cf.newCoordinateID()
        name = f"y{idx}"
        deg = int(rng.integers(1, max_degree + 1))
        dist = rng.choice(["normal", "uniform", "exponential"])

        if dist == "normal":
            mu = float(rng.uniform(-2.0, 2.0))
            sigma = float(rng.uniform(0.5, 2.0))
            coord = cf.createNormalCoordinate(
                coord_id, name, dict(mu=mu, sigma=sigma), deg
            )
        elif dist == "uniform":
            a = float(rng.uniform(-3.0, 1.0))
            b = float(a + rng.uniform(1.0, 4.0))
            coord = cf.createUniformCoordinate(coord_id, name, dict(a=a, b=b), deg)
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
    coord_ids = list(cs.coordinates.keys())
    terms: List[tuple[float, Counter]] = []
    terms.append((float(rng.uniform(-2.0, 2.0)), Counter()))

    for _ in range(max_terms - 1):
        coeff = float(rng.uniform(-2.0, 2.0))
        degs = Counter()
        for cid in coord_ids:
            cap = cs.coordinates[cid].degree
            deg = int(rng.integers(0, min(max_degree, cap) + 1))
            if deg:
                degs[cid] = deg
        if degs:
            terms.append((coeff, degs))

    return PolyFunction(terms, coordinates=cs.coordinates)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(rows: Iterable[dict], path: str) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(results: Sequence[dict], output_dir: str) -> None:
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


def run_profiles(cfg: ProfileConfig) -> List[dict]:
    rng = _rng(cfg.seed)
    results: List[dict] = []
    modes: Sequence[InnerProductMode] = list(InnerProductMode)

    for basis in (
        BasisFunctionType.TENSOR_DEGREE,
        BasisFunctionType.TOTAL_DEGREE,
    ):
        cs = _build_coordinate_system(
            basis,
            num_coords=cfg.num_coords,
            max_degree=cfg.max_degree,
            rng=rng,
        )
        profiler = CoordinateSystemProfiler(cs)
        poly = _random_polynomial(cs, rng, max_degree=cfg.max_degree)
        nbasis = cs.getNumBasisFunctions()

        for rank in ("vector", "matrix"):
            for trial in range(cfg.trials):
                stats = profiler.benchmark(
                    function=poly,
                    rank=rank,
                    modes=modes,
                    sparse=cfg.sparse,
                    symmetric=True,
                )
                for mode, result in stats.items():
                    results.append(
                        {
                            "rank": rank,
                            "basis_type": basis.name,
                            "mode": mode.value,
                            "trial": trial,
                            "elapsed": result.elapsed,
                            "sparse": result.sparse,
                            "num_coords": cfg.num_coords,
                            "max_degree": cfg.max_degree,
                            "nbasis": nbasis,
                            "seed": cfg.seed,
                        }
                    )
    return results


def parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark decomposition operators across inner-product modes."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-degree", type=int, default=2)
    parser.add_argument("--num-coords", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--dense", action="store_true", help="Use dense assembly (sparse=False)."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only write CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiles/output",
        help="Directory for CSV/plot outputs.",
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
    _write_csv(results, csv_path)

    if cfg.enable_plots:
        _plot(results, cfg.output_dir)

    print(f"Stored {len(results)} measurements in {csv_path}")
    if cfg.enable_plots:
        print(f"Plots saved under {cfg.output_dir}")


if __name__ == "__main__":
    main()
