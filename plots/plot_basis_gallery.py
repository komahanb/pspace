from __future__ import annotations

import argparse
import os
from collections import Counter
from itertools import combinations
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from pspace.numeric import BasisFunctionType, DistributionType
from pspace.plotter import NumericCoordinateSystem as PlottingNumericCoordinateSystem

try:  # pragma: no cover
    from plots.helpers import build_coordinate_systems, random_polynomial, rng
except ImportError:  # pragma: no cover
    from helpers import build_coordinate_systems, random_polynomial, rng


DEFAULT_SEED = 2025


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def basis_type_from_string(name: str) -> BasisFunctionType:
    lookup = {
        "tensor": BasisFunctionType.TENSOR_DEGREE,
        "total": BasisFunctionType.TOTAL_DEGREE,
    }
    key = name.strip().lower()
    if key in lookup:
        return lookup[key]
    return BasisFunctionType[name.upper()]


def coordinate_range(coord, num_points: int = 200) -> Tuple[np.ndarray, Tuple[float, float]]:
    dist = coord.distribution
    if dist == DistributionType.UNIFORM:
        a = float(coord.dist_coords["a"])
        b = float(coord.dist_coords["b"])
    elif dist == DistributionType.NORMAL:
        mu = float(coord.dist_coords["mu"])
        sigma = float(coord.dist_coords["sigma"])
        a = mu - 3.0 * sigma
        b = mu + 3.0 * sigma
    elif dist == DistributionType.EXPONENTIAL:
        mu = float(coord.dist_coords["mu"])
        beta = float(coord.dist_coords["beta"])
        a = mu
        b = mu + 6.0 * beta
    else:
        domain = coord.domain()
        a = float(domain[0])
        b = float(domain[1])
    xs = np.linspace(a, b, num_points)
    return xs, (a, b)


def plot_axis_basis(output_dir: str, plotting_cs: PlottingNumericCoordinateSystem) -> None:
    for cid, coord in plotting_cs.coordinates.items():
        xs, _ = coordinate_range(coord)
        fig, ax = plt.subplots()
        for deg in range(0, coord.degree + 1):
            values = [coord.psi_y(float(x), deg) for x in xs]
            ax.plot(xs, values, label=f"ψ_{deg}")
        ax.set_title(f"Axis {coord.name} basis functions")
        ax.set_xlabel(coord.name)
        ax.set_ylabel("ψ")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"axis_{cid}_basis.png"))
        plt.close(fig)


def plot_pairwise_basis(output_dir: str, plotting_cs: PlottingNumericCoordinateSystem, max_degree: int = 3) -> None:
    coord_items = list(plotting_cs.coordinates.items())
    for (cid_i, coord_i), (cid_j, coord_j) in combinations(coord_items, 2):
        xs, _ = coordinate_range(coord_i, 100)
        ys, _ = coordinate_range(coord_j, 100)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        for deg_i in range(1, min(max_degree, coord_i.degree) + 1):
            for deg_j in range(1, min(max_degree, coord_j.degree) + 1):
                Z = np.array(
                    [
                        [
                            coord_i.psi_y(float(x), deg_i) * coord_j.psi_y(float(y), deg_j)
                            for y in ys
                        ]
                        for x in xs
                    ]
                )
                fig, ax = plt.subplots()
                im = ax.imshow(
                    Z,
                    extent=[ys[0], ys[-1], xs[0], xs[-1]],
                    origin="lower",
                    cmap="coolwarm",
                )
                ax.set_xlabel(coord_j.name)
                ax.set_ylabel(coord_i.name)
                ax.set_title(f"ψ_{deg_i}({coord_i.name}) ψ_{deg_j}({coord_j.name})")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                filename = f"pair_{cid_i}_{cid_j}_deg_{deg_i}_{deg_j}.png"
                fig.savefig(os.path.join(output_dir, filename))
                plt.close(fig)


def plot_quadrature(output_dir: str, plotting_cs: PlottingNumericCoordinateSystem) -> None:
    degrees = Counter({cid: coord.degree for cid, coord in plotting_cs.coordinates.items()})
    plotting_cs.build_quadrature(degrees)
    fig = plotting_cs.get_last_figure()
    if fig is not None:
        fig.savefig(os.path.join(output_dir, "quadrature.png"))
        plt.close(fig)


def plot_polynomial_surfaces(
    output_dir: str,
    numeric_cs,
    poly: PolyFunction,
) -> None:
    coord_items = list(numeric_cs.coordinates.items())
    if not coord_items:
        return

    # One-dimensional plots
    for cid, coord in coord_items:
        xs, _ = coordinate_range(coord)
        values = [poly({cid: float(x)}) for x in xs]
        fig, ax = plt.subplots()
        ax.plot(xs, values)
        ax.set_title(f"PolyFunction along axis {coord.name}")
        ax.set_xlabel(coord.name)
        ax.set_ylabel("f(y)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"poly_axis_{cid}.png"))
        plt.close(fig)

    # Two-dimensional slices
    if len(coord_items) >= 2:
        for (cid_i, coord_i), (cid_j, coord_j) in combinations(coord_items, 2):
            xs, _ = coordinate_range(coord_i, 80)
            ys, _ = coordinate_range(coord_j, 80)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            Z = np.zeros_like(X)
            for ii in range(X.shape[0]):
                for jj in range(X.shape[1]):
                    point = {cid_i: float(X[ii, jj]), cid_j: float(Y[ii, jj])}
                    Z[ii, jj] = poly(point)
            fig, ax = plt.subplots()
            contour = ax.contourf(
                X,
                Y,
                Z,
                levels=20,
                cmap="viridis",
            )
            ax.set_xlabel(coord_i.name)
            ax.set_ylabel(coord_j.name)
            ax.set_title(f"PolyFunction contour: {coord_i.name} vs {coord_j.name}")
            fig.colorbar(contour, ax=ax)
            fig.tight_layout()
            filename = f"poly_contour_{cid_i}_{cid_j}.png"
            fig.savefig(os.path.join(output_dir, filename))
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate basis and polynomial visualisations for a NumericCoordinateSystem."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--basis-type", type=str, default="tensor", help="tensor or total")
    parser.add_argument("--num-coords", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="plots/output/basis_gallery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    generator = rng(args.seed)
    basis_type = basis_type_from_string(args.basis_type)

    numeric_cs, plotting_cs = build_coordinate_systems(
        basis_type,
        num_coords=args.num_coords,
        max_degree=args.max_degree,
        generator=generator,
    )

    # Basis plots
    plot_axis_basis(args.output_dir, plotting_cs)
    plot_pairwise_basis(args.output_dir, plotting_cs)

    # Quadrature visualisation
    plot_quadrature(args.output_dir, plotting_cs)

    # Polynomial surfaces
    polynomial = random_polynomial(numeric_cs, generator, max_degree=args.max_degree)
    plot_polynomial_surfaces(args.output_dir, numeric_cs, polynomial)

    print(f"Gallery written to {args.output_dir}")


if __name__ == "__main__":
    main()
