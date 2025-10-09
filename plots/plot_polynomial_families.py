from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from pspace.orthogonal_polynomials import (
    hermite as hermite_standard,
    laguerre as laguerre_standard,
    legendre as legendre_standard,
    unit_hermite,
    unit_laguerre,
    unit_legendre,
)


def _standardize_styles() -> None:
    try:
        plt.style.use("seaborn-colorblind")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.autolayout": True,
            "lines.linewidth": 2.5,
            "lines.markersize": 6,
        }
    )


def _plot_family(
    x: np.ndarray,
    evaluator: Iterable[float],
    label: str,
    axis,
    color: str,
) -> None:
    axis.plot(x, evaluator, "-", marker="o", label=label, color=color, markevery=6)


def _decorate(axis, ylabel: str, xlabel: str = "z") -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


def _write(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path("plots/output")
    output_dir = Path(output_dir)

    _standardize_styles()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Hermite (physicists') polynomials
    z = np.linspace(-2.0, 2.0, 200)
    fig, ax = plt.subplots()
    for degree in range(6):
        _plot_family(z, hermite_standard(z, degree), f"$H_{degree+1}$", ax, colors[degree % len(colors)])
    _decorate(ax, "$H(z)$")
    _write(fig, output_dir / "hermite-polynomials.pdf")

    # Orthonormal Hermite
    fig, ax = plt.subplots()
    for degree in range(6):
        _plot_family(z, unit_hermite(z, degree), f"$\\widehat{{H_{degree+1}}}$", ax, colors[degree % len(colors)])
    _decorate(ax, "$\\widehat{H}(z)$")
    _write(fig, output_dir / "unit-hermite-polynomials.pdf")

    # Legendre on [0,1]
    z = np.linspace(0.0, 1.0, 200)
    fig, ax = plt.subplots()
    for degree in range(6):
        _plot_family(z, legendre_standard(z, degree), f"$P_{degree+1}$", ax, colors[degree % len(colors)])
    _decorate(ax, "$P(z)$")
    _write(fig, output_dir / "legendre-polynomials.pdf")

    # Orthonormal Legendre
    fig, ax = plt.subplots()
    for degree in range(6):
        _plot_family(z, unit_legendre(z, degree), f"$\\widehat{{P_{degree+1}}}$", ax, colors[degree % len(colors)])
    _decorate(ax, "$\\widehat{P}(z)$")
    _write(fig, output_dir / "unit-legendre-polynomials.pdf")

    # Laguerre on [0,4]
    z = np.linspace(0.0, 4.0, 200)
    fig, ax = plt.subplots()
    for degree in range(6):
        evaluator = unit_laguerre(z, degree) * math.factorial(degree)
        _plot_family(z, evaluator, f"$L_{degree+1}$", ax, colors[degree % len(colors)])
    _decorate(ax, "$L(z)$")
    _write(fig, output_dir / "laguerre-polynomials.pdf")

    # Orthonormal Laguerre
    fig, ax = plt.subplots()
    for degree in range(6):
        _plot_family(z, unit_laguerre(z, degree), f"$\\widehat{{L_{degree+1}}}$", ax, colors[degree % len(colors)])
    _decorate(ax, "$\\widehat{L}(z)$")
    _write(fig, output_dir / "unit-laguerre-polynomials.pdf")


if __name__ == "__main__":
    main()
