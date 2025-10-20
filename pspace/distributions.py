from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import sympy as sp

from .core import DistributionType
from .orthogonal_polynomials import unit_hermite, unit_legendre, unit_laguerre


# --------------------------------------------------------------------------- #
# Numeric distributions                                                       #
# --------------------------------------------------------------------------- #


class NumericDistribution(ABC):
    kind: DistributionType

    def __init__(self, **params: float) -> None:
        self.params = {key: float(value) for key, value in params.items()}

    @abstractmethod
    def domain(self) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def physical_to_standard(self, yscalar: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def quadrature_to_physical(self, xscalar: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def standard_to_physical(self, zscalar: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def psi_z(self, zscalar: float, degree: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def gaussian_quadrature(self, degree: int):
        raise NotImplementedError

    def weight(self):
        return None


class NumericNormalDistribution(NumericDistribution):
    kind = DistributionType.NORMAL

    def __init__(self, mu: float, sigma: float) -> None:
        super().__init__(mu=mu, sigma=sigma)

    def domain(self) -> Tuple[float, float]:
        return -math.inf, math.inf

    def physical_to_standard(self, yscalar: float) -> float:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return (yscalar - mu) / sigma

    def quadrature_to_physical(self, xscalar: float) -> float:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return mu + sigma * math.sqrt(2.0) * xscalar

    def standard_to_physical(self, zscalar: float) -> float:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return mu + sigma * zscalar

    def psi_z(self, zscalar: float, degree: int) -> float:
        return unit_hermite(zscalar, degree)

    def gaussian_quadrature(self, degree: int):
        from .stochastic_utils import minnum_quadrature_points

        npts = minnum_quadrature_points(degree)
        x, w = np.polynomial.hermite.hermgauss(npts)
        w    = w / math.sqrt(math.pi)
        return x, w


class NumericUniformDistribution(NumericDistribution):
    kind = DistributionType.UNIFORM

    def __init__(self, a: float, b: float) -> None:
        super().__init__(a=a, b=b)

    def domain(self) -> Tuple[float, float]:
        return self.params["a"], self.params["b"]

    def physical_to_standard(self, yscalar: float) -> float:
        a = self.params["a"]
        b = self.params["b"]
        return (yscalar - a) / (b - a)

    def quadrature_to_physical(self, xscalar: float) -> float:
        a = self.params["a"]
        b = self.params["b"]
        return (b - a) * xscalar + a

    def standard_to_physical(self, zscalar: float) -> float:
        a = self.params["a"]
        b = self.params["b"]
        return (b - a) * zscalar + a

    def psi_z(self, zscalar: float, degree: int) -> float:
        return unit_legendre(zscalar, degree)

    def gaussian_quadrature(self, degree: int):
        from .stochastic_utils import minnum_quadrature_points

        npts       = minnum_quadrature_points(degree)
        xi, w      = np.polynomial.legendre.leggauss(npts)
        x_shifted  = 0.5 * (xi + 1.0)
        w_shifted  = 0.5 * w
        return x_shifted, w_shifted


class NumericExponentialDistribution(NumericDistribution):
    kind = DistributionType.EXPONENTIAL

    def __init__(self, mu: float, beta: float) -> None:
        super().__init__(mu=mu, beta=beta)

    def domain(self) -> Tuple[float, float]:
        return self.params["mu"], math.inf

    def physical_to_standard(self, yscalar: float) -> float:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return (yscalar - mu) / beta

    def quadrature_to_physical(self, xscalar: float) -> float:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return mu + beta * xscalar

    def standard_to_physical(self, zscalar: float) -> float:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return mu + beta * zscalar

    def psi_z(self, zscalar: float, degree: int) -> float:
        return unit_laguerre(zscalar, degree)

    def gaussian_quadrature(self, degree: int):
        from .stochastic_utils import minnum_quadrature_points

        npts = minnum_quadrature_points(degree)
        x, w = np.polynomial.laguerre.laggauss(npts)
        return x, w


# --------------------------------------------------------------------------- #
# Symbolic distributions                                                     #
# --------------------------------------------------------------------------- #


class SymbolicDistribution(ABC):
    kind: DistributionType

    def __init__(self, **params: Any) -> None:
        self.params = {key: sp.sympify(value) for key, value in params.items()}

    @abstractmethod
    def domain(self) -> Tuple[sp.Expr, sp.Expr]:
        raise NotImplementedError

    @abstractmethod
    def weight(self, symbol: sp.Symbol) -> sp.Expr:
        raise NotImplementedError

    @abstractmethod
    def physical_to_standard(self, value: sp.Expr) -> sp.Expr:
        raise NotImplementedError

    @abstractmethod
    def quadrature_to_physical(self, value: sp.Expr) -> sp.Expr:
        raise NotImplementedError

    @abstractmethod
    def standard_to_physical(self, value: sp.Expr) -> sp.Expr:
        raise NotImplementedError

    @abstractmethod
    def psi_z(self, value: sp.Expr, degree: int) -> sp.Expr:
        raise NotImplementedError

    def gaussian_quadrature(self, degree: int):
        raise NotImplementedError("Symbolic distributions do not provide quadrature nodes.")


class SymbolicNormalDistribution(SymbolicDistribution):
    kind = DistributionType.NORMAL

    def __init__(self, mu: Any, sigma: Any) -> None:
        super().__init__(mu=mu, sigma=sigma)

    def domain(self) -> Tuple[sp.Expr, sp.Expr]:
        return -sp.oo, sp.oo

    def weight(self, symbol: sp.Symbol) -> sp.Expr:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return sp.exp(-((symbol - mu) ** 2) / (2 * sigma**2)) / (sp.sqrt(2 * sp.pi) * sigma)

    def physical_to_standard(self, value: sp.Expr) -> sp.Expr:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return (value - mu) / sigma

    def quadrature_to_physical(self, value: sp.Expr) -> sp.Expr:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return mu + sigma * sp.sqrt(2) * value

    def standard_to_physical(self, value: sp.Expr) -> sp.Expr:
        mu    = self.params["mu"]
        sigma = self.params["sigma"]
        return mu + sigma * value

    def psi_z(self, value: sp.Expr, degree: int) -> sp.Expr:
        if degree == 0:
            return sp.Integer(1)
        if degree == 1:
            return value
        prev = sp.Integer(1)
        curr = value
        for n in range(2, degree + 1):
            nxt = value * curr - (n - 1) * prev
            prev, curr = curr, nxt
        return curr / sp.sqrt(sp.factorial(degree))


class SymbolicUniformDistribution(SymbolicDistribution):
    kind = DistributionType.UNIFORM

    def __init__(self, a: Any, b: Any) -> None:
        super().__init__(a=a, b=b)

    def domain(self) -> Tuple[sp.Expr, sp.Expr]:
        return self.params["a"], self.params["b"]

    def weight(self, symbol: sp.Symbol) -> sp.Expr:
        a = self.params["a"]
        b = self.params["b"]
        return sp.Integer(1) / (b - a)

    def physical_to_standard(self, value: sp.Expr) -> sp.Expr:
        a = self.params["a"]
        b = self.params["b"]
        return (value - a) / (b - a)

    def quadrature_to_physical(self, value: sp.Expr) -> sp.Expr:
        a = self.params["a"]
        b = self.params["b"]
        return (b - a) * value + a

    def standard_to_physical(self, value: sp.Expr) -> sp.Expr:
        a = self.params["a"]
        b = self.params["b"]
        return (b - a) * value + a

    def psi_z(self, value: sp.Expr, degree: int) -> sp.Expr:
        poly = sp.legendre(degree, 2 * value - 1)
        return sp.sqrt(2 * degree + 1) * poly


class SymbolicExponentialDistribution(SymbolicDistribution):
    kind = DistributionType.EXPONENTIAL

    def __init__(self, mu: Any, beta: Any) -> None:
        super().__init__(mu=mu, beta=beta)

    def domain(self) -> Tuple[sp.Expr, sp.Expr]:
        return self.params["mu"], sp.oo

    def weight(self, symbol: sp.Symbol) -> sp.Expr:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return sp.exp(-(symbol - mu) / beta) / beta

    def physical_to_standard(self, value: sp.Expr) -> sp.Expr:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return (value - mu) / beta

    def quadrature_to_physical(self, value: sp.Expr) -> sp.Expr:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return mu + beta * value

    def standard_to_physical(self, value: sp.Expr) -> sp.Expr:
        mu   = self.params["mu"]
        beta = self.params["beta"]
        return mu + beta * value

    def psi_z(self, value: sp.Expr, degree: int) -> sp.Expr:
        return sp.laguerre(degree, value)


# --------------------------------------------------------------------------- #
# Factory helpers                                                            #
# --------------------------------------------------------------------------- #

NUMERIC_DISTRIBUTIONS: Dict[DistributionType, Any] = {
    DistributionType.NORMAL: NumericNormalDistribution,
    DistributionType.UNIFORM: NumericUniformDistribution,
    DistributionType.EXPONENTIAL: NumericExponentialDistribution,
}

SYMBOLIC_DISTRIBUTIONS: Dict[DistributionType, Any] = {
    DistributionType.NORMAL: SymbolicNormalDistribution,
    DistributionType.UNIFORM: SymbolicUniformDistribution,
    DistributionType.EXPONENTIAL: SymbolicExponentialDistribution,
}

