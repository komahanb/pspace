#!/usr/bin/env python

#=====================================================================#
# சாத்தியவியல், இடவியல், காலவியல் பகுதி வேறுபாட்டுச்
# சமன்பாடுகளுக்கான கணிதப் பகுப்பாய்வுத் தொகுதி

# ஆசிரியர் : கோமகன் பூபதி (komahan@gatech.edu)
#————————————————————————————————————————————————————————————————————-#
# MATHEMATICAL ANALYSIS MODULE FOR PROBABILISTIC-SPATIO-TEMPORAL
# PARTIAL DIFFERENTIAL EQUATIONS
#
# Author    : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

# External modules
import math
import sympy as sp
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from abc         import ABC, abstractmethod
from collections import Counter
from enum        import Enum
from itertools   import product

# Local modules
from .stochastic_utils import (minnum_quadrature_points,
                               generate_basis_tensor_degree,
                               generate_basis_total_degree,
                               sum_degrees,
                               safe_zero_degrees,
                               sum_degrees_union_matrix,
                               sum_degrees_union_vector)

from .orthogonal_polynomials import unit_hermite
from .orthogonal_polynomials import unit_legendre
from .orthogonal_polynomials import unit_laguerre

#=====================================================================#
# Enums
#=====================================================================#

class CoordinateType(Enum):
    """
    DOMAIN TYPES
    """
    PROBABILISTIC = 1
    SPATIAL       = 2
    TEMPORAL      = 3


class DistributionType(Enum):
    """
    GEOMETRY: DENSITY DISTRIBUTION
    """
    NORMAL      = 0
    UNIFORM     = 1
    EXPONENTIAL = 2
    POISSON     = 3
    BINORMAL    = 4


class BasisFunctionType(Enum):
    """
    VECTOR-SPACE CONSTRUCTION METHODS
    """
    TENSOR_DEGREE   = 0
    TOTAL_DEGREE    = 1
    ADAPTIVE_DEGREE = 2

#=====================================================================#
# PointSampler hierarchy
#=====================================================================#

class PointSampler(ABC):
    """
    Abstract interface for sampling the parameter space.

    Iteration protocol:  for point, weight in sampler: ...
      point  : dict {cid: float}  — parameter values
      weight : float              — integration weight (sum to 1)

    Concrete implementations:
      QuadratureSampler  — tensor-product Gauss rule, exact for polynomials
      MonteCarloSampler  — pseudo-random draws, uniform weight 1/N
    """

    @abstractmethod
    def __iter__(self):
        """Yield (point: dict{cid: float}, weight: float) pairs."""

    @property
    @abstractmethod
    def n_points(self):
        """Total number of sample points."""

    def integrate(self, fn):
        """
        Compute E[fn] = Σ fn(point) * weight over all samples.
        fn : callable({cid: float}) -> float
        """
        total = 0.0
        for point, weight in self:
            total += fn(point) * weight
        return total


class QuadratureSampler(PointSampler):
    """
    Tensor-product Gauss quadrature sampler.
    Points and weights are exact for polynomials up to degree 2*npts-1 per axis.

    Parameters
    ----------
    cs     : CoordinateSystem
    degree : int
        Polynomial degree to integrate exactly (per axis).
    """

    def __init__(self, cs, degree):
        deg_counter = Counter({cid: degree for cid in cs.param_ids})
        self._qmap  = cs.build_quadrature(deg_counter)

    def __iter__(self):
        for q in self._qmap.values():
            yield {cid: float(v) for cid, v in q['Y'].items()}, float(q['W'])

    @property
    def n_points(self):
        return len(self._qmap)


class MonteCarloSampler(PointSampler):
    """
    Pseudo-random Monte Carlo sampler.
    Each point carries equal weight 1/N; E[f] ≈ (1/N) Σ f(p_i).

    Parameters
    ----------
    cs       : CoordinateSystem
    n        : int
        Number of random samples.
    rng_seed : int
        Seed for reproducibility.
    """

    def __init__(self, cs, n, rng_seed=42):
        rng           = np.random.default_rng(rng_seed)
        self._samples = cs.mc_samples(n, rng)   # {cid: ndarray(n)}
        self._cids    = list(self._samples.keys())
        self._n       = n
        self._weight  = 1.0 / n

    def __iter__(self):
        for i in range(self._n):
            point = {cid: float(self._samples[cid][i]) for cid in self._cids}
            yield point, self._weight

    @property
    def n_points(self):
        return self._n

    @property
    def raw_samples(self):
        """Raw sample arrays {cid: ndarray} for vectorized physics evaluation."""
        return self._samples


class PolyFunction:
    def __init__(self, terms):
        """
        terms : list of (coeff, Counter) pairs
        Example:
            [
              (3, Counter()),              # constant
              (3, Counter({0:1})),         # 3*y0
              (3, Counter({0:2, 1:1}))     # 3*y0^2 * y1
            ]
        """
        self._terms = []
        self._degrees = []          # list of Counters
        self._max_degrees = Counter()

        for t in terms:
            if isinstance(t, tuple) and isinstance(t[1], Counter):
                coeff, degs = t
                self._terms.append((coeff, degs))
                self._degrees.append(degs)
                for k, v in degs.items():
                    self._max_degrees[k] = max(self._max_degrees.get(k, 0), v)
            else:
                raise TypeError(f"Invalid term format: {t!r}")

    @property
    def terms(self):
        return self._terms

    @property
    def degrees(self):
        """List[Counter]: degree structure per monomial"""
        return self._degrees

    @property
    def max_degrees(self):
        """Counter: max degree per axis (union of monomials)"""
        return self._max_degrees

    def __call__(self, Y):
        total = 0.0
        for coeff, degs in self._terms:
            mon = coeff
            for cid, d in degs.items():
                mon *= Y[cid] ** d
            total += mon
        return total

    def derivative(self, cid):
        """
        Differentiate with respect to coordinate axis `cid`.

        For each monomial  coeff * prod_k(y_k^d_k):
          d/dy_i = coeff * d_i * y_i^(d_i-1) * prod_{k≠i}(y_k^d_k)

        Returns a new PolyFunction (zero constant if no terms survive).
        Pure arithmetic on stored coefficients and degree counters — no sympy.
        """
        new_terms = []
        for coeff, degs in self._terms:
            d = degs.get(cid, 0)
            if d > 0:
                new_degs = Counter(degs)
                new_degs[cid] -= 1
                if new_degs[cid] == 0:
                    del new_degs[cid]
                new_terms.append((coeff * d, new_degs))
        if not new_terms:
            return PolyFunction([(0.0, Counter())])
        return PolyFunction(new_terms)

    def __repr__(self):
        return f"PolyFunction({self._terms})"

class OrthoPolyFunction:
    """
    Polynomial function expressed in orthonormal basis (Legendre, Hermite, etc.)
    Works with CoordinateSystem and its Coordinate axes.
    """

    def __init__(self, terms, coordinates):
        """
        Parameters
        ----------
        terms : list[(coeff: float, degs: Counter)]
            List of terms (basis coefficient, degree counter per coordinate).
        coordinates : dict[int, Coordinate]
            Dictionary of coordinate objects {cid: Coordinate}.
        """
        self._terms = terms
        self._coords = coordinates

    def __call__(self, Y):
        """
        Evaluate function at point Y in computational coordinates.
        Y : dict {cid: float}
        """
        total = 0.0
        for coeff, degs in self._terms:
            term_val = coeff
            for cid, d in degs.items():
                # Evaluate orthogonal basis polynomial at Y[cid]
                term_val *= self._coords[cid].psi_y(Y[cid], d)
            total += term_val
        return total

    def toPolyFunction(self):
        """
        Expand OrthoPolyFunction into monomial PolyFunction
        using change-of-basis matrices (currently supports Legendre).
        """
        from numpy.polynomial import legendre as npleg
        from numpy.polynomial import polynomial as nppoly
        from .core import PolyFunction  # adjust import if needed

        monomial_terms = Counter()

        for coeff, degs in self._terms:
            # Build tensor product expansion for each coordinate
            expansions = []
            for cid, d in degs.items():
                coord = self._coords[cid]
                if coord.dist_type.name == "UNIFORM":  # Legendre basis
                    Pn = npleg.Legendre.basis(d)
                    poly = Pn.convert(kind=nppoly.Polynomial)
                    coeffs_power = np.array(poly.coef, dtype=float)
                    s = np.sqrt((2*d+1)/2.0)  # orthonormal scale
                    coeffs_power *= s
                    expansions.append((cid, coeffs_power))
                else:
                    raise NotImplementedError("toPolyFunction only supports Legendre for now")

            # Recursive tensor product accumulation
            def recurse(idx, running_coeff, running_degs):
                if idx == len(expansions):
                    monomial_terms[running_degs] += coeff * running_coeff
                    return
                cid, coeffs_power = expansions[idx]
                for p, val in enumerate(coeffs_power):
                    if abs(val) < 1e-15:
                        continue
                    recurse(idx+1, running_coeff*val,
                            running_degs + Counter({cid: p}))

            recurse(0, 1.0, Counter())

        # Build PolyFunction terms
        terms = [(float(val), degs) for degs, val in monomial_terms.items() if abs(val) > 1e-15]
        return PolyFunction(terms)

    def coeffs(self):
        """Return coefficients directly."""
        return {tuple(sorted(d.items())): c for c, d in self._terms}

    def __repr__(self):
        return f"OrthoPolyFunction({len(self._terms)} terms, basis=orthonormal)"

#=====================================================================#
# Coordinate Base Class
#=====================================================================#

class Coordinate(object):
    def __init__(self, coord_data):
        self.id           = coord_data['coord_id']
        self.name         = coord_data['coord_name']
        self.type         = coord_data['coord_type']
        self.distribution = coord_data['dist_type']
        self.degree       = coord_data['monomial_degree']
        self.symbol       = sp.Symbol(self.name)   # y
        self.rho          = None                   # rho(y)

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    def weight(self):
        """Return symbolic weight function ρ(y) attached to this coordinate."""
        return self.rho

    #-----------------------------------------------------------------#
    # VARIABLE TRANSFORMATIONS
    #-----------------------------------------------------------------#

    def physical_to_standard(self, yscalar):
        """Map physical y -> standard z"""
        raise NotImplementedError

    def quadrature_to_physical(self, xscalar):
        """Map quadrature x -> physical y"""
        raise NotImplementedError

    def standard_to_physical(self, zscalar):
        """Map standard z -> physical y"""
        raise NotImplementedError

    def quadrature_to_standard(self, xscalar):
        """Map quadrature x -> standard z"""
        return self.physical_to_standard(self.quadrature_to_physical(xscalar))

    #-----------------------------------------------------------------#
    # BASIS EVALUATIONS
    #-----------------------------------------------------------------#

    def psi_z(self, zscalar, degree):
        raise NotImplementedError

    def psi_y(self, yscalar, degree):
        z = self.physical_to_standard(yscalar)
        return self.psi_z(z, degree)

    def psi_x(self, xscalar, degree):
        z = self.quadrature_to_standard(xscalar)
        return self.psi_z(z, degree)

    #-----------------------------------------------------------------#
    # subclass provides a 1D Gauss rule for the needed degree
    #-----------------------------------------------------------------#

    def gaussian_quadrature(self, degree):
        """
        Return native Gauss rule (x_nodes, w_nodes) sized for `degree`.
        Subclass chooses the correct family and normalization.
        """
        raise NotImplementedError

    def getQuadraturePointsWeights(self, degree):
        x, w = self.gaussian_quadrature(degree)
        z    = self.quadrature_to_standard(x)
        y    = np.array([self.standard_to_physical(zz) for zz in z])
        return {'yq': y, 'zq': z, 'wq': w}

    #-----------------------------------------------------------------#
    # Distribution moments  (subclass implements)
    #-----------------------------------------------------------------#

    def mean(self):
        """Return the mean (expected value) of this coordinate's distribution."""
        raise NotImplementedError

    def variance(self):
        """Return the variance of this coordinate's distribution."""
        raise NotImplementedError

    def mc_samples(self, n, rng):
        """Draw n random samples from this coordinate's distribution."""
        raise NotImplementedError

#=====================================================================#
# Coordinate Implementations
#=====================================================================#

class NormalCoordinate(Coordinate):
    def __init__(self, pdata):
        super().__init__(pdata)
        mu               = sp.sympify(pdata['dist_coords']['mu'])
        sigma            = sp.sympify(pdata['dist_coords']['sigma'])
        self.dist_coords = {'mu': mu, 'sigma': sigma}
        self.rho         = sp.exp(-(self.symbol - mu)**2 / (2*sigma**2)) / (sp.sqrt(2*sp.pi) * sigma)

    def domain(self):
        return -sp.oo, sp.oo

    def physical_to_standard(self, yscalar):
        """Map physical y -> standard z"""
        mu, sigma = self.dist_coords['mu'], self.dist_coords['sigma']
        return (yscalar - mu) / sigma

    def quadrature_to_physical(self, xscalar):
        """Map quadrature x -> physical y"""
        mu, sigma = self.dist_coords['mu'], self.dist_coords['sigma']
        return mu + sigma * np.sqrt(2) * xscalar

    def standard_to_physical(self, zscalar):
        """Map standard z -> physical y"""
        mu, sigma = self.dist_coords['mu'], self.dist_coords['sigma']
        return mu + sigma * zscalar

    def psi_z(self, z, degree):
        return unit_hermite(z, degree)

    def gaussian_quadrature(self, degree):
        npts = minnum_quadrature_points(degree)
        x, w = np.polynomial.hermite.hermgauss(npts)
        w    = w / np.sqrt(np.pi)
        return x, w

    def mean(self):
        return float(self.dist_coords['mu'])

    def variance(self):
        sigma = float(self.dist_coords['sigma'])
        return sigma ** 2

    def mc_samples(self, n, rng):
        return rng.normal(self.mean(), float(self.dist_coords['sigma']), n)

class UniformCoordinate(Coordinate):
    def __init__(self, pdata):
        super().__init__(pdata)
        a                = sp.sympify(pdata['dist_coords']['a'])
        b                = sp.sympify(pdata['dist_coords']['b'])
        self.dist_coords = {'a': a, 'b': b}
        self.rho         = sp.Rational(1, b - a)

    def domain(self):
        return self.dist_coords['a'], self.dist_coords['b']

    def physical_to_standard(self, yscalar):
        a, b = self.dist_coords['a'], self.dist_coords['b']
        return (yscalar - a) / (b - a)

    def quadrature_to_physical(self, xscalar):
        a, b = self.dist_coords['a'], self.dist_coords['b']
        return (b - a) * xscalar + a

    def standard_to_physical(self, zscalar):
        a, b = self.dist_coords['a'], self.dist_coords['b']
        return (b - a) * zscalar + a

    def psi_z(self, z, degree):
        return unit_legendre(z, degree)

    def gaussian_quadrature(self, degree):
        npts       = minnum_quadrature_points(degree)
        xi, w      = np.polynomial.legendre.leggauss(npts)  # on [-1,1]
        x_shifted  = 0.5 * (xi + 1.0)                       # map to [0,1]
        w_shifted  = 0.5 * w
        return x_shifted, w_shifted

    def mean(self):
        a, b = float(self.dist_coords['a']), float(self.dist_coords['b'])
        return (a + b) / 2.0

    def variance(self):
        a, b = float(self.dist_coords['a']), float(self.dist_coords['b'])
        return (b - a) ** 2 / 12.0

    def mc_samples(self, n, rng):
        a, b = float(self.dist_coords['a']), float(self.dist_coords['b'])
        return rng.uniform(a, b, n)

class ExponentialCoordinate(Coordinate):
    def __init__(self, pdata):
        super().__init__(pdata)
        mu               = sp.sympify(pdata['dist_coords']['mu'])
        beta             = sp.sympify(pdata['dist_coords']['beta'])
        self.dist_coords = {'mu': mu, 'beta': beta}
        self.rho         = sp.exp(-(self.symbol - mu)/beta) / beta

    def domain(self):
        return self.dist_coords['mu'], sp.oo

    def physical_to_standard(self, yscalar):
        mu, beta = self.dist_coords['mu'], self.dist_coords['beta']
        return (yscalar - mu) / beta

    def quadrature_to_physical(self, xscalar):
        mu, beta = self.dist_coords['mu'], self.dist_coords['beta']
        return mu + beta * xscalar

    def standard_to_physical(self, zscalar):
        mu, beta = self.dist_coords['mu'], self.dist_coords['beta']
        return mu + beta * zscalar

    def psi_z(self, z, degree):
        return unit_laguerre(z, degree)

    def gaussian_quadrature(self, degree):
        npts = minnum_quadrature_points(degree)
        x, w = np.polynomial.laguerre.laggauss(npts)
        return x, w

    def mean(self):
        mu, beta = float(self.dist_coords['mu']), float(self.dist_coords['beta'])
        return mu + beta

    def variance(self):
        beta = float(self.dist_coords['beta'])
        return beta ** 2

    def mc_samples(self, n, rng):
        mu, beta = float(self.dist_coords['mu']), float(self.dist_coords['beta'])
        return mu + rng.exponential(beta, n)

#=====================================================================#
# Coordinate Factory
#=====================================================================#

class CoordinateFactory:
    def __init__(self):
        self.next_coord_id = 0
        return

    def newCoordinateID(self):
        pid                = self.next_coord_id
        self.next_coord_id = self.next_coord_id + 1
        return pid

    def createNormalCoordinate(self, coord_id, coord_name, dist_coords, max_monomial_dof):
        pdata                    = {}
        pdata['coord_id']        = coord_id
        pdata['coord_name']      = coord_name
        pdata['coord_type']      = CoordinateType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.NORMAL
        pdata['dist_coords']     = dist_coords
        pdata['monomial_degree'] = max_monomial_dof
        return NormalCoordinate(pdata)

    def createUniformCoordinate(self, coord_id, coord_name, dist_coords, max_monomial_dof):
        pdata                    = {}
        pdata['coord_id']        = coord_id
        pdata['coord_name']      = coord_name
        pdata['coord_type']      = CoordinateType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.UNIFORM
        pdata['dist_coords']     = dist_coords
        pdata['monomial_degree'] = max_monomial_dof
        return UniformCoordinate(pdata)

    def createExponentialCoordinate(self, coord_id, coord_name, dist_coords, max_monomial_dof):
        pdata                    = {}
        pdata['coord_id']        = coord_id
        pdata['coord_name']      = coord_name
        pdata['coord_type']      = CoordinateType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.EXPONENTIAL
        pdata['dist_coords']     = dist_coords
        pdata['monomial_degree'] = max_monomial_dof
        return ExponentialCoordinate(pdata)

#=====================================================================#
# Coordinate System
#=====================================================================#

class CoordinateSystem:
    """
    1) holds coordinates (axes),
    2) manages basis (multi-indices),
    3) integrates inner products via tensor-product quadrature.
    """
    def __init__(self, basis_type, verbose=False):
        self.coordinates        = {}    # {cid : Coordinate}
        self.basis_construction = basis_type
        self.basis              = None  # {basis_id: Counter({cid:deg,...})}
        self.verbose            = bool(verbose)

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    #-----------------------------------------------------------------#
    # Basis and initialization
    #-----------------------------------------------------------------#

    def evaluate_basis(self, yscalar, degree: int):
        """ψ(y) = ψ(z(y)), evaluated in Y-frame."""
        zscalar = self.physical_to_standard(yscalar)
        return self.psi_z(zscalar, degree)

    def getNumBasisFunctions(self):
        return len(self.basis) if self.basis is not None else 0

    def getNumCoordinateAxes(self):
        return len(self.coordinates)

    def getMonomialDegreeCoordinates(self):
        return {cid: coord.degree for cid, coord in self.coordinates.items()}

    def addCoordinateAxis(self, coordinate):
        self.coordinates[coordinate.id] = coordinate

    def initialize(self):
        max_deg_map = self.getMonomialDegreeCoordinates()
        if self.basis_construction == BasisFunctionType.TENSOR_DEGREE:
            self.basis = generate_basis_tensor_degree(max_deg_map)
        elif self.basis_construction == BasisFunctionType.TOTAL_DEGREE:
            self.basis = generate_basis_total_degree(max_deg_map)
        elif self.basis_construction == BasisFunctionType.ADAPTIVE_DEGREE:
            # Basis is set externally by make_adaptive_cs(); nothing to do here.
            # If called before make_adaptive_cs() has populated self.basis,
            # start with just the mean mode so the CS is at least valid.
            if self.basis is None:
                max_deg_map = self.getMonomialDegreeCoordinates()
                full = generate_basis_total_degree(max_deg_map)
                self.basis = {0: full[0]}   # mean mode only

    #-----------------------------------------------------------------#
    # Distribution statistics derived from coordinates
    #-----------------------------------------------------------------#

    @property
    def param_ids(self):
        """Ordered list of coordinate IDs."""
        return list(self.coordinates.keys())

    def mean_point(self):
        """Return {cid: mean} for all coordinate axes."""
        return {cid: coord.mean() for cid, coord in self.coordinates.items()}

    def covariance_matrix(self):
        """
        Return diagonal parameter covariance matrix (n_params x n_params).
        Off-diagonal terms are zero (independent parameters assumed).
        """
        variances = [coord.variance() for coord in self.coordinates.values()]
        return np.diag(variances)

    def mc_samples(self, n, rng):
        """Draw n Monte Carlo samples for all coordinate axes."""
        return {cid: coord.mc_samples(n, rng)
                for cid, coord in self.coordinates.items()}

    def find_modes(self, param_degrees=None, total_degree=None, exact=True):
        """
        Query the basis for modes whose multi-index matches a degree specification.

        Parameters
        ----------
        param_degrees : dict[int, int], optional
            Required degree for each specified **local** parameter index
            (0-based insertion order, as assigned by ``make_cs``).
            Only modes whose multi-index has *exactly* these degrees for the
            specified parameters are returned.  Unspecified parameters are
            handled according to the ``exact`` flag.
            If omitted, no per-parameter constraint is applied.

        total_degree : int, optional
            If given, only modes whose total polynomial degree
            ``sum(alpha.values()) == total_degree`` are returned.

        exact : bool, default True
            Controls how **unspecified** parameters (those absent from
            ``param_degrees``) are treated:

            * ``True``  — unspecified parameters must have degree **0**.
              Use this to select *pure* modes (e.g. purely linear in k₁).
            * ``False`` — unspecified parameters may have any degree.
              Use this to select modes that *involve* certain parameters
              regardless of what the remaining parameters contribute.

        Returns
        -------
        dict[int, Counter]
            ``{mode_id: degree_counter}`` for every basis mode that
            satisfies all supplied constraints.  Returns an empty dict
            when no modes match.

        Examples
        --------
        Assume a 2-parameter basis (k1=local 0, k2=local 1), degree 3:

        >>> cs_3 = cs.make_cs(3)

        # Purely linear in k1, constant in k2:
        >>> cs_3.find_modes({0: 1}, exact=True)
        {2: Counter({0: 1, 1: 0})}

        # All modes that are linear in k1 (any degree in k2):
        >>> cs_3.find_modes({0: 1}, exact=False)
        {2: ..., 5: ..., 8: ...}   # (1,0), (1,1), (1,2)

        # Specific cross-term (1,1):
        >>> cs_3.find_modes({0: 1, 1: 1}, exact=True)
        {5: Counter({0: 1, 1: 1})}

        # All level-2 modes:
        >>> cs_3.find_modes(total_degree=2)
        {3: ..., 4: ..., 5: ...}   # (2,0), (0,2), (1,1)

        # Level-2 modes that involve k1:
        >>> cs_3.find_modes({0: 1}, total_degree=2, exact=False)
        {5: Counter({0: 1, 1: 1})}   # only (1,1)

        # Only the mean mode:
        >>> cs_3.find_modes({}, exact=True)
        {0: Counter({0: 0, 1: 0})}
        """
        if self.basis is None:
            raise RuntimeError("CoordinateSystem not initialized; call initialize() first.")

        # Distinguish "no constraint" (None) from "empty constraint" ({})
        # so that find_modes(total_degree=2) returns all level-2 modes rather
        # than enforcing exact=True against an empty param_degrees dict.
        no_param_constraint = (param_degrees is None)
        param_degrees = param_degrees if param_degrees is not None else {}
        result = {}

        for mode_id, degs in self.basis.items():
            # 1. Check specified parameters match exactly
            if any(degs.get(k, 0) != d for k, d in param_degrees.items()):
                continue

            # 2. Check unspecified parameters are zero (exact mode)
            if exact and not no_param_constraint:
                specified = set(param_degrees.keys())
                if any(degs.get(k, 0) != 0
                       for k in range(len(self.coordinates))
                       if k not in specified):
                    continue

            # 3. Check total degree constraint
            if total_degree is not None and sum(degs.values()) != total_degree:
                continue

            result[mode_id] = degs

        return result

    def param_to_mode_map(self):
        """
        Return the unique degree-1 mode for each parameter (all others degree 0).

        This is the canonical mapping used to connect physical parameters to
        their PC modes in a level-1 (linear) basis expansion.  It delegates
        to ``find_modes`` with ``exact=True``.

        Returns
        -------
        dict[int, int]
            ``{local_param_index: mode_id}``.  Parameters with no matching
            mode (e.g. basis built at degree 0) are omitted.

        See Also
        --------
        find_modes : general mode query; this method is the specialisation
            ``find_modes({k: 1}, exact=True)`` for every local param index k.
        """
        result = {}
        for k in range(len(self.coordinates)):
            matches = self.find_modes({k: 1}, exact=True)
            if matches:
                result[k] = next(iter(matches))   # exactly one match
        return result

    def make_cs(self, degree, basis_type=None):
        """
        Return a new CoordinateSystem with the same distributions but
        `degree` as the expansion order. Useful for SGM at higher orders.
        Coordinate IDs are preserved so existing PolyFunctions remain valid.
        """
        basis_type = basis_type or self.basis_construction
        cf         = CoordinateFactory()
        cs_new     = CoordinateSystem(basis_type)
        for coord in self.coordinates.values():
            dist = {k: float(v) for k, v in coord.dist_coords.items()}
            cid  = cf.newCoordinateID()   # preserves insertion order (0,1,2,...)
            dist_type = coord.distribution
            if dist_type == DistributionType.UNIFORM:
                nc = cf.createUniformCoordinate(cid, coord.name, dist, degree)
            elif dist_type == DistributionType.NORMAL:
                nc = cf.createNormalCoordinate(cid, coord.name, dist, degree)
            elif dist_type == DistributionType.EXPONENTIAL:
                nc = cf.createExponentialCoordinate(cid, coord.name, dist, degree)
            else:
                raise NotImplementedError(f"make_cs: unsupported distribution {dist_type}")
            cs_new.addCoordinateAxis(nc)
        cs_new.initialize()
        return cs_new

    def make_adaptive_cs(self, strategy, stopping=None, starting=None, verbose=False):
        """
        Build a CoordinateSystem whose basis is grown adaptively.

        The adaptive loop has three orthogonal axes of control:

        * **starting** — determines the initial active set before enrichment
          begins (default: mean mode only).
        * **strategy** — selects which candidate modes to add at each step.
        * **stopping** — decides when to terminate the loop.

        Parameters
        ----------
        strategy : AdaptiveBasisStrategy
            Selects a subset of candidate modes to add at each iteration.
        stopping : StoppingCriterion, optional
            Returns ``True`` when the loop should terminate.
            Defaults to :class:`CandidatePoolExhaustedStopping`.
        starting : StartingCriterion, optional
            Builds the initial active set from the candidate pool.
            Defaults to :class:`MeanOnlyStarting` (mean mode only).
        verbose : bool, default False
            Print a one-line summary at each enrichment step.

        Returns
        -------
        CoordinateSystem
            New CS with ``ADAPTIVE_DEGREE`` basis type and the grown basis.
            Mode IDs are renumbered 0, 1, 2, … in the same relative order
            as in the reference (full total-degree) CS.

        Examples
        --------
        Level-by-level (reproduces ISQS up to degree 2):

        >>> from pspace.adaptive import LevelByLevelStrategy, MaxIterationsStopping
        >>> cs_a = cs.make_adaptive_cs(
        ...     LevelByLevelStrategy(max_degree=3),
        ...     stopping=MaxIterationsStopping(2),
        ...     verbose=True,
        ... )

        Sensitivity-driven with warm-start from level 1:

        >>> from pspace.adaptive import (SensitivityDrivenStrategy,
        ...                              LevelStarting)
        >>> variances = [c.variance() for c in cs.coordinates.values()]
        >>> cs_a = cs.make_adaptive_cs(
        ...     SensitivityDrivenStrategy(max_degree=3, variances=variances),
        ...     starting=LevelStarting(level=1),
        ... )

        Downward-closed with BSF seed:

        >>> from pspace.adaptive import (DownwardClosedStrategy,
        ...                              SensitivityStarting)
        >>> cs_a = cs.make_adaptive_cs(
        ...     DownwardClosedStrategy(max_degree=3),
        ...     starting=SensitivityStarting(variances, top_k=1),
        ... )
        """
        from .adaptive import (CandidatePoolExhaustedStopping,
                                MeanOnlyStarting,
                                RelativeGrowthStopping)

        if stopping is None:
            stopping = CandidatePoolExhaustedStopping()
        if starting is None:
            starting = MeanOnlyStarting()

        # Let direction-aware stopping criteria know which mode we're in
        if hasattr(stopping, '_set_direction'):
            stopping._set_direction(strategy.direction)

        # Full reference CS at the strategy's max_degree
        cs_ref = self.make_cs(strategy.max_degree)

        # Split the full basis into:
        #   baseline — modes with |α| < min_degree, always active (pre-seeded)
        #   pool     — modes with |α| >= min_degree, subject to Starting/Strategy
        min_deg  = strategy.min_degree
        baseline = {mid: Counter(degs)
                    for mid, degs in cs_ref.basis.items()
                    if sum(degs.values()) < min_deg}
        pool     = {mid: Counter(degs)
                    for mid, degs in cs_ref.basis.items()
                    if sum(degs.values()) >= min_deg}

        # Delegate initial active set to the starting criterion (operates on pool)
        active = {**baseline, **starting.initialize(pool, cs_ref)}

        # Give the strategy a chance to reject an incompatible initial set
        # (e.g. DownwardClosedStrategy requires a downward-closed seed)
        strategy.validate_initial_set(active, cs_ref)

        iteration = 0
        # In grow mode the pool is consumed into active; loop while pool non-empty.
        # In decay mode active is consumed back into pool; loop while active non-empty.
        _source = lambda: pool if strategy.direction == 'grow' else active
        while _source():
            if stopping.should_stop(active, pool, cs_ref, iteration):
                break

            selected = strategy.select(active, pool, cs_ref)
            if not selected:
                break

            if strategy.direction == 'grow':
                for mid in selected:
                    active[mid] = pool.pop(mid)
            else:  # decay — evict selected modes from active back to pool
                for mid in selected:
                    pool[mid] = active.pop(mid)

            # Notify RelativeGrowthStopping of this step's ratio
            if isinstance(stopping, RelativeGrowthStopping):
                stopping.record(len(selected), len(active))

            if verbose:
                n_new   = len(selected)
                n_act   = len(active)
                max_lev = max(sum(d.values()) for d in active.values())
                print(f"[adaptive] iter {iteration:3d}: "
                      f"+{n_new} modes -> active={n_act}, max_level={max_lev}")

            iteration += 1

        # Renumber mode IDs to be contiguous (preserve relative order)
        sorted_ids = sorted(active.keys())
        new_basis  = {new_id: active[old_id]
                      for new_id, old_id in enumerate(sorted_ids)}

        # Build the returned CS: same axes, ADAPTIVE_DEGREE, custom basis
        cs_out = self.make_cs(strategy.max_degree)
        cs_out.basis_construction = BasisFunctionType.ADAPTIVE_DEGREE
        cs_out.basis = new_basis
        return cs_out


        """
        Factory method returning a PointSampler for this CoordinateSystem.

        Parameters
        ----------
        implementation : str
            'quadrature'   — QuadratureSampler (Gauss tensor-product rule)
            'monte-carlo'  — MonteCarloSampler  (pseudo-random draws)
        **kwargs
            quadrature : degree (int, default 4)
            monte-carlo: n (int, default 10_000), rng_seed (int, default 42)

        Returns
        -------
        PointSampler instance
        """
        if implementation == 'quadrature':
            return QuadratureSampler(self, degree=kwargs.get('degree', 4))
        elif implementation == 'monte-carlo':
            return MonteCarloSampler(self,
                                     n=kwargs.get('n', 10_000),
                                     rng_seed=kwargs.get('rng_seed', 42))
        else:
            raise ValueError(
                f"Unknown sampler '{implementation}'. "
                f"Choose 'quadrature' or 'monte-carlo'."
            )

    #-----------------------------------------------------------------#
    # Quadrature
    #-----------------------------------------------------------------#

    def print_quadrature(self, qmap):
        if not self.verbose:
            return
        print("Quadrature rule:")
        print("-" * 80)
        for q, data in qmap.items():
            y_str = ", ".join(f"y{cid}={val:.6g}" for cid, val in data['Y'].items())
            z_str = ", ".join(f"z{cid}={val:.6g}" for cid, val in data['Z'].items())
            print(f"q={q:3d} : {y_str:<36} | {z_str:<36} | W={data['W']:.6g}")
        print("-" * 80)
        print("sum W =", sum(d['W'] for d in qmap.values()))

    def build_quadrature(self, degrees: Counter):
        """
        degrees : Counter({cid: p_i}) — per-axis polynomial degree
        Returns : {q_index: {'Y':{cid:y}, 'Z':{cid:z}, 'W':w}}
        """
        cids  = list(self.coordinates.keys())
        one_d = {cid: self.coordinates[cid].getQuadraturePointsWeights(
                    int(degrees.get(cid, 0))) for cid in cids}
        sizes = {cid: len(one_d[cid]['wq']) for cid in cids}

        qmap, ctr = {}, 0
        for i_tuple in product(*[range(sizes[cid]) for cid in cids]):
            y, z, w = {}, {}, 1.0
            for cid, i in zip(cids, i_tuple):
                y[cid] = one_d[cid]['yq'][i]
                z[cid] = one_d[cid]['zq'][i]
                w     *= one_d[cid]['wq'][i]
            qmap[ctr] = {'Y': y, 'Z': z, 'W': w}
            ctr      += 1

        self.print_quadrature(qmap)

        return qmap

    #-----------------------------------------------------------------#
    # Basis evaluation
    #-----------------------------------------------------------------#

    def evaluateBasisDegreesY(self, y_by_cid, degrees_counter):
        val = 1.0
        for cid, deg in degrees_counter.items():
            val *= self.coordinates[cid].psi_y(y_by_cid[cid], deg)
        return val

    def evaluateBasisIndexY(self, y_by_cid, basis_id):
        degrees = self.basis[basis_id]
        return self.evaluateBasisDegreesY(y_by_cid, degrees)

    #-----------------------------------------------------------------#
    # Sparsity Detection Utilities
    #-----------------------------------------------------------------#

    def sparse_vector(self, dmapi, dmapf):
        """
        Detect sparsity for <f, ψ_i>.

        dmapi : Counter({axis: degree}) for basis ψ_i
        dmapf : Counter({axis: degree}) for function f
        """
        for axis, deg in dmapi.items():
            if deg > dmapf.get(axis, 0):
                return False
        return True

    def monomial_vector_sparsity_mask(self, f_deg: Counter):
        mask = set()
        for i, psi_i in self.basis.items():
            if self.sparse_vector(psi_i, f_deg):
                mask.add(i)
        return mask

    def polynomial_vector_sparsity_mask(self, f_degrees: list[Counter]):
        mask = set()
        for f_deg in f_degrees:
            mask |= self.monomial_vector_sparsity_mask(f_deg)
        return mask

    #-----------------------------------------------------------------#
    # Inner products
    #-----------------------------------------------------------------#

    def inner_product(self, f_eval, g_eval,
                      f_deg: Counter|None=None,
                      g_deg: Counter|None=None):
        """
        <f, g> in Y-frame = ∑ f(Y_q) g(Y_q) W_q
        """
        coord_ids = list(self.coordinates.keys())
        f_deg     = f_deg or safe_zero_degrees(coord_ids)
        g_deg     = g_deg or safe_zero_degrees(coord_ids)
        need      = sum_degrees(f_deg, g_deg)

        qmap = self.build_quadrature(need)
        s    = 0.0
        for q in qmap.values():
            y  = q['Y']
            s += f_eval(y) * g_eval(y) * q['W']
        return s

    def inner_product_basis(self,
                            i_id: int, j_id: int,
                            f_eval=None, f_deg: Counter|None=None):
        """
        <ψ_i, f, ψ_j> in Y-frame.
        """
        psi_i = self.basis[i_id]
        psi_j = self.basis[j_id]

        coord_ids = list(self.coordinates.keys())
        f_deg     = f_deg or safe_zero_degrees(coord_ids)
        need      = sum_degrees(psi_i, psi_j, f_deg)

        qmap = self.build_quadrature(need)
        s    = 0.0
        for q in qmap.values():
            y   = q['Y']
            val = (self.evaluateBasisDegreesY(y, psi_i) *
                   self.evaluateBasisDegreesY(y, psi_j))
            if f_eval is not None:
                val *= f_eval(y)
            s   += val * q['W']
        return s

    #-----------------------------------------------------------------#
    # Decomposition
    #-----------------------------------------------------------------#

    def decompose(self,
                  function : PolyFunction,
                  sparse   : bool = True,
                  analytic : bool = False):
        """
        Coefficients c_k = <f, ψ_k> in Y-frame.

        Parameters
        ----------
        function : PolyFunction
            Polynomial function to decompose.
        sparse   : bool
            If True, restrict to admissible basis indices.
        analytic : bool
            If True, compute coefficients with Sympy integrals instead of quadrature.

        Returns
        -------
        coeffs : dict {basis_id: coefficient}
        """
        coords = self.coordinates
        symbols = {cid: coord.symbol for cid, coord in coords.items()}

        #---------------------------------------------------------------#
        # Build admissible mask
        #---------------------------------------------------------------#
        if sparse:
            mask = self.polynomial_vector_sparsity_mask(function.degrees)
        else:
            mask = self.basis.keys()

        coeffs = {}
        for k in mask:
            psi_k = self.basis[k]

            if analytic:
                #-------------------------------------------------------#
                # Analytic Sympy integration
                #-------------------------------------------------------#
                psi_expr = 1
                for cid, deg in psi_k.items():
                    z = coords[cid].physical_to_standard(coords[cid].symbol)
                    psi_expr *= coords[cid].psi_z(z, deg)

                integrand = function(symbols) * psi_expr * sp.Mul(*[c.weight() for c in coords.values()])
                val = integrand
                for cid, coord in coords.items():
                    y = coord.symbol
                    a, b = coord.domain()
                    val = sp.integrate(val, (y, a, b))

                coeffs[k] = sp.simplify(val)

            else:
                #-------------------------------------------------------#
                # Numerical quadrature
                #-------------------------------------------------------#
                need = sum_degrees_union_vector(function.max_degrees, psi_k)
                qmap = self.build_quadrature(need)

                s = 0.0
                for q in qmap.values():
                    y = q['Y']
                    s += function(y) * self.evaluateBasisDegreesY(y, psi_k) * q['W']
                coeffs[k] = s

        # Optionally fill zeroes
        if sparse:
            for k in self.basis:
                if k not in coeffs:
                    coeffs[k] = 0

        return coeffs

    def admissible_pair(self, deg_i: Counter, deg_j: Counter, f_deg: Counter) -> bool:
        """
        Axis-wise admissibility rule for a single monomial.

        admissible_pair          = atomic check (axis-wise rule)
        monomial_sparsity_mask   = per-monomial mask
        polynomial_sparsity_mask = per-polynomial union of masks

        Parameters
        ----------
        deg_i, deg_j : Counter
            Degree structure of basis functions psi_i, psi_j.
        f_deg : Counter
            Degree structure of one monomial in f.

        Returns
        -------
        bool
            True if <psi_i, psi_j, f_monomial> can be nonzero.

        Rule
        ----
        For every axis d:
            |deg_i(d) - deg_j(d)| <= f_deg(d) <= deg_i(d) + deg_j(d)
        """

        """
        Axis-wise admissibility rule for a single monomial.

        If f_deg is empty (constant monomial), then all (i,j) pairs are admissible.
        """

        # Constant monomial -> don't filter anything
        if not f_deg:
            return deg_i == deg_j

        axes = set(deg_i) | set(deg_j) | set(f_deg)
        for d in axes:
            di, dj, df = deg_i.get(d, 0), deg_j.get(d, 0), f_deg.get(d, 0)
            if not (abs(di - dj) <= df):
                return False
        return True

    def monomial_sparsity_mask(self, f_deg: Counter, symmetric: bool = False):
        """
        Sparsity mask for a single monomial term in f.
        Constant monomial admits all (i,j).
        """
        mask = set()
        basis_keys = sorted(self.basis.keys())

        if not f_deg:
            # Constant term: admit everything
            for ii, i in enumerate(basis_keys):
                jstart = ii if symmetric else 0
                for j in basis_keys[jstart:]:
                    mask.add((i, j))
            return mask

        for ii, i in enumerate(basis_keys):
            jstart = ii if symmetric else 0
            for j in basis_keys[jstart:]:
                if self.admissible_pair(self.basis[i], self.basis[j], f_deg):
                    mask.add((i, j))
        return mask

    def polynomial_sparsity_mask(self, f_degrees: list[Counter], symmetric: bool = False):
        """
        Sparsity mask for a full polynomial f, as union of monomial masks.

        Parameters
        ----------
        f_degrees : list of Counters
            Each Counter gives the degree structure of one monomial in f.
        symmetric : bool
            If True, only return pairs (i,j) with i <= j.

        Returns
        -------
        mask : set of (i,j) tuples
        """
        mask = set()
        for f_deg in f_degrees:
            mask |= self.monomial_sparsity_mask(f_deg, symmetric=symmetric)
        return mask

    def decompose_matrix(self, function, sparse=False, symmetric=True):
        """
        Assemble A_ij = ∫ psi_i(y) psi_j(y) f(y) w(y) dy (dense).

        Parameters
        ----------
        function  : PolyFunction
            Callable with .degrees property for sparsity.
        sparse    : bool
            If True, restrict to admissible pairs.
        symmetric : bool
            If True, compute only i ≤ j and mirror.

        Returns
        -------
        A : np.ndarray
            Dense (nbasis x nbasis) coefficient matrix.
        """
        nbasis = self.getNumBasisFunctions()
        A      = np.zeros((nbasis, nbasis))

        # Build admissible mask
        if sparse:
            mask = self.polynomial_sparsity_mask(function.degrees, symmetric=symmetric)
        else:
            if symmetric:
                mask = {(i,j) for i in self.basis for j in self.basis if i <= j}
            else:
                mask = {(i,j) for i in self.basis for j in self.basis}

        qcache = {}
        for i, j in mask:
            psi_i, psi_j = self.basis[i], self.basis[j]
            need = sum_degrees_union_matrix(function.max_degrees, psi_i, psi_j)

            key = tuple(sorted(need.items()))
            qmap = qcache.get(key)
            if qmap is None:
                qmap = self.build_quadrature(need)
                qcache[key] = qmap

            s = 0.0
            for q in qmap.values():
                y = q['Y']
                s += (function(y)
                      * self.evaluateBasisDegreesY(y, psi_i)
                      * self.evaluateBasisDegreesY(y, psi_j)) * q['W']

            A[i,j] = s
            if symmetric and i != j:
                A[j,i] = s

        return A

    def decompose_matrix_analytic(self, function, sparse=False, symmetric=True):
        """
        Assemble A_ij = ∫ psi_i(y) psi_j(y) f(y) w(y) dy (dense),
        using analytic (Sympy) integration.

        Parameters
        ----------
        function  : PolyFunction
            Polynomial function to decompose (with .degrees and .max_degrees).
        sparse    : bool
            If True, restrict to admissible pairs.
        symmetric : bool
            If True, compute only i ≤ j and mirror.

        Returns
        -------
        A : np.ndarray
            Dense (nbasis x nbasis) coefficient matrix.
        """
        import sympy as sp

        nbasis = self.getNumBasisFunctions()
        A      = np.zeros((nbasis, nbasis))

        coords  = self.coordinates
        symbols = {cid: coord.symbol for cid, coord in coords.items()}

        # Build admissible mask
        if sparse:
            mask = self.polynomial_sparsity_mask(function.degrees, symmetric=symmetric)
        else:
            if symmetric:
                mask = {(i, j) for i in self.basis for j in self.basis if i <= j}
            else:
                mask = {(i, j) for i in self.basis for j in self.basis}

        for i, j in mask:
            psi_i, psi_j = self.basis[i], self.basis[j]

            # Build integrand symbolically
            psi_expr_i, psi_expr_j = 1, 1
            for cid, deg in psi_i.items():
                z = coords[cid].physical_to_standard(coords[cid].symbol)
                psi_expr_i *= coords[cid].psi_z(z, deg)
            for cid, deg in psi_j.items():
                z = coords[cid].physical_to_standard(coords[cid].symbol)
                psi_expr_j *= coords[cid].psi_z(z, deg)

            # Function f(y) expanded
            f_expr = 0
            for coeff, degs in function.terms:
                mon = coeff
                for cid, d in degs.items():
                    mon *= symbols[cid] ** d
                f_expr += mon

            w_expr = sp.Mul(*[c.weight() for c in coords.values()])

            # Full integrand: f * ψ_i * ψ_j * weight
            integrand = f_expr * psi_expr_i * psi_expr_j * w_expr

            val = integrand
            for cid, coord in coords.items():
                y = coord.symbol
                a, b = coord.domain()
                val = sp.integrate(val, (y, a, b))

            # Simplify and cast to float
            A[i, j] = float(sp.simplify(val))
            if symmetric and i != j:
                A[j, i] = A[i, j]

        return A

    #-----------------------------------------------------------------#
    # Consistency checks
    # TestCoordinateSystem and supply the instance of cs
    #-----------------------------------------------------------------#

    def check_orthonormality(self):
        """
        """
        nbasis = self.getNumBasisFunctions()
        A = np.zeros((nbasis, nbasis))
        for ii in range(nbasis):
            for jj in range(nbasis):
                A[ii,jj] = self.inner_product_basis(ii, jj)
        return np.linalg.norm(A - np.eye(nbasis), ord=np.inf)

    def check_decomposition_numerical_symbolic(self,
                                               function: PolyFunction,
                                               sparse: bool = True,
                                               tol=1e-10,
                                               verbose=True):
        """
        Cross-check numerical vs analytic decomposition.
        """

        # ensure basis is orthonormal first
        ortho_tol = self.check_orthonormality()

        from timeit import default_timer as timer

        #-------------------------------------------------------------#
        # Numerical decomposition (quadrature)
        #-------------------------------------------------------------#

        start_num   = timer()
        coeffs_num  = self.decompose(function, sparse=sparse, analytic=False)
        elapsed_num = timer() - start_num

        #-------------------------------------------------------------#
        # Analytic decomposition (Sympy)
        #-------------------------------------------------------------#

        start_sym   = timer()
        coeffs_sym  = self.decompose(function, sparse=sparse, analytic=True)
        elapsed_sym = timer() - start_sym

        #-------------------------------------------------------------#
        # Compare coefficients
        #-------------------------------------------------------------#

        diffs, ok = {}, True
        for k in coeffs_num.keys():
            num_val = float(coeffs_num[k])
            try:
                ana_val = float(coeffs_sym[k].evalf())
            except Exception:
                ana_val = float(sp.N(coeffs_sym[k], 15))
            err = abs(num_val - ana_val)
            if err > tol:
                ok = False
            diffs[k] = (num_val, ana_val, err)

        #-------------------------------------------------------------#
        # Reporting
        #-------------------------------------------------------------#

        if verbose or not ok:
            status = "PASSED" if ok else "FAILED"
            print(f"[Consistency Check] {status} with tol = {tol}, ortho tol = {ortho_tol:.2e}")
            print(f"[Elapsed Time] numerical {elapsed_num:.3e}  analytic = {elapsed_sym:.3e}")
            header = f"{'Basis':<7} {'numerical':>12} {'analytic':>12} {'error':>12}"
            print(header)
            print("-" * len(header))
            for k, (n, a, e) in diffs.items():
                print(f"{k:<7d} {n:12.6f} {a:12.6f} {e:12.2e}")
            print("-" * len(header))

        return ok, diffs

    #-----------------------------------------------------------------#
    # Sparse vs full Assembly (selectively employ dot products)
    #-----------------------------------------------------------------#

    def check_decomposition_numerical_sparse_full(self, function: PolyFunction,
                                                  tol=1e-12, verbose=True):
        """
        Cross-check sparse vs full assembly of rank 1 decomposition
        coefficients
        """
        from timeit import default_timer as timer

        start_sparse   = timer()
        coeffs_sparse  = self.decompose(function, sparse = True)
        elapsed_sparse = timer() - start_sparse

        start_full   = timer()
        coeffs_full  = self.decompose(function, sparse = False)
        elapsed_full = timer() - start_full

        diffs, ok = {}, True
        for k in coeffs_sparse.keys():
            coeff_sparse = coeffs_sparse[k]
            coeff_full   = coeffs_full[k]

            err = abs(coeff_sparse - coeff_full)
            if err > tol:
                ok = False

            diffs[k] = (coeff_sparse, coeff_full, err)

        if verbose or not ok:
            print(f"[Assembly Check] {'PASSED' if ok else 'FAILED'} with tol = {tol}")
            print(f"[Elapsed Time] Sparse {elapsed_sparse} Full = {elapsed_full} Ratio = {elapsed_full/elapsed_sparse}")
            header = f"{'Basis':<7} {'Sparse':>12} {'Full':>12} {'Error':>12}"
            print(header)
            print("-" * len(header))
            for k, (n, a, e) in diffs.items():
                print(f"{k:<7d} {float(n):12.6f} {float(a):12.6f} {float(e):12.2e}")
            print("-" * len(header))

        return ok, diffs

    def check_decomposition_matrix_sparse_full(self, function, tol=1e-12, verbose=True):
        """
        Cross-check sparse vs full assembly of rank-2 (matrix) decomposition
        coefficients.
        """
        from timeit import default_timer as timer

        #-------------------------------------------------------------#
        # Assemble sparse + full
        #-------------------------------------------------------------#

        start_sparse   = timer()
        A_sparse       = self.decompose_matrix(function, sparse=True, symmetric=True)
        elapsed_sparse = timer() - start_sparse

        start_full   = timer()
        A_full       = self.decompose_matrix(function, sparse=False, symmetric=True)
        elapsed_full = timer() - start_full

        #-------------------------------------------------------------#
        # Compute differences
        #-------------------------------------------------------------#

        diffs, ok = {}, True
        nbasis = self.getNumBasisFunctions()
        for i in range(nbasis):
            for j in range(nbasis):
                vsparse = A_sparse[i, j]
                vfull   = A_full[i, j]
                err     = abs(vsparse - vfull)
                if err > tol:
                    ok = False
                diffs[(i, j)] = (vsparse, vfull, err)

        #---------------------------------------------------------------#
        # Report
        #---------------------------------------------------------------#

        if verbose or not ok:
            print(f"[Matrix Assembly Check] {'PASSED' if ok else 'FAILED'} "
                  f"with tol = {tol}")
            print(f"[Elapsed Time] Sparse {elapsed_sparse:.4e}  "
                  f"Full {elapsed_full:.4e}  "
                  f"Ratio {elapsed_full/elapsed_sparse:.2f}")
            header = f"{'i':<3} {'j':<3} {'Sparse':>12} {'Full':>12} {'Error':>12}"
            print(header)
            print("-" * len(header))
            for (i, j), (vs, vf, e) in diffs.items():
                if abs(e) > tol:  # only print significant diffs
                    print(f"{i:<3d} {j:<3d} {float(vs):12.6f} "
                          f"{float(vf):12.6f} {float(e):12.2e}")
            print("-" * len(header))

        return ok, diffs

    def check_decomposition_matrix_numerical_symbolic(self, function, tol=1e-12, verbose=True):
        """
        Cross-check numerical vs analytic assembly of rank-2 (matrix)
        decomposition coefficients.

        Parameters
        ----------
        function : PolyFunction
            Polynomial function to decompose.
        tol : float
            Absolute tolerance for consistency check.
        verbose : bool
            Print detailed report if True.

        Returns
        -------
        ok : bool
            True if all entries match within tolerance.
        diffs : dict
            Mapping (i,j) -> (numerical, analytic, error).
        """
        from timeit import default_timer as timer

        #-------------------------------------------------------------#
        # Assemble numerical + analytic
        #-------------------------------------------------------------#

        start_num = timer()
        A_num     = self.decompose_matrix(function, sparse=True, symmetric=True)
        elapsed_num = timer() - start_num

        start_an  = timer()
        A_an      = self.decompose_matrix_analytic(function, sparse=True, symmetric=True)
        elapsed_an = timer() - start_an

        #-------------------------------------------------------------#
        # Compute differences
        #-------------------------------------------------------------#

        diffs, ok = {}, True
        nbasis = self.getNumBasisFunctions()
        for i in range(nbasis):
            for j in range(nbasis):
                vnum = A_num[i, j]
                van  = A_an[i, j]
                err  = abs(vnum - van)
                if err > tol:
                    ok = False
                diffs[(i, j)] = (vnum, van, err)

        #-------------------------------------------------------------#
        # Report
        #-------------------------------------------------------------#

        if verbose or not ok:
            print(f"[Matrix Numerical vs Analytic Check] {'PASSED' if ok else 'FAILED'} "
                  f"with tol = {tol}")
            print(f"[Elapsed Time] numerical {elapsed_num:.4e}  "
                  f"analytic {elapsed_an:.4e}  "
                  f"Ratio {elapsed_an/elapsed_num:.2f}")
            header = f"{'i':<3} {'j':<3} {'Numerical':>12} {'Analytic':>12} {'Error':>12}"
            print(header)
            print("-" * len(header))
            for (i, j), (vn, va, e) in diffs.items():
                if abs(e) > tol:  # only print significant diffs
                    print(f"{i:<3d} {j:<3d} {vn:12.6f} {va:12.6f} {e:12.2e}")
            print("-" * len(header))

        return ok, diffs

#=====================================================================#
# State Equation Interface
#=====================================================================#

class StateEquation:
    """
    Generic state equation in coefficient form:
        Operator * a_state = RHS
    where Operator is assembled in the CoordinateSystem basis.
    """

    def __init__(self, name, operator_fn, rhs_fn, coord_system):
        """
        Parameters
        ----------
        name : str
            Identifier (e.g., "reconstruction", "diffusion", "helmholtz")
        operator_fn : callable | PolyFunction
            Defines the operator kernel f(y) for assembling A_ij = <ψ_i, f, ψ_j>
        rhs_fn : callable | PolyFunction | np.ndarray
            Defines RHS b_i = <ψ_i, f> or direct coefficients
        coord_system : CoordinateSystem
            The coordinate system in which this state equation lives.
        """
        self.name = name
        self.operator_fn = operator_fn
        self.rhs_fn = rhs_fn
        self.cs = coord_system
        self.operator_matrix = None
        self.rhs_vector = None
        self.solution = None

    #-------------------------------------------------------------#
    # Assembly
    #-------------------------------------------------------------#
    def assemble(self, analytic=False, sparse=True, symmetric=True):
        """Assemble operator and RHS in the coordinate basis."""
        cs = self.cs
        if isinstance(self.operator_fn, PolyFunction):
            if analytic:
                A = cs.decompose_matrix_analytic(self.operator_fn,
                                                 sparse=sparse, symmetric=symmetric)
            else:
                A = cs.decompose_matrix(self.operator_fn,
                                        sparse=sparse, symmetric=symmetric)
        elif callable(self.operator_fn):
            raise NotImplementedError("Callable operator assembly not yet implemented")
        else:
            A = np.asarray(self.operator_fn)

        # RHS
        if isinstance(self.rhs_fn, PolyFunction):
            b = cs.decompose(self.rhs_fn, sparse=sparse)
            b = np.array([b[k] for k in sorted(b.keys())])
        elif callable(self.rhs_fn):
            raise NotImplementedError("Callable RHS not yet implemented")
        else:
            b = np.asarray(self.rhs_fn)

        self.operator_matrix = A
        self.rhs_vector = b

    #-------------------------------------------------------------#
    # Preconditioning / Whitening
    #-------------------------------------------------------------#
    def precondition(self, method="cholesky"):
        """Compute and apply preconditioner P such that P⁻¹ A P⁻ᵀ ≈ I."""
        A = self.operator_matrix
        if method == "cholesky":
            L = np.linalg.cholesky(A)
            P = L
        elif method == "spectral":
            eigval, eigvec = np.linalg.eigh(A)
            P = eigvec @ np.diag(np.sqrt(eigval))
        else:
            raise ValueError(f"Unknown preconditioner {method}")

        self.P = P
        self.P_inv = np.linalg.inv(P)
        self.operator_whitened = self.P_inv @ A @ self.P_inv.T
        self.rhs_whitened = self.P_inv @ self.rhs_vector

    #-------------------------------------------------------------#
    # Solve
    #-------------------------------------------------------------#
    def solve(self):
        """Solve the whitened or raw system."""
        A = getattr(self, "operator_whitened", self.operator_matrix)
        b = getattr(self, "rhs_whitened", self.rhs_vector)
        x = np.linalg.solve(A, b)
        # Back transform if whitened
        if hasattr(self, "P"):
            x = self.P_inv.T @ x
        self.solution = x
        return x

    #-------------------------------------------------------------#
    # Diagnostic
    #-------------------------------------------------------------#
    def condition_number(self):
        A = self.operator_matrix
        return np.linalg.cond(A)

    def __repr__(self):
        return f"StateEquation({self.name}, nbasis={self.cs.getNumBasisFunctions()})"
