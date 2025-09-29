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

    def __repr__(self):
        return f"PolyFunction({self._terms})"

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
        else:
            raise NotImplementedError("ADAPTIVE_DEGREE path not implemented")

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

        dmapi     : Counter({axis: degree}) for basis ψ_i
        dmapf     : Counter({axis: degree}) for function f
        basis_type: "tensor" or "total"
        """
        if self.basis_construction == BasisFunctionType.TENSOR_DEGREE:
            # Axis-by-axis cutoff
            return all(dmapi[k] <= dmapf[k] for k in dmapf.keys())
        elif self.basis_construction == BasisFunctionType.TOTAL_DEGREE:
            # Global cutoff (total degree)
            return sum(dmapi.values()) <= sum(dmapf.values())
        else:
            raise ValueError("Unknown basis_type")

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
