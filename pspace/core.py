#!/usr/bin/env python

#=====================================================================#
# சீரற்ற பகுதி வேறுபாட்டுச் சமன்பாடுகளுக்கான
# சுருக்கமான கணிதப் பகுப்பாய்வு தொகுதி
#————————————————————————————————————————————————————————————————————-#
# ABSTRACT MATHEMATICAL ANALYSIS MODULE FOR STOCHASTIC PARTIAL
# DIFFERENTIAL EQUATIONS
#=====================================================================#
# X   பரப்புகள் (சாத்தியவியல், இடவியல், காலவியல்)
# XX  பரிமாணங்கள் (அச்சுகள்) சாத்தியவியல் பரவல்களுடன்
# XXX நிலைகள் (அடித்தளச் செயல்பாடுகள் மற்றும் முழுமையாக்கம்)
#————————————————————————————————————————————————————————————————————-#
# X   DOMAINS (PROBABILISTIC, SPATIAL, TEMPORAL)
# XX  DIMENSIONS (AXES) with PROBABILITY DISTRIBUTIONS
# XXX MODES (BASIS FUNCTIONS AND QUADRATURES)
#=====================================================================#
# ஆசிரியர் : கோமகன் பூபதி (komahan.boopathy@gmail.com)
#————————————————————————————————————————————————————————————————————-#
# Author    : Komahan Boopathy (komahan.boopathy@gmail.com)
#=====================================================================#

#=====================================================================#
# External modules
#=====================================================================#

import math
import sympy as sp
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from collections import Counter
from enum        import Enum
from itertools   import product

#=====================================================================#
# Local modules
#=====================================================================#

from .stochastic_utils       import (
    minnum_quadrature_points,
    generate_basis_tensor_degree,
    generate_basis_total_degree,
    sum_degrees,
    safe_zero_degrees
)

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

    #-----------------------------------------------------------------#
    # canonical mappings (must be implemented by subclasses)
    #-----------------------------------------------------------------#
    def to_standard(self, yscalar):
        """Map physical y -> standard z (basis domain)."""
        raise NotImplementedError

    def to_physical(self, zscalar):
        """Map standard z -> physical y (user domain)."""
        raise NotImplementedError

    def weight(self):
        """Return symbolic weight function ρ(y) attached to this coordinate."""
        return self.rho

    #-----------------------------------------------------------------#
    # basis eval in y (thin wrapper over z)
    #-----------------------------------------------------------------#
    def evaluateBasisAtY(self, yscalar, degree):
        z = self.to_standard(yscalar)
        return self._evaluateBasisAtZ(z, degree)

    #-----------------------------------------------------------------#
    # subclass provides the family polynomial in z
    #-----------------------------------------------------------------#
    def _evaluateBasisAtZ(self, zscalar, degree):
        raise NotImplementedError

    #-----------------------------------------------------------------#
    # subclass provides a 1D Gauss rule for the needed degree
    #-----------------------------------------------------------------#
    def _quad_rule_xw(self, degree):
        """
        Return native Gauss rule (x_nodes, w_nodes) sized for `degree`.
        Subclass chooses the correct family and normalization.
        """
        raise NotImplementedError

    #-----------------------------------------------------------------#
    # lift native x-rule to (z,y,w) consistently
    #-----------------------------------------------------------------#
    def getQuadraturePointsWeights(self, degree):
        """
        Return {'yq','zq','wq'} where:
          - zq is the standard variable nodes (basis is orthonormal here)
          - yq is the physical nodes (user functions evaluated here)
          - wq integrates in z-domain (Jacobian absorbed)
        """
        x, w = self._quad_rule_xw(degree)

        # Map x -> z (family dependent); many families have z == x
        z = self._x_to_z(x)

        # Map z -> y using distribution parameters
        y = np.array([self.to_physical(zz) for zz in z])

        return {'yq': y, 'zq': z, 'wq': w}

    #-----------------------------------------------------------------#
    # Default identity for families where x==z
    #-----------------------------------------------------------------#
    def _x_to_z(self, x):
        return np.asarray(x)

#=====================================================================#
# Coordinate Implementations
#=====================================================================#

class NormalCoordinate(Coordinate):
    def __init__(self, pdata):
        super().__init__(pdata)
        self.dist_coords = pdata['dist_coords']

        mu, sigma = self.dist_coords['mu'], self.dist_coords['sigma']
        self.rho  = 1/(sp.sqrt(2*sp.pi)*sigma) * sp.exp(-(self.symbol - mu)**2/(2*sigma**2))

    def to_standard(self, y):
        mu, s = self.dist_coords['mu'], self.dist_coords['sigma']
        return (y - mu) / s

    def to_physical(self, z):
        mu, s = self.dist_coords['mu'], self.dist_coords['sigma']
        return mu + s * z

    def _evaluateBasisAtZ(self, z, degree):
        return unit_hermite(z, degree)

    def _quad_rule_xw(self, degree):
        npts    = minnum_quadrature_points(degree)
        x, w    = np.polynomial.hermite.hermgauss(npts)
        w       = w / np.sqrt(np.pi)
        return x, w

    def _x_to_z(self, x):
        return np.sqrt(2.0) * np.asarray(x)

class UniformCoordinate(Coordinate):
    def __init__(self, pdata):
        super().__init__(pdata)
        self.dist_coords = pdata['dist_coords']

        a, b   = self.dist_coords['a'], self.dist_coords['b']
        self.rho = sp.Piecewise((1/(b-a), (self.symbol>=a) & (self.symbol<=b)), (0, True))

    def to_standard(self, y):
        a, b = self.dist_coords['a'], self.dist_coords['b']
        return (y - a) / (b - a) * 2.0 - 1.0

    def to_physical(self, z):
        a, b = self.dist_coords['a'], self.dist_coords['b']
        return (b - a) * (z + 1.0) / 2.0 + a

    def _evaluateBasisAtZ(self, z, degree):
        return unit_legendre(z, degree)

    def _quad_rule_xw(self, degree):
        npts = minnum_quadrature_points(degree)
        x, w = np.polynomial.legendre.leggauss(npts)
        w    = w / 2.0
        return x, w

class ExponentialCoordinate(Coordinate):
    def __init__(self, pdata):
        super().__init__(pdata)
        self.dist_coords = pdata['dist_coords']

        mu, beta = self.dist_coords['mu'], self.dist_coords['beta']
        self.rho = (1/beta) * sp.exp(-(self.symbol - mu)/beta)

    def to_standard(self, y):
        mu, b = self.dist_coords['mu'], self.dist_coords['beta']
        return (y - mu) / b

    def to_physical(self, z):
        mu, b = self.dist_coords['mu'], self.dist_coords['beta']
        return mu + b * z

    def _evaluateBasisAtZ(self, z, degree):
        return unit_laguerre(z, degree)

    def _quad_rule_xw(self, degree):
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
        pid               = self.next_coord_id
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
        self.coordinates        = {}    # cid -> Coordinate
        self.basis_construction = basis_type
        self.basis              = None  # {basis_id: Counter({cid:deg,...})}
        self.verbose            = bool(verbose)

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    #-----------------------------------------------------------------#
    # Basis and initialization
    #-----------------------------------------------------------------#

    def basis_at_y(self, yscalar, degree: int):
        """ψ(y) = ψ(z(y)), evaluated symbolically in Y-frame."""
        z = self.to_standard(yscalar)
        return self.evaluateBasisFunction(z, degree)

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
            val *= self.coordinates[cid].evaluateBasisAtY(y_by_cid[cid], deg)
        return val

    def evaluateBasisIndexY(self, y_by_cid, basis_id):
        degrees = self.basis[basis_id]
        return self.evaluateBasisDegreesY(y_by_cid, degrees)

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

    def inner_product_basis(self, i_id: int, j_id: int,
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

    def decompose(self, f_eval, f_deg: Counter):
        """
        Coefficients c_k = <f, ψ_k> in Y-frame.
        """
        coeffs = {}
        for k, psi_k in self.basis.items():
            need = sum_degrees(f_deg, psi_k)
            qmap = self.build_quadrature(need)

            s = 0.0
            for q in qmap.values():
                y = q['Y']
                s += f_eval(y) * self.evaluateBasisDegreesY(y, psi_k) * q['W']
            coeffs[k] = s

        return coeffs

    def decompose_analytic(self, f_eval, f_deg: Counter):
        """
        Analytic decomposition using SymPy:
        c_k = ∫ f(y) ψ_k(y) ρ(y) dy
        """
        coords  = self.coordinates
        symbols = {cid: coord.symbol for cid, coord in coords.items()}
        f_expr  = f_eval(symbols)

        coeffs = {}
        for k, psi_k in self.basis.items():
            psi_expr = 1
            for cid, deg in psi_k.items():
                z        = coords[cid].to_standard(coords[cid].symbol)
                psi_expr *= coords[cid]._evaluateBasisAtZ(z, deg)

            integrand = f_expr * psi_expr * sp.Mul(*[c.weight()
                                                     for c in coords.values()])

            val = integrand
            for cid, coord in coords.items():
                y = coord.symbol
                if isinstance(coord, UniformCoordinate):
                    a, b = coord.dist_coords['a'], coord.dist_coords['b']
                    val  = sp.integrate(val, (y, a, b))
                elif isinstance(coord, NormalCoordinate):
                    val  = sp.integrate(val, (y, -sp.oo, sp.oo))
                elif isinstance(coord, ExponentialCoordinate):
                    mu = coord.dist_coords['mu']
                    val = sp.integrate(val, (y, float(mu), sp.oo))
                else:
                    raise NotImplementedError(
                        f"Analytic integration not set up for {coord}"
                    )

            coeffs[k] = sp.simplify(val)

        return coeffs

    #-----------------------------------------------------------------#
    # Analytic decomposition (to fix)
    #-----------------------------------------------------------------#

    def decompose_analytic_to_fix(self, f_eval, f_deg: Counter):
        """
        Analytic decomposition using SymPy:
          c_k = ∫ f(y) ψ_k(y) ρ(y) dy over the domain.

        Arguments
        ---------
        f_eval : callable({cid: sympy.Symbol}) -> sympy.Expr
            User-supplied function of physical variables y.
        f_deg : Counter({cid: degree})
            Polynomial degrees of f (used in numeric path).

        Returns
        -------
        coeffs : dict {basis_id: sympy.Expr or constant}
        """
        coords  = self.coordinates
        symbols = {cid: coord.symbol for cid, coord in coords.items()}
        f_expr  = f_eval(symbols)

        coeffs = {}
        for k, psi_k in self.basis.items():
            # Build basis polynomial ψ_k(y) = ∏ ψ_{deg}(y_cid)
            psi_expr = 1
            for cid, deg in psi_k.items():
                y        = coords[cid].symbol
                psi_expr *= coords[cid].basis_at_y(y, deg)

            # Integrand in physical space
            integrand = f_expr * psi_expr

            # Nested univariate integrals with weight per axis
            val = integrand
            for cid, coord in coords.items():
                y = coord.symbol
                w = coord.weight(y)
                a, b = coord.domain()
                val  = sp.integrate(val * w, (y, a, b))

            coeffs[k] = sp.simplify(val)

        return coeffs

    #-----------------------------------------------------------------#
    # Consistency check
    #-----------------------------------------------------------------#

    def check_decomposition_consistency(self, f_eval, f_deg: Counter,
                                        tol=1e-10, verbose=True):
        """
        Cross-check numerical vs analytic decomposition.
        """
        coeffs_num = self.decompose(f_eval, f_deg)
        coeffs_sym = self.decompose_analytic(f_eval, f_deg)

        diffs, ok = {}, True
        for k in coeffs_num.keys():
            num_val = float(coeffs_num[k])
            try:
                ana_val = float(coeffs_sym[k].evalf())
            except TypeError:
                ana_val = float(sp.N(coeffs_sym[k], 15))
            err = abs(num_val - ana_val)
            if err > tol:
                ok = False
            diffs[k] = (num_val, ana_val, err)

        if verbose:
            print(f"[ConsistencyCheck] {'PASSED' if ok else 'FAILED'} "
                  f"with tol={tol}")
            for k, (n, a, e) in diffs.items():
                print(f"Basis {k}: num={n:.6g}, ana={a:.6g}, err={e:.2e}")

        return ok, diffs
