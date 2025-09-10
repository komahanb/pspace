#!/usr/bin/env python

#=====================================================================#
# ABSTRACT MATHEMATICAL ANALYSIS MODULE FOR STOCHASTIC PARTIAL
# DIFFERENTIAL EQUATIONS
#=====================================================================#
# X   DOMAINS (PROBABILISTIC, SPATIAL, TEMPORAL)
# XX  DIMENSIONS (AXES) with PROBABILITY DISTRIBUTIONS
# XXX MODES (BASIS FUNCTIONS AND QUADRATURES)
#=====================================================================#
# Author : Komahan Boopathy (komahan.boopathy@gmail.com)
#=====================================================================#

# External modules
import math

import numpy as np
np.set_printoptions(precision=3,suppress=True)

from collections import Counter
from enum        import Enum
from itertools   import product

# Local modules
from .stochastic_utils       import minnum_quadrature_points, generate_basis_tensor_degree, sum_degrees, safe_zero_degrees
from .orthogonal_polynomials import unit_hermite
from .orthogonal_polynomials import unit_legendre
from .orthogonal_polynomials import unit_laguerre

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

class Coordinate(object):
    def __init__(self, coord_data):
        self.id           = coord_data['coord_id']
        self.name         = coord_data['coord_name']
        self.type         = coord_data['coord_type']
        self.distribution = coord_data['dist_type']
        self.degree       = coord_data['monomial_degree']

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    def getQuadraturePointsWeights(self, npoints):
        pass

    def evaluateBasisFunction(self, z, d):
        pass

class ExponentialCoordinate(Coordinate):
    def __init__(self, pdata):
        super(ExponentialCoordinate, self).__init__(pdata)
        self.dist_coords = pdata['dist_coords']

    def evaluateBasisFunction(self, zscalar, degree):
        return unit_laguerre(zscalar, degree)

    def getQuadraturePointsWeights(self, degree):
        # calculate the required number of quadrature points for the
        # degree
        npoints = minnum_quadrature_points(degree)

        #This is based on interval [0, \inf] with the weight function f(xi)
        # = \exp(-xi) which is also the standard PDF f(z) = \exp(-z).
        xi, w = np.polynomial.laguerre.laggauss(npoints)
        mu    = self.dist_coords['mu']
        beta  = self.dist_coords['beta']

        # scale weights to unity (Area under exp(-xi) in [0,inf] is 1.0
        w = w/1.0

        # transformation of variables
        y = mu + beta*xi

        # assert if weights don't add up to unity
        eps = np.finfo(np.float64).eps
        assert((1.0 - 2.0*eps <= np.sum(w) <= 1.0 + 2.0*eps) == True)

        # Return quadrature point in standard space as well
        z = xi

        return {'yq' : y, 'zq' : z, 'wq' : w}

class NormalCoordinate(Coordinate):
    def __init__(self, pdata):
        super(NormalCoordinate, self).__init__(pdata)
        self.dist_coords = pdata['dist_coords']

    def evaluateBasisFunction(self, zscalar, degree):
        return unit_hermite(zscalar, degree)

    def getQuadraturePointsWeights(self, degree):
        # calculate the required number of quadrature points for the
        # degree
        npoints = minnum_quadrature_points(degree)

        # This is based on physicist unnormlized weight exp(-x*x).
        x, w = np.polynomial.hermite.hermgauss(npoints)
        mu    = self.dist_coords['mu']
        sigma = self.dist_coords['sigma']

        # scale weights to unity (Area under exp(-x*x) in [-inf,inf] is pi
        w = w/np.sqrt(np.pi)

        # transformation of variables
        y = mu + sigma*np.sqrt(2)*x

        # assert if weights don't add up to unity
        eps = np.finfo(np.float64).eps
        assert((1.0 - 2.0*eps <= np.sum(w) <= 1.0 + 2.0*eps) == True)

        # Return quadrature point in standard space as well
        z = (y-mu)/sigma

        return {'yq' : y, 'zq' : z, 'wq' : w}

class UniformCoordinate(Coordinate):
    def __init__(self, pdata):
        super(UniformCoordinate, self).__init__(pdata)
        self.dist_coords = pdata['dist_coords']

    def evaluateBasisFunction(self, zscalar, degree):
        return unit_legendre(zscalar, degree)

    def getQuadraturePointsWeights(self, degree):
        # calculate the required number of quadrature points for the
        # degree
        npoints = minnum_quadrature_points(degree)

        # This is based on  weight 1.0 on interval [-1,1]
        x, w = np.polynomial.legendre.leggauss(npoints)
        a = self.dist_coords['a']
        b = self.dist_coords['b']

        # scale weights to unity
        w = w/2.0

        # transformation of variables
        y = (b-a)*x/2 + (b+a)/2

        # assert if weights don't add up to unity
        eps = np.finfo(np.float64).eps
        assert((1.0 - 2.0*eps <= np.sum(w) <= 1.0 + 2.0*eps) == True)

        # Return quadrature point in standard space as well
        z = (y-a)/(b-a)

        return {'yq' : y, 'zq' : z, 'wq' : w}

class CoordinateFactory:
    def __init__(self):
        self.next_coord_id = 0
        return

    def newCoordinateID(self):
        pid = self.next_coord_id
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
        pdata['coord_id']        = coord_id
        return NormalCoordinate(pdata)

    def createUniformCoordinate(self, coord_id, coord_name, dist_coords, max_monomial_dof):
        pdata                    = {}
        pdata['coord_id']        = coord_id
        pdata['coord_name']      = coord_name
        pdata['coord_type']      = CoordinateType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.UNIFORM
        pdata['dist_coords']     = dist_coords
        pdata['monomial_degree'] = max_monomial_dof
        pdata['coord_id']        = coord_id
        return UniformCoordinate(pdata)

    def createExponentialCoordinate(self, coord_id, coord_name, dist_coords, max_monomial_dof):
        pdata                    = {}
        pdata['coord_id']        = coord_id
        pdata['coord_name']      = coord_name
        pdata['coord_type']      = CoordinateType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.EXPONENTIAL
        pdata['dist_coords']     = dist_coords
        pdata['monomial_degree'] = max_monomial_dof
        pdata['coord_id']        = coord_id
        return ExponentialCoordinate(pdata)

class CoordinateSystem:
    """
    1. Stores all coordinates (axes, dimensions)
    2. Manages basis functions
    3. Manages integrations (inner-product) along these dimensions through quadrature
    """
    def __init__(self, basis_type, verbose = False):
        self.coordinates        = {}    # cid -> Coordinate
        self.basis_construction = basis_type
        self.basis              = None  # {basis_id: Counter({cid:deg,...})}
        self.verbose            = True

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    def getNumBasisFunctions(self):
        return len(self.basis)

    def getNumCoordinateAxes(self):
        return len(self.coordinates.keys())

    def getMonomialDegreeCoordinates(self):
        # map cid -> configured max degree for that coordinate (basis richness)
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

    def evaluateBasisDegrees(self, z, counter):
        val = 1.0
        for cid, cdeg in counter.items():
            val *= self.coordinates[cid].evaluateBasisFunction(z[cid], cdeg)
        return val

    def evaluateBasisIndex(self, z, basis_id):
        val = 1.0
        for cid, cdeg in counter.items():
            val *= self.coordinates[cid].evaluateBasisFunction(z[cid], self.basis[basis_id])
        return val

    def print_quadrature(self, qmap):
        """
        Pretty-print a quadrature map (qmap) in tabular style.
        """
        print("Quadrature rule:")
        print("-" * 80)
        for q, data in qmap.items():
            y_str = ", ".join(f"y{cid} = {val:.4g}" for cid, val in data['Y'].items())
            z_str = ", ".join(f"z{cid} = {val:.4g}" for cid, val in data['Z'].items())
            w_str = f"W = {data['W']:.4g}"
            print(f"q ={q:3d} :  {y_str:<40} | {z_str:<40} | {w_str}")
        print("-" * 80)

    def build_quadrature(self, degrees: Counter):
        """
        Build tensor-product quadrature from per-axis polynomial degree needs.
        degrees: Counter({cid: p_i}) — degree of the integrand along each axis.
        Uses each coordinate's 1D rule with n_i = minnum_quadrature_points(p_i).

        qmap =
        {
          q_index :
          {
            'Y': {cid: physical_value, ...},
            'Z': {cid: standard_value, ...},
            'W': weight
          },
          ...
        }

        """
        cids = list(self.coordinates.keys())
        q1d  = {}  # cid -> {'yq','zq','wq'}
        npts = {}  # cid -> Ni

        # obtain 1D rules per axis
        for cid in cids:
            p_i = int(degrees.get(cid, 0))
            one_d = self.coordinates[cid].getQuadraturePointsWeights(p_i)
            q1d[cid]  = one_d
            npts[cid] = len(one_d['wq'])

        # tensor them
        qmap = {}
        ctr  = 0
        ranges = [range(npts[cid]) for cid in cids]
        for idx_tuple in product(*ranges):
            y, z, w = {}, {}, 1.0
            for cid, i in zip(cids, idx_tuple):
                y[cid] = q1d[cid]['yq'][i]
                z[cid] = q1d[cid]['zq'][i]
                w     *= q1d[cid]['wq'][i]
            qmap[ctr] = {'Y': y, 'Z': z, 'W': w}
            ctr += 1

        if self.verbose is True:
            self.print_quadrature(qmap)

        return qmap

    # --- inner products & decomposition ---

    def inner_product(self, f_eval, g_eval, f_deg: Counter|None=None, g_deg: Counter|None=None):
        """
        <f, g> = ∫ f(z) g(z) ρ(z) dz, evaluated by exact quadrature if f_deg/g_deg supplied.
        - f_eval, g_eval: callables taking z_by_cid: dict(cid->z)
        - f_deg, g_deg: Counter({cid: degree}) describing polynomial degrees of f,g per axis.
                        If None, assumed 0 along each axis (safe but possibly under-integrated).
        """
        coord_ids = list(self.coordinates.keys())
        f_deg = f_deg or safe_zero_degrees(coord_ids)
        g_deg = g_deg or safe_zero_degrees(coord_ids)

        # Required per-axis polynomial degree for the integrand f*g
        need = sum_degrees(f_deg, g_deg)

        # Build exact-enough quadrature
        qmap = self.build_quadrature(need)

        s = 0.0
        for q in qmap.values():
            z = q['Z']
            s += f_eval(z) * g_eval(z) * q['W']
        return s

    def inner_product_basis(self, i_id: int, j_id: int, f_eval=None, f_deg: Counter|None=None):
        """
        <ψ_i, f, ψ_j> with exact quadrature deduced from degrees.
        - f_eval: callable(z_by_cid) or None (acts as 1.0)
        - f_deg:  Counter({cid: degree}) or None (treated as zeros)
        """
        psi_i = self.basis[i_id]
        psi_j = self.basis[j_id]

        coord_ids = list(self.coordinates.keys())
        f_deg = f_deg or safe_zero_degrees(coord_ids)

        # integrand degree per axis = deg(ψ_i)+deg(ψ_j)+deg(f)
        need = sum_degrees(psi_i, psi_j, f_deg)

        # Quadrature sized to integrate exactly
        qmap = self.build_quadrature(need)

        s = 0.0
        for q in qmap.values():
            z = q['Z']
            val = self.evaluateBasisDegrees(z, psi_i) * self.evaluateBasisDegrees(z, psi_j)
            if f_eval is not None:
                val *= f_eval(z)
            s += val * q['W']
        return s

    def decompose(self, f_eval, f_deg: Counter):
        """
        Coefficients c_k = <f, ψ_k>, with quadrature sized from deg(f) + deg(ψ_k).
        Returns dict {basis_id: coefficient}.
        """
        coeffs = {}
        for k, psi_k in self.basis.items():
            # per-axis degree need = deg(f) + deg(ψ_k)
            need = sum_degrees(f_deg, psi_k)
            qmap = self.build_quadrature(need)

            s = 0.0
            for q in qmap.values():
                z = q['Z']
                s += f_eval(z) * self.evaluateBasisDegrees(z, psi_k) * q['W']
            coeffs[k] = s
        return coeffs
