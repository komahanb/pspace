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
from .stochastic_utils       import minnum_quadrature_points, generate_basis_tensor_degree
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
    def __init__(self, basis_type):
        self.coordinates        = {}    # p0, p1, p2, ...
        self.basis_construction = basis_type
        self.basis              = None

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__) + "\n"

    def getNumCoordinateAxes(self):
        return len(self.coordinates.keys())

    def getMonomialDegreeCoordinates(self):
        return {cid: coord.degree for cid, coord in self.coordinates.items()}

    def addCoordinateAxis(self, coordinate):
        self.coordinates[coordinate.id] = coordinate

    def initialize(self):
        if self.basis_construction == BasisFunctionType.TENSOR_DEGREE:
            self.basis = generate_basis_tensor_degree(self.getMonomialDegreeCoordinates())
        elif self.basis_construction == BasisFunctionType.TOTAL_DEGREE:
            self.basis = generate_basis_total_degree(self.getMonomialDegreeCoordinates())
        else:
            raise NOT_IMPLEMENTED

    def evaluateBasis(self, z, counter):
        val = 1.0
        for cid, cdeg in counter.items():
            val *= self.coordinates[cid].evaluateBasisFunction(z[cid], cdeg)
        return val
