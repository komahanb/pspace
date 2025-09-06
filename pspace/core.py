#!/usr/bin/env python
from __future__ import print_function

# External modules
import math

import numpy as np
np.set_printoptions(precision=3,suppress=True)

from collections import Counter
from enum        import Enum
from itertools   import product

# Local modules
from .stochastic_utils       import tensor_indices, nqpts, sparse
from .orthogonal_polynomials import unit_hermite
from .orthogonal_polynomials import unit_legendre
from .orthogonal_polynomials import unit_laguerre
from .plotter                import plot_jacobian, plot_vector

def index(ii):
    return ii
    if ii == 0:
        return 0
    if ii == 1:
        return 1
    if ii == 2:
        return 3
    if ii == 3:
        return 2

## TODO
# Parameters are Monomials
# Deterministic parameters are constant monomials (degree 0)
# Probabilistic parameters are highest degree monomials

class ParameterType(Enum):
    """
    Enumeration for types of parameters. Assumption is that any
    parameter that you create will either be 'deterministic' or
    'probabilistic' in nature.
    """
    DETERMINISTIC = 1
    PROBABILISTIC = 2

class DistributionType(Enum):
    """
    Enumeration of probability distribution types.
    """
    NONE        = 0
    NORMAL      = 1
    UNIFORM     = 2
    EXPONENTIAL = 3
    POISSON     = 4
    BINORMAL    = 5

class Parameter(object):
    """
    A hashable parameter object wrapping information about the parameter
    used in computations. Hashable implies being able to serve as keys
    dictionaries.

    Author: Komahan Boopathy

    """
    def __init__(self, pdata):
        self.param_id        = pdata['param_id']
        self.param_name      = pdata['param_name']
        self.param_type      = pdata['param_type']
        self.dist_type       = pdata['dist_type']
        self.monomial_degree = pdata['monomial_degree']
        return

    def __str__(self):
         return str(self.__class__.__name__) + " " + str(self.__dict__)

    def __hash__(self):
        return hash((self.param_id))

    def __eq__(self, other):
        return (self.param_id) == (other.param_id)

    def __ne__(self, other):
        return not(self == other)

    def getParameterValue(self):
        return self.param_value

    def getParameterType(self):
        return self.param_type

    def getDistributionType(self):
        return self.dist_type

    def getParameterID(self):
        return self.param_id

    def setParameterID(self, pid):
        self.param_id = pid
        return

    def getQuadraturePointsWeights(self, npoints):
        pass

    def evalOrthoNormalBasis(self, z, d):
        pass


    def checkConsistency(self, max_degree=5, npoints=20, tol=1e-12, verbose=True):
        """
        Check orthonormality of unit polynomials under quadrature.

        Parameters
        ----------
        max_degree : int
            Highest polynomial degree to check.
        npoints : int
            Number of quadrature points to use.
        tol : float
            Numerical tolerance for delta_{mn}.
        verbose : bool
            Print results if True.

        Returns
        -------
        ok : bool
            True if all checks pass within tolerance.
        errors : list
            List of (m,n,value) where error > tol.
        """
        # 1D quadrature from this parameter
        qmap = self.getQuadraturePointsWeights(npoints)
        z = qmap['zq']
        w = qmap['wq']

        errors = []
        ok = True

        # check inner products
        for m in range(max_degree+1):
            pm = self.evalOrthoNormalBasis(z, m)
            for n in range(max_degree+1):
                pn = self.evalOrthoNormalBasis(z, n)
                ip = np.sum(pm*pn*w)  # quadrature inner product
                target = 1.0 if m == n else 0.0
                if abs(ip - target) > tol:
                    ok = False
                    errors.append((m, n, ip))
                    if verbose:
                        print(f"Fail: <phi_{m}, phi_{n}> = {ip:.6e} (expected {target})")
        if verbose and ok:
            print(f"[{self.__class__.__name__}] consistency check passed "
                  f"for degrees ≤ {max_degree} with {npoints} points.")
        return ok, errors

class DeterministicParameter(Parameter):
    def __init__(self, pdata):
        super(DeterministicParameter, self).__init__(pdata)
        self.param_value = pdata['param_value']
        return

    def getQuadraturePointsWeights(self, npoints):
        cmap = {'yq' : self.param_value, 'zq' : self.param_value, 'wq' : 1.0}
        return cmap

    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate. Note: For
        the deterministic case, the value is always one.
        """
        return 1.0

class ProbabilisticParameter(Parameter):
    def __init__(self, pdata):
        super(ProbabilisticParameter, self).__init__(pdata)
        self.dist_params = pdata['dist_params']
        return

    def getDistributionParameters(self, key):
        return self.dist_params[key]

class ExponentialParameter(Parameter):
    def __init__(self, pdata):
        super(ExponentialParameter, self).__init__(pdata)
        self.dist_params = pdata['dist_params']
        return

    def getQuadraturePointsWeights(self, npoints):
        """
         numpy.polynomial.laguerre.laggauss(deg)[source]
        """

        # This is based on interval [0, \inf] with the weight
        # function f(xi) = \exp(-xi) which is also the standard
        # PDF f(z) = \exp(-z)

        xi, w = np.polynomial.laguerre.laggauss(npoints)
        mu    = self.dist_params['mu']
        beta  = self.dist_params['beta']

        # scale weights to unity (Area under exp(-xi) in [0,inf] is 1.0
        w = w/1.0

        # transformation of variables
        y = mu + beta*xi

        # assert if weights don't add up to unity
        eps = np.finfo(np.float64).eps
        assert((1.0 - 2.0*eps <= np.sum(w) <= 1.0 + 2.0*eps) == True)

        # Return quadrature point in standard space as well
        z = xi # (y-mu)/beta

        # Store in map
        cmap = {'yq' : y, 'zq' : z, 'wq' : w}
        return cmap

    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate.
        """
        return unit_laguerre(z,d)

class NormalParameter(Parameter):
    def __init__(self, pdata):
        super(NormalParameter, self).__init__(pdata)
        self.dist_params = pdata['dist_params']
        return

    def getQuadraturePointsWeights(self, npoints):
        # This is based on physicist unnormlized weight exp(-x*x).
        x, w = np.polynomial.hermite.hermgauss(npoints)
        mu    = self.dist_params['mu']
        sigma = self.dist_params['sigma']

        # scale weights to unity (Area under exp(-x*x) in [-inf,inf] is pi
        w = w/np.sqrt(np.pi)

        # transformation of variables
        y = mu + sigma*np.sqrt(2)*x

        # assert if weights don't add up to unity
        eps = np.finfo(np.float64).eps
        assert((1.0 - 2.0*eps <= np.sum(w) <= 1.0 + 2.0*eps) == True)

        # Return quadrature point in standard space as well
        z = (y-mu)/sigma

        # Store in map
        cmap = {'yq' : y, 'zq' : z, 'wq' : w}
        return cmap

    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate.
        """
        return unit_hermite(z, d)

class UniformParameter(Parameter):
    def __init__(self, pdata):
        super(UniformParameter, self).__init__(pdata)
        self.dist_params = pdata['dist_params']
        return

    def getQuadraturePointsWeights(self, npoints):
        # This is based on  weight 1.0 on interval [-1,1]
        x, w = np.polynomial.legendre.leggauss(npoints)
        a = self.dist_params['a']
        b = self.dist_params['b']

        # scale weights to unity
        w = w/2.0

        # transformation of variables
        y = (b-a)*x/2 + (b+a)/2

        # assert if weights don't add up to unity
        eps = np.finfo(np.float64).eps
        assert((1.0 - 2.0*eps <= np.sum(w) <= 1.0 + 2.0*eps) == True)

        # Return quadrature point in standard space as well
        z = (y-a)/(b-a)

        # Store in map
        cmap = {'yq' : y, 'zq' : z, 'wq' : w}
        return cmap

    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate.
        """
        return unit_legendre(z,d)

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class ParameterFactory:
    """
    This class takes in primitives and makes data strucuture required
    for other classes in this module (this is like creating elements).

    Deterministic parameter is a parameter whose polynomial degree is
    zero and stochastic parameter is a parameter whose polynomial
    degree is non zero.
    """
    def __init__(self):
        self.next_param_id = 0
        return

    def getParameterID(self):
        pid = self.next_param_id
        self.next_param_id = self.next_param_id + 1
        return pid

    def createDeterministicParameter(self, pname, pvalue):
        # Prepare map for calling constructor of deterministic
        # parameter
        pdata                    = {}
        pdata['param_name']      = pname
        pdata['param_type']      = ParameterType.DETERMINISTIC
        pdata['dist_type']       = DistributionType.NONE
        pdata['param_value']     = pvalue
        pdata['monomial_degree'] = 0
        pdata['param_id']        = self.getParameterID()
        return DeterministicParameter(pdata)

    def createNormalParameter(self, pname, dist_params, monomial_degree):
        # Prepare map for calling constructor of Normal/Gaussian
        # parameter
        pdata                    = {}
        pdata['param_name']      = pname
        pdata['param_type']      = ParameterType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.NORMAL
        pdata['dist_params']     = dist_params
        pdata['monomial_degree'] = monomial_degree
        pdata['param_id']        = self.getParameterID()
        return NormalParameter(pdata)

    def createUniformParameter(self, pname, dist_params, monomial_degree):
        # Prepare map for calling constructor of Uniform
        # parameter
        pdata                    = {}
        pdata['param_name']      = pname
        pdata['param_type']      = ParameterType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.UNIFORM
        pdata['dist_params']     = dist_params
        pdata['monomial_degree'] = monomial_degree
        pdata['param_id']        = self.getParameterID()
        return UniformParameter(pdata)

    def createExponentialParameter(self, pname, dist_params, monomial_degree):
        # Prepare map for calling constructor of Uniform
        # parameter
        pdata                    = {}
        pdata['param_name']      = pname
        pdata['param_type']      = ParameterType.PROBABILISTIC
        pdata['dist_type']       = DistributionType.EXPONENTIAL
        pdata['dist_params']     = dist_params
        pdata['monomial_degree'] = monomial_degree
        pdata['param_id']        = self.getParameterID()
        return ExponentialParameter(pdata)

class ParameterContainer:
    """
    Class that contains all stochastic parameters and handles
    quadrature and evaluation of basis functions.

    This object is simply a container for objects of type Parameter.

    Author: Komahan Boopathy

    """
    def __init__(self):
        self.num_terms = 1

        # container for storing all parameters
        self.parameter_map = {}

        # replace with DegreeSet class
        self.basistermwise_parameter_degrees = {} # For each parameter and basis entry what
                                             # is the degree according to tensor
                                             # product

        # Replace with basis class
        self.psi_map = {}

        return

    def __str__(self):
        return str(self.__class__.__name__) + " " + str(self.__dict__)

    def getNumParameters(self):
        return len(self.parameter_map.keys())

    def initialize(self):
        """
        Initialize the container after all 'paramters' are added.
        """
        # Create degree map
        sp_hd_map = self.getParameterHighestDegreeMap(exclude=ParameterType.DETERMINISTIC)
        self.basistermwise_parameter_degrees = tensor_indices(sp_hd_map)
        return

    def getNumQuadraturePoints(self):
        """
        """
        param_nqpts_map = Counter()
        pkeys = self.parameter_map.keys()
        for pid in pkeys:
            param_nqpts_map[pid] = self.getParameter(pid).monomial_degree
        return param_nqpts_map

    def getNumQuadraturePointsFromDegree(self,dmap):
        """
        Supply a map whose keys are parameterids and values are
        monomial degrees and this function will return a map with
        parameterids as keys and number of corresponding quadrature
        points along the monomial dimension.
        """
        ## pids = dmap.keys()
        ## param_nqpts_map = Counter()
        ## for pid in pids:
        ##     param_nqpts_map[pid] = nqpts(dmap[pid])
        ## return param_nqpts_map

        pids = dmap.keys()
        param_nqpts_map = Counter()
        for pid in self.parameter_map.keys(): #pids:
            param_nqpts_map[pid] = self.parameter_map[pid].monomial_degree #nqpts(dmap[pid])
        return param_nqpts_map

    def getParameterDegreeForBasisTerm(self, paramid, kthterm):
        """
        What is the polynomial degree of the corresponding k-th or
        Hermite/Legendre basis function? For univariate stochastic
        case d == k, but will change for multivariate case based on
        tensor product or other rules used to construct the
        multivariate basis set.
        """
        return self.basistermwise_parameter_degrees[kthterm][paramid]

    def getParameters(self):
        return self.parameter_map

    def getParameter(self, paramid):
        return self.parameter_map[paramid]

    def getParameterHighestDegreeMap(self, exclude=ParameterType.DETERMINISTIC):
        degree_map = {}
        for paramkey in self.parameter_map.keys():
            param = self.parameter_map[paramkey]
            if exclude != param.getParameterType():
                degree_map[paramkey] = param.monomial_degree
        return degree_map

    def getNumStochasticBasisTerms(self):
        return self.num_terms

    def addParameter(self, new_parameter):

        # do you want to hold separate maps for deterministic and
        # stochastic parameters?

        # Add parameter object to the map of parameters
        self.parameter_map[new_parameter.getParameterID()] = new_parameter

        # Increase the number of stochastic terms (tensor pdt rn)
        self.num_terms = self.num_terms*new_parameter.monomial_degree

        return

    def initializeQuadrature(self, param_nqpts_map):
        self.quadrature_map = self.getQuadraturePointsWeights(param_nqpts_map)
        return

    def W(self, q):
        wmap = self.quadrature_map[q]['W']
        return wmap

    def psi(self, k, zmap):
        paramids = zmap.keys()
        ans = 1.0
        for paramid in paramids:
            # Deterministic ones return one! maybe we can avoid!
            d   = self.getParameterDegreeForBasisTerm(paramid, k)
            val = self.getParameter(paramid).evalOrthoNormalBasis(zmap[paramid],d)
            ans = ans*val
        return ans

    def Z(self, q, key='pid'):
        if key == 'pid':
            # use pid as key
            return self.quadrature_map[q]['Z']
        else:
            # use name as key
            qmap = self.quadrature_map[q]['Z']
            nmap = {}
            for pid in qmap.keys():
                nmap[self.getParameter(pid).param_name] = qmap[pid]
            return nmap

    def Y(self, q, key='pid'):
        if key == 'pid':
            # use pid as key
            return self.quadrature_map[q]['Y']
        else:
            # use name as key
            qmap = self.quadrature_map[q]['Y']
            nmap = {}
            for pid in qmap.keys():
                nmap[self.getParameter(pid).param_name] = qmap[pid]
            return nmap

    def evalOrthoNormalBasis(self, k, q):
        return self.psi(k, self.Z(q))


    def getQuadraturePointsWeights(self, param_nqpts_map):
        """
        Return a map of quadrature point index → quadrature data (Y,Z,W).
        Works for arbitrary number of parameters.
        """
        pids  = list(param_nqpts_map.keys())
        nqpts = list(param_nqpts_map.values())

        # fetch 1D quadrature maps for each parameter
        maps = [self.getParameter(pid).getQuadraturePointsWeights(n)
                for pid, n in zip(pids, nqpts)]

        qmap = {}
        ctr = 0

        # Cartesian product of index ranges
        for idx_tuple in product(*[range(n) for n in nqpts]):
            yvec, zvec, w = {}, {}, 1.0
            for pid, i, m in zip(pids, idx_tuple, maps):
                yvec[pid] = m['yq'][i]
                zvec[pid] = m['zq'][i]
                w        *= m['wq'][i]
            qmap[ctr] = {'Y': yvec, 'Z': zvec, 'W': w}
            ctr += 1
        return qmap


    def getQuadraturePointsWeightsOld(self, param_nqpts_map):
        """
        Return a map of k : qmap, where k is the global basis index
        """
        params = list(param_nqpts_map.keys())
        nqpts  = list(param_nqpts_map.values())

        # exclude deterministic terms?
        total_quadrature_points = int(np.prod(nqpts))
        num_vars = len(params)

        # Initialize map with empty values corresponding to each key
        qmap = {}
        for key in range(total_quadrature_points):
            qmap[key] = []

        if num_vars == 1:

            # Get 1d-quadrature maps
            map0 = self.getParameter(0).getQuadraturePointsWeights(nqpts[0])
            pid0 = self.getParameter(0).getParameterID()

            # Tensor product of 1D-quadrature to get N-D quadrature
            ctr = 0

            for i0 in range(nqpts[0]):

                yvec = { pid0 : map0['yq'][i0] }

                zvec = { pid0 : map0['zq'][i0] }

                w  = map0['wq'][i0]

                data = {'Y' : yvec, 'Z' : zvec, 'W' : w}

                qmap[ctr] = data

                ctr += 1


            return qmap


        elif num_vars == 2:

            # Get 1d-quadrature maps
            map0 = self.getParameter(0).getQuadraturePointsWeights(nqpts[0])
            map1 = self.getParameter(1).getQuadraturePointsWeights(nqpts[1])

            pid0 = self.getParameter(0).getParameterID()
            pid1 = self.getParameter(1).getParameterID()

            # Tensor product of 1D-quadrature to get N-D quadrature
            ctr = 0

            for i0 in range(nqpts[0]):
                for i1 in range(nqpts[1]):

                    yvec = { pid0 : map0['yq'][i0],
                             pid1 : map1['yq'][i1] }

                    zvec = { pid0 : map0['zq'][i0],
                             pid1 : map1['zq'][i1] }

                        ## wvec = { pid0 : map0['wq'][i0],
                        ##          pid1 : map1['wq'][i1],
                        ##          pid2 : map2['wq'][i2] }

                    w  = map0['wq'][i0]*map1['wq'][i1]

                    data = {'Y' : yvec, 'Z' : zvec, 'W' : w}

                    qmap[ctr] = data

                    ctr += 1

            # Check if the sum of weights is one

            return qmap

        elif num_vars == 3:

            # Get 1d-quadrature maps
            map0 = self.getParameter(0).getQuadraturePointsWeights(nqpts[0])
            map1 = self.getParameter(1).getQuadraturePointsWeights(nqpts[1])
            map2 = self.getParameter(2).getQuadraturePointsWeights(nqpts[2])

            pid0 = self.getParameter(0).getParameterID()
            pid1 = self.getParameter(1).getParameterID()
            pid2 = self.getParameter(2).getParameterID()

            # Tensor product of 1D-quadrature to get N-D quadrature
            ctr = 0

            for i0 in range(nqpts[0]):
                for i1 in range(nqpts[1]):
                    for i2 in range(nqpts[2]):

                        yvec = { pid0 : map0['yq'][i0],
                                 pid1 : map1['yq'][i1],
                                 pid2 : map2['yq'][i2] }

                        zvec = { pid0 : map0['zq'][i0],
                                 pid1 : map1['zq'][i1],
                                 pid2 : map2['zq'][i2] }

                        ## wvec = { pid0 : map0['wq'][i0],
                        ##          pid1 : map1['wq'][i1],
                        ##          pid2 : map2['wq'][i2] }

                        w  = map0['wq'][i0]*map1['wq'][i1]*map2['wq'][i2]

                        data = {'Y' : yvec, 'Z' : zvec, 'W' : w}

                        qmap[ctr] = data

                        ctr += 1

            # Check if the sum of weights is one

            return qmap

        elif num_vars == 4:

            # Get 1d-quadrature maps
            map0 = self.getParameter(0).getQuadraturePointsWeights(nqpts[0])
            map1 = self.getParameter(1).getQuadraturePointsWeights(nqpts[1])
            map2 = self.getParameter(2).getQuadraturePointsWeights(nqpts[2])
            map3 = self.getParameter(3).getQuadraturePointsWeights(nqpts[3])

            pid0 = self.getParameter(0).getParameterID()
            pid1 = self.getParameter(1).getParameterID()
            pid2 = self.getParameter(2).getParameterID()
            pid3 = self.getParameter(3).getParameterID()

            # Tensor product of 1D-quadrature to get N-D quadrature
            ctr = 0

            for i0 in range(nqpts[0]):
                for i1 in range(nqpts[1]):
                    for i2 in range(nqpts[2]):
                        for i3 in range(nqpts[3]):

                            yvec = { pid0 : map0['yq'][i0],
                                     pid1 : map1['yq'][i1],
                                     pid2 : map2['yq'][i2],
                                     pid3 : map3['yq'][i3]}


                            zvec = { pid0 : map0['zq'][i0],
                                     pid1 : map1['zq'][i1],
                                     pid2 : map2['zq'][i2],
                                     pid3 : map3['zq'][i3]}

                            w  = map0['wq'][i0]*map1['wq'][i1]*map2['wq'][i2]*map3['wq'][i3]

                            data = {'Y' : yvec, 'Z' : zvec, 'W' : w}

                            qmap[ctr] = data

                            ctr += 1

            # Check if the sum of weights is one

            return qmap

        elif num_vars == 5:

            # Get 1d-quadrature maps
            map0 = self.getParameter(0).getQuadraturePointsWeights(nqpts[0])
            map1 = self.getParameter(1).getQuadraturePointsWeights(nqpts[1])
            map2 = self.getParameter(2).getQuadraturePointsWeights(nqpts[2])
            map3 = self.getParameter(3).getQuadraturePointsWeights(nqpts[3])
            map4 = self.getParameter(4).getQuadraturePointsWeights(nqpts[4])

            pid0 = self.getParameter(0).getParameterID()
            pid1 = self.getParameter(1).getParameterID()
            pid2 = self.getParameter(2).getParameterID()
            pid3 = self.getParameter(3).getParameterID()
            pid4 = self.getParameter(4).getParameterID()

            # Tensor product of 1D-quadrature to get N-D quadrature
            ctr = 0

            for i0 in range(nqpts[0]):
                for i1 in range(nqpts[1]):
                    for i2 in range(nqpts[2]):
                        for i3 in range(nqpts[3]):
                            for i4 in range(nqpts[4]):

                                yvec = { pid0 : map0['yq'][i0],
                                         pid1 : map1['yq'][i1],
                                         pid2 : map2['yq'][i2],
                                         pid3 : map3['yq'][i3],
                                         pid4 : map4['yq'][i4]}


                                zvec = { pid0 : map0['zq'][i0],
                                         pid1 : map1['zq'][i1],
                                         pid2 : map2['zq'][i2],
                                         pid3 : map3['zq'][i3],
                                         pid4 : map4['zq'][i4]}

                                w  = map0['wq'][i0]*map1['wq'][i1]*map2['wq'][i2]*map3['wq'][i3]*map4['wq'][i4]

                                data = {'Y' : yvec, 'Z' : zvec, 'W' : w}

                                qmap[ctr] = data

                                ctr += 1
            return qmap

    def projectResidual(self, elem, time, res, X, v, dv, ddv):
        """
        Project the elements deterministic residual onto stochastic
        basis and place in global stochastic residual array
        """

        # size of deterministic element state vector
        ndisps = elem.numDisplacements()
        nnodes = elem.numNodes()
        nddof = ndisps*nnodes
        nsdof = ndisps*self.getNumStochasticBasisTerms()

        for i in range(self.getNumStochasticBasisTerms()):

            # Initialize quadrature with number of gauss points
            # necessary for i-th basis entry
            self.initializeQuadrature(
                self.getNumQuadraturePointsFromDegree(
                    self.basistermwise_parameter_degrees[i]
                    )
                )

            # Quadrature Loop
            rtmp = np.zeros((nddof))

            for q in self.quadrature_map.keys():

                # Set the parameter values into the element
                elem.setParameters(self.Y(q,'name'))

                # Create space for fetching deterministic residual
                # vector
                resq = np.zeros((nddof))
                uq   = np.zeros((nddof))
                udq  = np.zeros((nddof))
                uddq = np.zeros((nddof))

                # Obtain states at quadrature nodes
                for k in range(self.num_terms):
                    psiky = self.evalOrthoNormalBasis(k,q)
                    uq[:] += v[k*nddof:(k+1)*nddof]*psiky
                    udq[:] += dv[k*nddof:(k+1)*nddof]*psiky
                    uddq[:] += ddv[k*nddof:(k+1)*nddof]*psiky

                # Fetch the deterministic element residual
                elem.addResidual(time, resq, X, uq, udq, uddq)

                # Project the determinic element residual onto the
                # stochastic basis and place in global residual array
                psiq   = self.evalOrthoNormalBasis(i,q)
                alphaq = self.W(q)
                rtmp[:] += resq*psiq*alphaq

            # Distribute the residual
            for ii in range(nnodes):
                # Local indices
                listart = index(ii)*ndisps
                liend   = (index(ii)+1)*ndisps
                gistart = index(ii)*nsdof + i*ndisps
                giend   = index(ii)*nsdof + (i+1)*ndisps

                # Place in global residul array node by node
                #print(gistart, giend, listart, liend)
                res[gistart:giend] += rtmp[listart:liend]

        # print("res=", res)
        # plot_vector(res, 'stochatic-element-residual.pdf', normalize=True, precision=1.0e-6)

        return

    def projectJacobian(self,
                        elem,
                        time, J, alpha, beta, gamma,
                        X, v, dv, ddv):
        """
        Project the elements deterministic jacobian matrix onto
        stochastic basis and place in global stochastic jacobian matrix
        """
        # All stochastic parameters are assumed to be of degree 1
        # (constant terms)
        dmapf = Counter()
        for pid in self.parameter_map.keys():
            dmapf[pid] = 1

        # size of deterministic element state vector
        ndisps = elem.numDisplacements()
        nnodes = elem.numNodes()
        nddof = ndisps*nnodes
        nsdof = ndisps*self.getNumStochasticBasisTerms()

        for i in range(self.getNumStochasticBasisTerms()):
            imap = self.basistermwise_parameter_degrees[i]

            for j in range(self.getNumStochasticBasisTerms()):
                jmap = self.basistermwise_parameter_degrees[j]

                smap = sparse(imap, jmap, dmapf)

                if False not in smap.values():

                    dmap = Counter()
                    dmap.update(imap)
                    dmap.update(jmap)
                    dmap.update(dmapf)
                    nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)

                    # Initialize quadrature with number of gauss points
                    # necessary for i,j-th jacobian entry
                    self.initializeQuadrature(nqpts_map)

                    jtmp = np.zeros((nddof,nddof))

                    # Quadrature Loop
                    for q in self.quadrature_map.keys():

                        try:
                            elem.setParameters(self.Y(q,'name'))
                        except:
                            print('exception')
                            raise

                        # Create space for fetching deterministic
                        # jacobian, and state vectors that go as input
                        Aq   = np.zeros((nddof,nddof))
                        uq   = np.zeros((nddof))
                        udq  = np.zeros((nddof))
                        uddq = np.zeros((nddof))
                        for k in range(self.num_terms):
                            psiky = self.evalOrthoNormalBasis(k,q)
                            uq[:] += v[k*nddof:(k+1)*nddof]*psiky
                            udq[:] += dv[k*nddof:(k+1)*nddof]*psiky
                            uddq[:] += ddv[k*nddof:(k+1)*nddof]*psiky

                        # Fetch the deterministic element jacobian matrix
                        elem.addJacobian(time, Aq, alpha, beta, gamma, X, uq, udq, uddq)

                        # Project the determinic element jacobian onto the
                        # stochastic basis and place in the global matrix
                        psiziw = self.W(q)*self.evalOrthoNormalBasis(i,q)
                        psizjw = self.evalOrthoNormalBasis(j,q)
                        jtmp[:,:] += Aq*psiziw*psizjw

                    # Distribute blocks (16 times)
                    for ii in range(0,nnodes):
                        for jj in range(0,nnodes):

                            # Local indices
                            listart = index(ii)*ndisps
                            liend   = (index(ii)+1)*ndisps
                            ljstart = index(jj)*ndisps
                            ljend   = (index(jj)+1)*ndisps

                            gistart = index(ii)*nsdof + i*ndisps
                            giend   = index(ii)*nsdof + (i+1)*ndisps
                            gjstart = index(jj)*nsdof + j*ndisps
                            gjend   = index(jj)*nsdof + (j+1)*ndisps

                            if i == j:
                                J[gistart:giend, gjstart:gjend] += jtmp[listart:liend, ljstart:ljend]
                            else:
                                J[gistart:giend, gjstart:gjend] += jtmp[listart:liend, ljstart:ljend]
                                #J[gjstart:gjend, gistart:giend] += jtmp[listart:liend, ljstart:ljend]

        #print("J=", J)
        #plot_jacobian(J, 'stochatic-element-block.pdf', normalize=True, precision=1.0e-6)

        return

    def projectInitCond(self, elem, v, vd, vdd, xpts):
        """
        Project the elements deterministic initial condition onto
        stochastic basis and place in global stochastic init condition
        array
        """

        # size of deterministic element state vector
        ndisps = elem.numDisplacements()
        nnodes = elem.numNodes()
        nddof = ndisps*nnodes
        nsdof = ndisps*self.getNumStochasticBasisTerms()

        for k in range(self.getNumStochasticBasisTerms()):

            # Initialize quadrature with number of gauss points
            # necessary for k-th basis entry
            self.initializeQuadrature(
                self.getNumQuadraturePointsFromDegree(
                    self.basistermwise_parameter_degrees[k]
                    )
                )

            # Quadrature Loop
            utmp = np.zeros((nddof))
            udtmp = np.zeros((nddof))
            uddtmp = np.zeros((nddof))
            for q in self.quadrature_map.keys():

                # Set the paramter values into the element
                elem.setParameters(self.Y(q,'name'))

                # Create space for fetching deterministic initial
                # conditions
                uq = np.zeros((nddof))
                udq = np.zeros((nddof))
                uddq = np.zeros((nddof))

                # Fetch the deterministic initial conditions
                elem.getInitConditions(uq, udq, uddq, xpts)

                # Project the determinic initial conditions onto the
                # stochastic basis
                psizkw = self.W(q)*self.evalOrthoNormalBasis(k,q)
                utmp += uq*psizkw
                udtmp += udq*psizkw
                uddtmp += uddq*psizkw

            # Distribute values
            for ii in range(nnodes):
                # Local indices
                listart = index(ii)*ndisps
                liend   = (index(ii)+1)*ndisps
                gistart = index(ii)*nsdof + k*ndisps
                giend   = index(ii)*nsdof + (k+1)*ndisps

                # Place in initial condition array node after node
                v[gistart:giend] += utmp[listart:liend]
                vd[gistart:giend] += udtmp[listart:liend]
                vdd[gistart:giend] += uddtmp[listart:liend]

            ## # Replace numbers less than machine precision with zero to
            ## # avoid numerical issues
            ## if clean is True:
            ##     eps = np.finfo(np.float).eps
            ##     v[np.abs(v) < eps] = 0
            ##     vd[np.abs(vd) < eps] = 0
            ##     vdd[np.abs(vdd) < eps] = 0

        return

    def checkConsistency(self, max_degree=None, tol=1e-12, verbose=True):
        """
        Check orthonormality of the multivariate basis functions
        under the container's quadrature. Also prints a table of
        inner products for debugging.

        Parameters
        ----------
        max_degree : int or None
            Maximum number of basis terms to check. If None, uses
            all available basis terms.
        tol : float
            Numerical tolerance for delta_{ij}.
        verbose : bool
            Print results if True.

        Returns
        -------
        ok : bool
            True if all checks pass within tolerance.
        errors : list
            List of (i,j,value) where error > tol.
        gram : np.ndarray
            Inner product matrix (approximate identity).
        """
        # Ensure quadrature is initialized
        if not hasattr(self, "quadrature_map"):
            nqpts_map = self.getNumQuadraturePoints()
            self.initializeQuadrature(nqpts_map)

        nbasis = self.getNumStochasticBasisTerms()
        if max_degree is not None:
            nbasis = min(nbasis, max_degree+1)

        gram = np.zeros((nbasis, nbasis))
        errors = []
        ok = True

        # Build Gram matrix
        for i in range(nbasis):
            for j in range(nbasis):
                s = 0.0
                for q in self.quadrature_map.keys():
                    psi_i = self.evalOrthoNormalBasis(i,q)
                    psi_j = self.evalOrthoNormalBasis(j,q)
                    wq    = self.W(q)
                    s += psi_i * psi_j * wq
                gram[i,j] = s
                target = 1.0 if i == j else 0.0
                if abs(s - target) > tol:
                    ok = False
                    errors.append((i, j, s))

        if verbose:
            print(f"[ParameterContainer] Gram matrix for {nbasis} basis terms:")
            with np.printoptions(precision=3, suppress=True):
                print(gram)

            if ok:
                print(f"[ParameterContainer] consistency check passed "
                      f"for {nbasis} basis terms.")
            else:
                print(f"[ParameterContainer] FAILED: {len(errors)} inconsistencies found.")

        return ok, errors, gram


    def sparse(self, dmapi, dmapj, dmapf):
        smap = {}
        for key in dmapi.keys():
            if abs(dmapi[key] - dmapj[key]) <= dmapf[key]:
                smap[key] = True
            else:
                smap[key] = False
        return smap

    def getSymmetricNonZeroIndices(self, dmapf):
        nz = {}
        N = self.getNumStochasticBasisTerms()
        for i in range(N):
            dmapi = self.basistermwise_parameter_degrees[i]
            for j in range(i,N):
                dmapj = self.basistermwise_parameter_degrees[j]
                smap = self.sparse(dmapi, dmapj, dmapf)
                if False not in smap.values():
                    dmap = Counter()
                    dmap.update(dmapi)
                    dmap.update(dmapj)
                    dmap.update(dmapf)
                    nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)
                    nz[(i,j)] = nqpts_map
        return nz

    def getSparseJacobian(self, f, dmapf):
        # rename member functions for local readability
        w    = lambda q    : self.W(q)
        psiz = lambda i, q : self.evalOrthoNormalBasis(i,q)

        nzs = self.getSymmetricNonZeroIndices(dmapf)
        N   = self.getNumStochasticBasisTerms()
        A   = np.zeros((N, N))
        for index, nqpts in nzs.items():
            self.initializeQuadrature(nqpts)
            pids = self.getParameters().keys()
            i    = index[0]
            j    = index[1]
            for q in self.quadrature_map.keys():
                val      = w(q)*psiz(i,q)*psiz(j,q)*f(q)
                A[i, j] += val
                A[j, i] += val
        return A

    def getJacobian(self, f, dmapf):
        # rename member functions for local readability
        w    = lambda q    : self.W(q)
        psiz = lambda i, q : self.evalOrthoNormalBasis(i,q)

        N = self.getNumStochasticBasisTerms()
        A = np.zeros((N, N))
        for i in range(N):
            dmapi = self.basistermwise_parameter_degrees[i]
            for j in range(N):
                dmapj = self.basistermwise_parameter_degrees[j]

                dmap = Counter()
                dmap.update(dmapi)
                dmap.update(dmapj)
                dmap.update(dmapf)

                # add up the degree of both participating functions psizi
                # and psizj to determine the total degree of integrand
                nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)
                self.initializeQuadrature(nqpts_map)

                # Loop quadrature points
                pids = self.getParameters().keys()
                for q in self.quadrature_map.keys():
                    A[i,j] += w(q)*psiz(i,q)*psiz(j,q)*f(q)
        return A
