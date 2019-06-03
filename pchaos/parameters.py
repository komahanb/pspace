#!/usr/bin/env python
#from __future__ import print_function

# External modules
import numpy as np
np.set_printoptions(precision=3,suppress=True)
import math
from collections import Counter
from enum import Enum

# Local modules
from stochastic_utils import tensor_indices, nqpts, sparse
from orthogonal_polynomials import unit_hermite as Hhat
from orthogonal_polynomials import unit_legendre as Phat
from orthogonal_polynomials import unit_laguerre as Lhat
from plotter import plot_jacobian
    
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
    A hashable parameter object wrapping information about the
    parameter used in computations. Hashable implies being able to
    serve as keys dictionaries.

    Author: Komahan Boopathy
    """
    def __init__(self, pdata):
        self.param_id        = pdata['param_id']
        self.param_name      = pdata['param_name']
        self.param_type      = pdata['param_type']
        self.dist_type       = pdata['dist_type']
        self.monomial_degree = pdata['monomial_degree']
        self.basis_map       = {}
        self.quadrature_map  = {}
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
    
class DeterministicParameter(Parameter):
    def __init__(self, pdata):
        super(DeterministicParameter, self).__init__(pdata)
        self.param_value = pdata['param_value']
        return
        
    def getQuadraturePointsWeights(self, npoints):
        raise
        try:
            return self.quadrature_map[npoints] 
        except:
            # Store in map
            cmap = {'yq' : self.param_value, 'zq' : self.param_value, 'wq' : 1.0}
            self.quadrature_map[npoints] = cmap
            return cmap
    
    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate. If it
        exists in the map already, return the value from map. If not
        evaluate the orthonormal polynomial and return.

        Note: For the deterministic case, the value is always one.
        """
        zkey = hash(z)
        try:
            return self.basis_map[(d,zkey)]
        except:
            self.basis_map[(d,zkey)] = 1.0
            return val
    
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
        try:
            return self.quadrature_map[npoints]
        except:
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
            #assert((1.0 - eps <= np.sum(w) <= 1.0 + eps) == True)
    
            # Return quadrature point in standard space as well
            z = xi # (y-mu)/beta

            # Store in map
            cmap = {'yq' : y, 'zq' : z, 'wq' : w}
            self.quadrature_map[npoints] = cmap            
            return cmap


    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate. If it
        exists in the map already, return the value from map. If not
        evaluate the orthonormal polynomial and return.
        """
        zkey = hash((z,d))
        try:
            return self.basis_map[zkey]
        except:
            val = Lhat(z,d)
            self.basis_map[zkey] = val
            return val
    
class NormalParameter(Parameter):
    def __init__(self, pdata):
        super(NormalParameter, self).__init__(pdata)
        self.dist_params = pdata['dist_params']
        return

    def getQuadraturePointsWeights(self, npoints):
        try:
            return self.quadrature_map[npoints] 
        except:
            # This is based on physicist unnormlized weight exp(-x*x).
            x, w = np.polynomial.hermite.hermgauss(npoints)
            mu    = self.dist_params['mu']
            sigma = self.dist_params['sigma']
            
            # scale weights to unity (Area under exp(-x) in [0,inf] is 1.0
            w = w/np.sqrt(np.pi)
            
            # transformation of variables
            y = mu + sigma*np.sqrt(2)*x
    
            # assert if weights don't add up to unity
            eps = np.finfo(np.float64).eps
            assert((1.0 - eps <= np.sum(w) <= 1.0 + eps) == True)
    
            # Return quadrature point in standard space as well
            z = (y-mu)/sigma

            # Store in map
            cmap = {'yq' : y, 'zq' : z, 'wq' : w}
            self.quadrature_map[npoints] = cmap            
            return cmap

    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate. If it
        exists in the map already, return the value from map. If not
        evaluate the orthonormal polynomial and return.
        """        
        zkey = hash((z,d))
        try:
            return self.basis_map[zkey]
        except:
            val = Hhat(z,d)
            self.basis_map[zkey] = val
            return val        
        
class UniformParameter(Parameter):
    def __init__(self, pdata):
        super(UniformParameter, self).__init__(pdata)
        self.dist_params = pdata['dist_params']
        return

    def getQuadraturePointsWeights(self, npoints):                
        try:
            return self.quadrature_map[npoints]
        except:
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
            assert((1.0 - eps <= np.sum(w) <= 1.0 + eps) == True)
    
            # Return quadrature point in standard space as well
            z = (y-a)/(b-a)
            
            # Store in map
            cmap = {'yq' : y, 'zq' : z, 'wq' : w}
            self.quadrature_map[npoints] = cmap
            return cmap

    def evalOrthoNormalBasis(self, z, d):
        """
        Evaluate the orthonormal basis at supplied coordinate. If it
        exists in the map already, return the value from map. If not
        evaluate the orthonormal polynomial and return.
        """
        zkey = hash((z,d))
        try:
            return self.basis_map[zkey]
        except:
            val = Phat(z,d)
            self.basis_map[zkey] = val
            return val

class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.iteritems())))
    
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
    
    ## def evalOrthoNormalBasis(self, k, q):
    ##     print self.Z(q)
    ##     return self.psi(k, self.Z(q))
        
    def evalOrthoNormalBasis(self, k, q):
        zq = self.Z(q)
        zkey = HashableDict(zq)
        try:
            return self.psi_map[(k, zkey)]
        except:
            val = self.psi(k, zq)
            self.psi_map[(k,zkey)] = val
            return val

    def getQuadraturePointsWeights(self, param_nqpts_map):
        """
        Return a map of k : qmap, where k is the global basis index
        """

        ## TODO generalize and store things if necessary
        
        params = param_nqpts_map.keys()
        nqpts  = param_nqpts_map.values()
        
        # exclude deterministic terms?
        total_quadrature_points = np.prod(nqpts)
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
        n = elem.numDisplacements()*elem.numNodes()
        
        for i in range(self.getNumStochasticBasisTerms()):
                
            # Initialize quadrature with number of gauss points
            # necessary for i-th basis entry
            self.initializeQuadrature(
                self.getNumQuadraturePointsFromDegree(
                    self.basistermwise_parameter_degrees[i]
                    )
                )

            # Quadrature Loop
            for q in self.quadrature_map.keys():

                # Set the parameter values into the element
                elem.setParameters(self.Y(q,'name'))
                
                # Create space for fetching deterministic residual
                # vector
                resq = np.zeros((n))
                uq   = np.zeros((n))
                udq  = np.zeros((n))
                uddq = np.zeros((n))
    
                # Obtain states at quadrature nodes
                for k in range(self.num_terms):
                    psiky = self.evalOrthoNormalBasis(k,q)
                    uq[:] += v[k*n:(k+1)*n]*psiky
                    udq[:] += dv[k*n:(k+1)*n]*psiky
                    uddq[:] += ddv[k*n:(k+1)*n]*psiky

                # Fetch the deterministic element residual
                elem.addResidual(time, resq, X, uq, udq, uddq)
                
                # Project the determinic element residual onto the
                # stochastic basis and place in global residual array
                psiq   = self.evalOrthoNormalBasis(i,q)
                alphaq = self.W(q)        
                res[i*n:(i+1)*n] += resq*psiq*alphaq
    
        return
    
    def projectJacobian(self,
                        elem,
                        time, J, alpha, beta, gamma,
                        X, v, dv, ddv):
        print("JEnter", J)
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
        n = elem.numDisplacements()*elem.numNodes()
        
        for i in range(self.getNumStochasticBasisTerms()):
            imap = self.basistermwise_parameter_degrees[i]
            
            for j in range(i,self.getNumStochasticBasisTerms()):                
                jmap = self.basistermwise_parameter_degrees[j]
                
                smap = sparse(imap, jmap, dmapf)
                if False not in smap.values():
                    dmap = Counter()
                    dmap.update(imap)
                    dmap.update(jmap)
                    dmap.update(dmapf)
                    nqpts_map = self.getNumQuadraturePointsFromDegree(dmap)

                    # print i,j, nqpts_map
                    # Initialize quadrature with number of gauss points
                    # necessary for i,j-th jacobian entry
                    self.initializeQuadrature(nqpts_map)

                    jtmp = np.zeros((n,n))
                                    
                    # Quadrature Loop
                    for q in self.quadrature_map.keys():

                        try:
                            print('hey')
                            pmap = self.Y(q,'name')
                            print(type(pmap), type(elem), elem.__class__)
                            # Set the paramter values into the element
                            elem.setParameters(pmap)
                        except:
                            print('exception' )
    
                        # Create space for fetching deterministic
                        # jacobian, and state vectors that go as input
                        Aq   = np.zeros((n,n))
                        uq   = np.zeros((n))
                        udq  = np.zeros((n))
                        uddq = np.zeros((n))
                        for k in range(self.num_terms):
                            psiky = self.evalOrthoNormalBasis(k,q)
                            uq[:] += v[k*n:(k+1)*n]*psiky
                            udq[:] += dv[k*n:(k+1)*n]*psiky
                            uddq[:] += ddv[k*n:(k+1)*n]*psiky
                            
                        # Fetch the deterministic element jacobian matrix
                        elem.addJacobian(time, Aq, alpha, beta, gamma, X, uq, udq, uddq)
                                    
                        # Project the determinic element jacobian onto the
                        # stochastic basis and place in the global matrix
                        psiziw = self.W(q)*self.evalOrthoNormalBasis(i,q)
                        psizjw = self.evalOrthoNormalBasis(j,q)
                        jtmp[:,:] += Aq*psiziw*psizjw
                        #print("jtmp", i, j, jtmp[:,:])
    
                    # Add the scaled deterministic block to element jacobian
                    J[i*n:(i+1)*n,j*n:(j+1)*n] += jtmp[:,:]
                    
                    # If off diagonal add the symmetric counter part
                    if i != j:
                        np.set_printoptions(formatter={'float': '{: 0.3e}'.format})        
                        #print("jtmp", i, j, jtmp[:,:])
                        J[j*n:(j+1)*n,i*n:(i+1)*n] += jtmp[:,:]
        print("J", J)
                
        #plot_jacobian(J, 'stochatic-element-block.pdf', normalize= True, precision=1.0e-6)

        return

    def projectInitCond(self, elem, v, vd, vdd, xpts):
        """
        Project the elements deterministic initial condition onto
        stochastic basis and place in global stochastic init condition
        array
        """

        # size of deterministic element state vector
        n = elem.numDisplacements()*elem.numNodes()
        
        for k in range(self.getNumStochasticBasisTerms()):

            # Initialize quadrature with number of gauss points
            # necessary for k-th basis entry
            self.initializeQuadrature(
                self.getNumQuadraturePointsFromDegree(
                    self.basistermwise_parameter_degrees[k]
                    )
                )
            
            # Quadrature Loop
            for q in self.quadrature_map.keys():

                # Set the paramter values into the element
                elem.setParameters(self.Y(q,'name'))

                # Create space for fetching deterministic initial
                # conditions
                uq = np.zeros((n))
                udq = np.zeros((n))
                uddq = np.zeros((n))

                # Fetch the deterministic initial conditions
                elem.getInitConditions(uq, udq, uddq, xpts)

                # Project the determinic initial conditions onto the
                # stochastic basis
                psizkw = self.W(q)*self.evalOrthoNormalBasis(k,q)
                v[k*n:(k+1)*n] += uq*psizkw
                vd[k*n:(k+1)*n] += udq*psizkw
                vdd[k*n:(k+1)*n] += uddq*psizkw
    
            ## # Replace numbers less than machine precision with zero to
            ## # avoid numerical issues
            ## if clean is True:
            ##     eps = np.finfo(np.float).eps
            ##     v[np.abs(v) < eps] = 0
            ##     vd[np.abs(vd) < eps] = 0
            ##     vdd[np.abs(vdd) < eps] = 0
                
        return
