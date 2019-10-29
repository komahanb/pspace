import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
from pspace.core import ParameterFactory, ParameterContainer

def univariate(dmax):    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=4.0, sigma=0.50), dmax[0])
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.initialize()

    test(pc)
    
def bivariate(dmax):    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=4.0, sigma=0.50), dmax[0])
    k = pfactory.createExponentialParameter('k', dict(mu=4.0, beta=0.50), dmax[1])
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.initialize()

    test(pc)
    
def trivariate(dmax):    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=4.0, sigma=0.50), dmax[0])
    k = pfactory.createExponentialParameter('k', dict(mu=4.0, beta=0.50), dmax[1])
    m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), dmax[2])
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.addParameter(m)
    pc.initialize()

    test(pc)
        
def quadvariate(dmax):    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=4.0, sigma=0.50), dmax[0])
    k = pfactory.createExponentialParameter('k', dict(mu=4.0, beta=0.50), dmax[1])
    m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), dmax[2])
    d = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), dmax[3])
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.addParameter(m)
    pc.addParameter(d)

    pc.initialize()

    test(pc)

def pentavariate(dmax):    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=4.0, sigma=0.50), dmax[0])
    k = pfactory.createExponentialParameter('k', dict(mu=4.0, beta=0.50), dmax[1])
    m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), dmax[2])
    d = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), dmax[3])
    e = pfactory.createExponentialParameter('k', dict(mu=4.0, beta=0.50), dmax[4])
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.addParameter(m)
    pc.addParameter(d)
    pc.addParameter(e)

    pc.initialize()

    test(pc)

def test(pc):
    # Test getting ND quadrature points
    N = pc.getNumStochasticBasisTerms()
    
    A = np.zeros((N, N))    

    for i in range(N):
        
        dmapi = pc.basistermwise_parameter_degrees[i]
        param_nqpts_mapi = pc.getNumQuadraturePointsFromDegree(dmapi)
        
        for j in range(N):
            
            dmapj = pc.basistermwise_parameter_degrees[j]
            param_nqpts_mapj = pc.getNumQuadraturePointsFromDegree(dmapj)

            # add up the degree of both participating functions psizi
            # and psizj to determine the total degree of integrand

            pc.initializeQuadrature(param_nqpts_mapi + param_nqpts_mapj)

            # rename for readability
            w    = lambda q    : pc.W(q)
            psiz = lambda i, q : pc.evalOrthoNormalBasis(i,q)

            def fy(q):
                ymap = pc.Y(q)
                paramids = ymap.keys()
                ans = 1.0
                for paramid in paramids:
                    ans = ans*ymap[paramid]
                return ans

            def gy(q,pid):
                ymap = pc.Y(q)
                return ymap[pid]

            # Constant function
            pids = pc.getParameters().keys()
            for q in pc.quadrature_map.keys():
                A[i,j] += w(q)*psiz(i,q)*psiz(j,q)
                
    assert(np.allclose(A, np.eye(A.shape[0])) == True) 

def testall(nmax):
    for i in range(nmax):
        print (i, univariate([i+1]))
        for j in range(nmax):
            print (i, j, bivariate([i+1,j+1]))
            for k in range(nmax):    
                print (i, j, k, trivariate([i+1,j+1,k+1]))
                for l in range(nmax):    
                    print (i, j, k, l, quadvariate([i+1,j+1,k+1, l+1]))
                    for m in range(nmax):    
                        print (i, j, k, l, m, pentavariate([i+1,j+1,k+1, l+1, m+1]))

if __name__ == '__main__':
    
    for n in range(5):
        testall(n+1)
