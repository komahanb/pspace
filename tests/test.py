import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
from pchaos.parameters import ParameterFactory, ParameterContainer

if __name__ == '__main__':
    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=4.0, sigma=0.50), 3)
    k = pfactory.createExponentialParameter('k', dict(mu=4.0, beta=0.50), 3)
    m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), 3)
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.addParameter(m)
    pc.initialize()

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
