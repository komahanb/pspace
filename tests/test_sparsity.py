import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
from pchaos.parameters import ParameterFactory, ParameterContainer
from pchaos.plotter import plot_jacobian
from collections import Counter
a  = Counter({0:1})
def getJacobian(f, dmapf):
    # Test getting ND quadrature points
    N = pc.getNumStochasticBasisTerms()    
    A = np.zeros((N, N))
    
    for i in range(N):        
        dmapi = pc.basistermwise_parameter_degrees[i]
        
        for j in range(N):            
            dmapj = pc.basistermwise_parameter_degrees[j]

            dmap = Counter()
            dmap.update(dmapf)
            dmap.update(dmapi)
            dmap.update(dmapj)
            
            # add up the degree of both participating functions psizi
            # and psizj to determine the total degree of integrand
            nqpts_map = pc.getNumQuadraturePointsFromDegree(dmap)
            pc.initializeQuadrature(nqpts_map)
            #print dmap, nqpts_map

            # Loop quadrature points
            pids = pc.getParameters().keys()
            for q in pc.quadrature_map.keys():
                A[i,j] += w(q)*psiz(i,q)*psiz(j,q)*f(q)

    return A

if __name__ == '__main__':
    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=-4.0, sigma=0.50), 4)
    k = pfactory.createExponentialParameter('k', dict(mu=6.0, beta=1.0), 5)
    m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), 6)
    
    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.addParameter(m)
    pc.initialize()

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

    # Default map for number of quadrature points
    dmap = Counter()
    
    # Nonzero constant
    func = lambda q : 1.0
    dmap[0] = 0; dmap[1] = 0; dmap[2] = 0


    ## print dmap
    ## ddmap = dmap + dmap
    ## print ddmap
    ## print pc.getNumQuadraturePointsFromDegree(dmap)

    ## stop
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-identity.pdf')

    # Linear in y_1
    func = lambda q : gy(q,0)
    dmap[0] = 1; dmap[1] = 0; dmap[2] = 0
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1.pdf')

    # Linear in y_2
    func = lambda q : gy(q,1)
    dmap[0] = 0; dmap[1] = 1; dmap[2] = 0
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y2.pdf')

    # Linear in y_3
    func = lambda q : gy(q,2)
    dmap[0] = 0; dmap[1] = 0; dmap[2] = 1
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y3.pdf')

    # y_1 +  y_2 + y_3
    func = lambda q : gy(q,0) + gy(q,1) + gy(q,2)
    dmap[0] = 1; dmap[1] = 1; dmap[2] = 1
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1+y2+y3.pdf')

    # y_1 *  y_2
    func = lambda q : gy(q,0) * gy(q,1)
    dmap[0] = 1; dmap[1] = 1; dmap[2] = 0
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1y2.pdf')

    # y_2 *  y_3
    func = lambda q : gy(q,1) * gy(q,2)
    dmap[0] = 0; dmap[1] = 1; dmap[2] = 1
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y2y3.pdf')

    # y_1 *  y_3
    func = lambda q : gy(q,0) * gy(q,2)
    dmap[0] = 1; dmap[1] = 0; dmap[2] = 1
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1y3.pdf')

    #  y_1^2
    func = lambda q : gy(q,0)*gy(q,0)
    dmap[0] = 2; dmap[1] = 0; dmap[2] = 0
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1^2.pdf')

    #  y_2^2
    func = lambda q : gy(q,1)*gy(q,1)
    dmap[0] = 0; dmap[1] = 2; dmap[2] = 0
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y2^2.pdf')

    #  y_3^2
    func = lambda q : gy(q,2)*gy(q,2)
    dmap[0] = 0; dmap[1] = 0; dmap[2] = 2
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y3^2.pdf')
        
    # y_1 * y_2 * y_3
    func = lambda q : gy(q,0) * gy(q,1) * gy(q,2)
    dmap[0] = 1; dmap[1] = 1; dmap[2] = 1
    A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1y2y3.pdf')


    
