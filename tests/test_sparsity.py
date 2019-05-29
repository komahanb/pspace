import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from collections import Counter
from timeit import default_timer as timer

from pchaos.parameters import ParameterFactory, ParameterContainer
from pchaos.plotter import plot_jacobian

## def sparsity(a, b):
##     """
##     Nonzero entries occur when the parameterwise(variablewise) degree
##     of function 'b' matches or exceeds the degree of the basis vector
##     'a'.
##     """
##     akeys = a.keys()
##     smap = {}
##     for akey in akeys:
##         if b[akey] - a[akey] <= 0:
##             smap[akey] = True
##         else:
##             smap[akey] = False
##     return smap

def sparse(dmapi, dmapj, dmapf):
    ## print ''
    ## print ''
    ## print 'map i ', dmapi
    ## print 'map j ', dmapj
    ## print 'map f ', dmapf
    smap = {}
    #print "keys", dmapi.keys()
    for key in dmapi.keys():
        #print 'key', key, "diff", abs(dmapi[key] - dmapj[key]), dmapf[key]
        if abs(dmapi[key] - dmapj[key]) <= dmapf[key]:
            smap[key] = True
        else:
            smap[key] = False    
    return smap

def getSymmetricNonZeroIndices(pc, dmapf):
    nz = {}
    N = pc.getNumStochasticBasisTerms()
    for i in range(N):
        dmapi = pc.basistermwise_parameter_degrees[i]        
        for j in range(i,N):
            dmapj = pc.basistermwise_parameter_degrees[j]
            smap = sparse(dmapi, dmapj, dmapf)
            if False not in smap.values():
                dmap = Counter()
                dmap.update(dmapi)
                dmap.update(dmapj)
                dmap.update(dmapf)
                nqpts_map = pc.getNumQuadraturePointsFromDegree(dmap)
                nz[(i,j)] = nqpts_map
    return nz

def getSparseJacobian(pc, f, dmapf):
    nzs = getSymmetricNonZeroIndices(pc, dmapf)
    N = pc.getNumStochasticBasisTerms()
    A = np.zeros((N, N))
    for index, nqpts in nzs.items():
        pc.initializeQuadrature(nqpts)
        pids = pc.getParameters().keys()
        i = index[0]
        j = index[1]
        for q in pc.quadrature_map.keys():
            val = w(q)*psiz(i,q)*psiz(j,q)*f(q)
            A[i, j] += val
            A[j, i] += val
    return A

def getJacobian(f, dmapf):
    """
    """    
    # Test getting ND quadrature points
    N = pc.getNumStochasticBasisTerms()
    A = np.zeros((N, N))

    for i in range(N):
        dmapi = pc.basistermwise_parameter_degrees[i]
        
        for j in range(N):
            dmapj = pc.basistermwise_parameter_degrees[j]

            dmap = Counter()
            dmap.update(dmapi)
            dmap.update(dmapj)
            dmap.update(dmapf)
                        
            # add up the degree of both participating functions psizi
            # and psizj to determine the total degree of integrand
            nqpts_map = pc.getNumQuadraturePointsFromDegree(dmap)
            pc.initializeQuadrature(nqpts_map)
            # print dmap, nqpts_map

            # Loop quadrature points
            pids = pc.getParameters().keys()
            for q in pc.quadrature_map.keys():
                A[i,j] += w(q)*psiz(i,q)*psiz(j,q)*f(q)
    return A

if __name__ == '__main__':
    """    
    """
    start = timer()
    
    # Create "Parameter" using "Parameter Factory" object
    pfactory = ParameterFactory()
    c = pfactory.createNormalParameter('c', dict(mu=-4.0, sigma=0.50), 4)
    k = pfactory.createExponentialParameter('k', dict(mu=6.0, beta=1.0), 5)
    m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), 6)
    d = pfactory.createUniformParameter('d', dict(a=-5.0, b=4.0), 3)

    # Add "Parameter" into "ParameterContainer"
    pc = ParameterContainer()
    pc.addParameter(c)
    pc.addParameter(k)
    pc.addParameter(m)
    pc.addParameter(d)
    
    pc.initialize()

    suffix = 'eee'
    
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

    ## #  y_1^4
    ## func = lambda q :  gy(q,1)*gy(q,1)*gy(q,1)*gy(q,1)
    ## dmap[0] = 0; dmap[1] = 4; dmap[2] = 0
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y3^4-' + suffix + '.pdf')
    
    ## # Nonzero constant
    ## func = lambda q : 1.0
    ## dmap[0] = 0 ;  dmap[1] = 0; dmap[2] = 0
    ## #A = getJacobian(func, dmap)
    ## A = getSparseJacobian(pc, func, dmap)
    ## plot_jacobian(A, 'sparsity-identity-' + suffix + '.pdf')

    ## # Linear in y_1
    ## func = lambda q : gy(q,0)
    ## dmap[0] = 1; dmap[1] = 0; dmap[2] = 0
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y1-' + suffix + '.pdf')

    ## # Linear in y_2
    ## func = lambda q : gy(q,1)
    ## dmap[0] = 0; dmap[1] = 1; dmap[2] = 0
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y2-' + suffix + '.pdf')

    ## # Linear in y_3
    ## func = lambda q : gy(q,2)
    ## dmap[0] = 0; dmap[1] = 0; dmap[2] = 1
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y3-' + suffix + '.pdf')

    ## # y_1 +  y_2 + y_3
    ## func = lambda q : gy(q,0) + gy(q,1) + gy(q,2)
    ## dmap[0] = 1; dmap[1] = 1; dmap[2] = 1
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y1+y2+y3-' + suffix + '.pdf')

    ## # y_1 *  y_2
    ## func = lambda q : gy(q,0) * gy(q,1)
    ## dmap[0] = 1; dmap[1] = 1; dmap[2] = 0
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y1y2-' + suffix + '.pdf')

    ## # y_2 *  y_3
    ## func = lambda q : gy(q,1) * gy(q,2)
    ## dmap[0] = 0; dmap[1] = 1; dmap[2] = 1
    ## #A = getJacobian(func, dmap)
    ## A = getSparseJacobian(pc, func, dmap)
    ## plot_jacobian(A, 'sparsity-y2y3-' + suffix + '.pdf')

    ## # y_1 *  y_3
    ## func = lambda q : gy(q,0) * gy(q,2)
    ## dmap[0] = 1; dmap[1] = 0; dmap[2] = 1
    ## #A = getJacobian(func, dmap)
    ## A = getSparseJacobian(pc, func, dmap)
    ## plot_jacobian(A, 'sparsity-y1y3-' + suffix + '.pdf')

    ## #  y_1^2
    ## func = lambda q : gy(q,0)*gy(q,0)
    ## dmap[0] = 2; dmap[1] = 0; dmap[2] = 0
    ## #A = getJacobian(func, dmap)
    ## A = getSparseJacobian(pc, func, dmap)
    ## plot_jacobian(A, 'sparsity-y1^2-' + suffix + '.pdf')

    ## #  y_2^2
    ## func = lambda q : gy(q,1)*gy(q,1)
    ## dmap[0] = 0; dmap[1] = 2; dmap[2] = 0
    ## #A = getJacobian(func, dmap)
    ## A = getSparseJacobian(pc, func, dmap)
    ## plot_jacobian(A, 'sparsity-y2^2-' + suffix + '.pdf')

    ## #  y_3^2
    ## func = lambda q : gy(q,2)*gy(q,2)
    ## dmap[0] = 0; dmap[1] = 0; dmap[2] = 2
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y3^2-' + suffix + '.pdf')

    ## #  y_3^4
    ## func = lambda q : gy(q,2)*gy(q,2)*gy(q,2) + gy(q,2)*gy(q,2)*gy(q,2)*gy(q,2)
    ## dmap[0] = 0; dmap[1] = 0; dmap[2] = 4
    ## A = getSparseJacobian(pc, func, dmap)
    ## #A = getJacobian(func, dmap)
    ## plot_jacobian(A, 'sparsity-y3^4-' + suffix + '.pdf')
        
    # y_1 * y_2 * y_3*y_4
    func = lambda q : gy(q,0) * gy(q,1) * gy(q,2) * gy(q,3)
    dmap[0] = 1; dmap[1] = 1; dmap[2] = 1; dmap[3] = 1
    print 'first'
    A = getSparseJacobian(pc, func, dmap)
    print 'second'
    A = getSparseJacobian(pc, func, dmap)
    print 'third'
    A = getSparseJacobian(pc, func, dmap)
    #A = getJacobian(func, dmap)
    plot_jacobian(A, 'sparsity-y1y2y3y4-' + suffix + '.pdf')    
    end = timer()
    print "elapsed time:", end - start
