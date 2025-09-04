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

def test_standard_uniform():
    factory = ParameterFactory()
    dist_params = {'a': 0.0, 'b': 1.0}   # Uniform(0,1)
    param = factory.createUniformParameter("u_std", dist_params, monomial_degree=5)
    ok, errors = param.checkConsistency(max_degree=4, npoints=10)
    assert ok, f"Standard Uniform consistency failed: {errors}"
    print("Standard UniformParameter consistency check passed.")

def test_nonstandard_uniform():
    factory = ParameterFactory()
    dist_params = {'a': 2.0, 'b': 5.0}   # Uniform(2,5) — non-standard
    param = factory.createUniformParameter("u_nonstd", dist_params, monomial_degree=5)
    ok, errors = param.checkConsistency(max_degree=4, npoints=10)
    assert ok, f"Non-standard Uniform consistency failed: {errors}"
    print("Non-standard UniformParameter consistency check passed.")

def test_standard_normal():
    factory = ParameterFactory()
    dist_params = {'mu': 0.0, 'sigma': 1.0}  # Standard Normal N(0,1)
    param = factory.createNormalParameter("n_std", dist_params, monomial_degree=5)
    ok, errors = param.checkConsistency(max_degree=4, npoints=10)
    assert ok, f"Standard Normal consistency failed: {errors}"
    print("Standard NormalParameter consistency check passed.")

def test_nonstandard_normal():
    factory = ParameterFactory()
    dist_params = {'mu': 3.0, 'sigma': 2.0}  # Non-standard Normal N(3,4)
    param = factory.createNormalParameter("n_nonstd", dist_params, monomial_degree=5)
    ok, errors = param.checkConsistency(max_degree=4, npoints=10)
    assert ok, f"Non-standard Normal consistency failed: {errors}"
    print("Non-standard NormalParameter consistency check passed.")

def test_standard_exponential():
    factory = ParameterFactory()
    dist_params = {'mu': 0.0, 'beta': 1.0}   # Standard Exponential Exp(1)
    param = factory.createExponentialParameter("e_std", dist_params, monomial_degree=5)
    ok, errors = param.checkConsistency(max_degree=4, npoints=10)
    assert ok, f"Standard Exponential consistency failed: {errors}"
    print("Standard ExponentialParameter consistency check passed.")

def test_nonstandard_exponential():
    factory = ParameterFactory()
    dist_params = {'mu': 1.0, 'beta': 2.0}   # Shifted/Scaled Exponential
    param = factory.createExponentialParameter("e_nonstd", dist_params, monomial_degree=5)
    ok, errors = param.checkConsistency(max_degree=4, npoints=10)
    assert ok, f"Non-standard Exponential consistency failed: {errors}"
    print("Non-standard ExponentialParameter consistency check passed.")

def test_container_consistency():
    factory = ParameterFactory()
    pc = ParameterContainer()

    # Mix of standard and non-standard
    pc.addParameter(factory.createUniformParameter("u", {"a":0.0,"b":1.0}, monomial_degree=3))
    pc.addParameter(factory.createNormalParameter("n", {"mu":2.0,"sigma":1.5}, monomial_degree=3))
    pc.addParameter(factory.createExponentialParameter("e", {"mu":1.0,"beta":0.5}, monomial_degree=2))

    pc.initialize()

    ok, errors, gram = pc.checkConsistency(max_degree=5, tol=1e-10, verbose=True)
    ok, errors, gram = pc.checkConsistency(tol=1e-10, verbose=True)

    assert ok, f"Container consistency failed with errors: {errors}"
    print("ParameterContainer consistency check passed.")

if __name__ == "__main__":
    test_standard_uniform()
    test_nonstandard_uniform()

    test_standard_normal()
    test_nonstandard_normal()

    test_standard_exponential()
    test_nonstandard_exponential()

    test_container_consistency()

    #for n in range(5):
    #    testall(n+1)
