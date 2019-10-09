from pspace.core import ParameterFactory, ParameterContainer

# Create "Parameter" using "Parameter Factory" object
pfactory = ParameterFactory()
c = pfactory.createNormalParameter('c', dict(mu=-4.0, sigma=0.50), 4)
m = pfactory.createUniformParameter('m', dict(a=-5.0, b=4.0), 4)
k = pfactory.createExponentialParameter('k', dict(mu=6.0, beta=1.0), 5)
K = pfactory.createExponentialParameter('k', dict(mu=6.0, beta=1.0), 5)
C = pfactory.createNormalParameter('c', dict(mu=-4.0, sigma=0.50), 9)

# Add "Parameter" into "ParameterContainer"
pc = ParameterContainer()
pc.addParameter(c)
pc.addParameter(m)
pc.addParameter(k)
pc.addParameter(K)
pc.addParameter(C)

pc.initialize()

N = pc.getNumStochasticBasisTerms()
#zmap = {0:1.01, 1:1.00,2 :1.0001}
zmap = {0:1.01, 1:1.00, 2:1.0001, 3:2.0, 4:0.231312}

for j in range(1):
    for k in range(N):
        k+1, pc.psi(k,zmap)

## import forthogonal_polynomials as fortho
## import numpy as np

## fortho =  fortho.orthogonal_polynomials
## z = np.random.rand()
## for t in range(1):
##     for d in range(10):
##         print(d,fortho.hermite(z,d))
##         print(d,fortho.legendre(z,d))
##         print(d,fortho.laguerre(z,d))

## import fgaussian_quadrature as fquad
## fquad = fquad.gaussian_quadrature

## mu = 5.0
## sigma = 1.0
## beta = 1.0
## a = 0.0
## b = 1.0

## npoints = 10
## z = np.zeros(npoints)
## y = np.zeros(npoints)
## w = np.zeros(npoints)

## print("hermite")
## fquad.hermite_quadrature(npoints, mu, sigma, z, y, w)
## print(z)
## print(y)
## print(w,np.sum(w))

## print("legendre")
## fquad.legendre_quadrature(npoints, a, b, z, y, w)
## print(z)
## print(y)
## print(w,np.sum(w))

## print("laguerre")
## fquad.laguerre_quadrature(npoints, mu, beta, z, y, w)
## print(z)
## print(y)
## print(w,np.sum(w))
