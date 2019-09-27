import forthogonal_polynomials as fortho
import numpy as np

fortho =  fortho.orthogonal_polynomials
z = np.random.rand()
for t in range(1):
    for d in range(10):
        print(d,fortho.hermite(z,d))
        print(d,fortho.legendre(z,d))
        print(d,fortho.laguerre(z,d))

import fgaussian_quadrature as fquad
fquad = fquad.gaussian_quadrature

mu = 5.0
sigma = 1.0
beta = 1.0
a = 0.0
b = 1.0

npoints = 10
z = np.zeros(npoints)
y = np.zeros(npoints)
w = np.zeros(npoints)

print("hermite")
fquad.hermite_quadrature(npoints, mu, sigma, z, y, w)
print(z)
print(y)
print(w,np.sum(w))

print("legendre")
fquad.legendre_quadrature(npoints, a, b, z, y, w)
print(z)
print(y)
print(w,np.sum(w))

print("laguerre")
fquad.laguerre_quadrature(npoints, mu, beta, z, y, w)
print(z)
print(y)
print(w,np.sum(w))



