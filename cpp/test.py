import corthogonal_polynomials as cortho
import numpy as np

poly = cortho.PyOrthogonalPolynomials()
z = np.random.rand()
for t in range(1):
    print 'hermite'
    for d in range(10):
        print(d,poly.hermite(z,d) , poly.unit_hermite(z,d))
    print 'legendre'
    for d in range(10):
        print(d,poly.legendre(z,d), poly.unit_legendre(z,d))
    print 'laguerre'
    for d in range(10):
        print(d,poly.laguerre(z,d), poly.unit_laguerre(z,d))

