import forthogonal_polynomials as fortho
import numpy as np

fortho =  fortho.orthogonal_polynomials
z = np.random.rand()
for t in range(10000):
    for d in range(10):
        print(d,fortho.hermite(z,d))
        print(d,fortho.legendre(z,d))
        print(d,fortho.laguerre(z,d))

