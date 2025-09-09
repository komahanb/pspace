#=====================================================================#
# File to test jacobian matrix sparsity patterns
#=====================================================================#
# Author : Komahan Boopathy (komahan.boopathy@gmail.com)
#=====================================================================#

# system/os modules
import sys

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
outdir  = "tests/baseline"
os.makedirs(outdir, exist_ok = True)

# third party modules
import numpy as np
from collections import Counter

# local modules
from pspace.core import CoordinateFactory
from pspace.core import CoordinateSystem, BasisFunctionType

from pspace.plotter import plot_jacobian

if __name__ == '__main__':
    cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)
    cf = CoordinateFactory()

    #y0 = cf.createNormalCoordinate(0, 'y0', dict(mu=0.0, sigma=1.0), max_monomial_dof=1)
    #y1 = cf.createUniformCoordinate(1, 'y1', dict(a=-1, b=1),        max_monomial_dof=1)

    y0 = cf.createNormalCoordinate(cf.newCoordinateID(), 'y0', dict(mu = -4.0, sigma = 0.5), 3)
    y1 = cf.createExponentialCoordinate(cf.newCoordinateID(), 'y1', dict(mu = +6.0, beta = 1.0), 3)
    y2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'y2', dict(a = -5.0, b = 4.0), 3)

    cs.addCoordinateAxis(y0)
    cs.addCoordinateAxis(y1)
    cs.addCoordinateAxis(y2)

    cs.initialize()

    # decompose a vector to obtain coefficients
    f_eval = lambda z: 2 + 3*z[0] - z[1]
    f_deg  = Counter({0:1, 1:1, 2:0})
    coeffs = cs.decompose(f_eval, f_deg)
    print(coeffs)

    # inner product of two basis entries with f
    #val = cs.inner_product_basis(i_id=1, j_id=0, f_eval=f_eval, f_deg=f_deg)
    #print(val)


    stop

    # Domain Definition(ADAPTIVE, FIXED={TENSOR, COMPLETE})
    cfactory = CoordinateFactory()

    # With adaptive enrichment we can keep the complexity (basis set
    # size) tied to the intrinsic structure of the function to be
    # decomposed in the probabilistic domain, not the worst-case
    # degree cutoffs like 4, q5, 6

    # Random coordinates
    y0 = cfactory.createNormalCoordinate(cfactory.newCoordinateID(), 'y0', dict(mu = -4.0, sigma = 0.5), 3)
    y1 = cfactory.createExponentialCoordinate(cfactory.newCoordinateID(), 'y1', dict(mu = +6.0, beta = 1.0), 3)
    y2 = cfactory.createUniformCoordinate(cfactory.newCoordinateID(), 'y2', dict(a = -5.0, b = 4.0), 3)

    print(y0)
    print(y1)
    print(y2)

    # Deterministic coordinates (uniform distribution & monomial degree = 0)
    # x1 = cfactory.createUniformCoordinate('x1', dict(a=-2.0, b=2.0), 0)     # space
    # x2 = cfactory.createUniformCoordinate('x2', dict(a=-2.0, b=2.0), 0)
    # x3 = cfactory.createUniformCoordinate('x3', dict(a=-2.0, b=2.0), 0)
    # t  = cfactory.createUniformCoordinate('t',  dict(a=0.0, b=1.0), 0)      # time

    # Add "Coordinate" into "CoordinateSystem: create Axes in the Domain
    csystem = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

    csystem.addCoordinateAxis(y0)
    csystem.addCoordinateAxis(y1)
    csystem.addCoordinateAxis(y2)

    csystem.initialize()

    print(csystem)

    for i in range(csystem.getNumBasisFunctions()):
        print(csystem.evaluateBasis([0.0, 0.0, 0.0], csystem.basis[i]))


    # check inner_product(psi_i, psi_j)
    # check decompose(function)

    stop

    class Function:
        def __init__(self, func, dmap, name):
            self.func = func
            self.name = name
            self.dmap = dmap
            return

    # find a better way to do this
    def y(q, degree):
        ymap = pc.Y(q)
        return ymap[degree]

    def generate_baseline(F : Function):
        A = pc.getJacobian(F.func, F.dmap)
        np.save(os.path.join(outdir, f"matrix-full-assembly-{F.name}.npy"), A)
        np.savetxt(os.path.join(outdir, f"matrix-full-assembly-{F.name}.dat"), A, fmt='%f', delimiter=' ')

        A = pc.getSparseJacobian(F.func, F.dmap)
        np.save(os.path.join(outdir, f"matrix-sparse-assembly-{F.name}.npy"), A)
        np.savetxt(os.path.join(outdir, f"matrix-sparse-assembly-{F.name}.dat"), A, fmt='%f', delimiter=' ')


    # identity: y0^0 * y1^0 * y2^0
    func = lambda q : y(q,0) * y(q,1) * y(q,2)
    dmap = Counter()
    dmap[0] = 0; dmap[1] = 0; dmap[2] = 0;
    constant_function = Function(func, dmap, "identity")
    generate_baseline(constant_function)


    # identity: 2.0 * y0^0 * y1^0 * y2^0
    func = lambda q : 2.0 * y(q,0) * y(q,1) * y(q,2)
    dmap = Counter()
    dmap[0] = 0; dmap[1] = 0; dmap[2] = 0;
    constant_function = Function(func, dmap, "constant")
    generate_baseline(constant_function)

    # linear : define: y0^1 + y1^1 + y2^1
    dmap = Counter()
    dmap[0] = 1; dmap[1] = 1; dmap[2] = 1
    func = lambda q : y(q,0) + y(q,1) + y(q,2)
    linear_function = Function(func, dmap, "linear")
    generate_baseline(linear_function)

    # quadratic : define: y0^2 + y1^2 + y2^2
    dmap = Counter()
    dmap[0] = 2; dmap[1] = 2; dmap[2] = 2
    func = lambda q : y(q,0)**2 + y(q,1)**2 + y(q,2)**2
    quadratic_function = Function(func, dmap, "quadratic")
    generate_baseline(quadratic_function)

    # polynomial: y0^4 * y1 + y0 * y2 + y2^3
    dmap = Counter()
    dmap[0] = 4  # degree in y0
    dmap[1] = 1  # degree in y1
    dmap[2] = 3  # degree in y2

    func = lambda q : (y(q,0)**4 * y(q,1)) + (y(q,0) * y(q,2)) + (y(q,2)**3)
    poly_function = Function(func, dmap, "quintic")
    generate_baseline(poly_function)
