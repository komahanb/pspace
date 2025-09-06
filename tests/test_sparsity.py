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
from pspace.core import ParameterFactory
from pspace.core import ParameterContainer
from pspace.plotter import plot_jacobian

if __name__ == '__main__':
    # Domain Definition(ADAPTIVE, FIXED={TENSOR, COMPLETE})
    pfactory = ParameterFactory()

    # With adaptive enrichment we can keep the complexity (basis set
    # size) tied to the intrinsic structure of the function to be
    # decomposed in the probabilistic domain, not the worst-case
    # degree cutoffs like 4, 5, 6

    y1 = pfactory.createNormalParameter     ('y1', dict(mu = -4.0, sigma = 0.5), 3)
    y2 = pfactory.createExponentialParameter('y2', dict(mu = +6.0, beta  = 1.0), 3)
    y3 = pfactory.createUniformParameter    ('y3', dict(a  = -5.0, b     = 4.0), 3)

    # Add "Parameter" into "ParameterContainer: create Axes in the Domain
    pc = ParameterContainer()

    pc.addParameter(y1)
    pc.addParameter(y2)
    pc.addParameter(y3)

    pc.initialize()

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

        A = pc.getSparseJacobian(F.func, F.dmap)
        np.save(os.path.join(outdir, f"matrix-sparse-assembly-{F.name}.npy"), A)

    # constant: y0^0 * y1^0 * y2^0
    func = lambda q : y(q,0) * y(q,1) * y(q,2)
    dmap = Counter()
    dmap[0] = 0; dmap[1] = 0; dmap[2] = 0;
    constant_function = Function(func, dmap, "y1^0y2^0y3^0")

    # linear : define: y0^1 + y1^1 + y2^1
    dmap = Counter()
    dmap[0] = 1; dmap[1] = 1; dmap[2] = 1
    func = lambda q : y(q,0) + y(q,1) + y(q,2)
    linear_function = Function(func, dmap, "y1^1+y2^1+y3^1")

    # quadratic : define: y0^2 + y1^2 + y2^2
    dmap = Counter()
    dmap[0] = 2; dmap[1] = 2; dmap[2] = 2
    func = lambda q : y(q,0)**2 + y(q,1)**2 + y(q,2)**2
    quadratic_function = Function(func, dmap, "y1^2+y2^2+y3^2")
    generate_baseline(quadratic_function)
