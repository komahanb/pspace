#!/usr/bin/env python

import numpy as np
from collections import Counter

# Import from your package
from pspace.core import CoordinateFactory, CoordinateSystem, BasisFunctionType, PolyFunction

def main():

    #-------------------------------------------------------------#
    # 1. Build coordinate system with 2 axes (uniform for demo)
    #-------------------------------------------------------------#

    cf = CoordinateFactory()

    # y0 ~ Uniform[-1, 1], max deg = 2
    coord0 = cf.createUniformCoordinate(
        coord_id=cf.newCoordinateID(),
        coord_name="y0",
        dist_coords={'a': -1.0, 'b': 1.0},
        max_monomial_dof=3
    )

    # y1 ~ Uniform[-1, 1], max deg = 2
    coord1 = cf.createUniformCoordinate(
        coord_id=cf.newCoordinateID(),
        coord_name="y1",
        dist_coords={'a': -1.0, 'b': 1.0},
        max_monomial_dof=2
    )

    # Tensor-degree basis
    cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)
    cs.addCoordinateAxis(coord0)
    cs.addCoordinateAxis(coord1)
    cs.initialize()

    print(f"Num basis functions = {cs.getNumBasisFunctions()}")

    #-------------------------------------------------------------#
    # 2. Define polynomial f(y) = 3 + 3*y0 + 3*y0^2*y1
    #-------------------------------------------------------------#

    polyf = PolyFunction([
        (3.0, Counter({})),              # constant
        (3.0, Counter({0: 1})),          # linear term y0
        (3.0, Counter({0: 2, 1: 1}))     # mixed quadratic
    ])

    #-------------------------------------------------------------#
    # 3. Run sparse vs full assembly check
    #-------------------------------------------------------------#

    print(polyf.degrees)

    ok, diffs = cs.check_decomposition_matrix_sparse_full(polyf, tol=1e-10, verbose=True)
    print("[Result] Matrix assembly check:", "PASSED" if ok else "FAILED")

if __name__ == "__main__":
    main()
