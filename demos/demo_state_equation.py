#!/usr/bin/env python3
import numpy as np
from collections import Counter
from pspace.core import BasisFunctionType, PolyFunction, InnerProductMode
from pspace.numeric import (
    CoordinateFactory,
    NumericCoordinateSystem,
    StateEquation,
)

#---------------------------------------------------------------------#
# 1. Coordinate system setup
#---------------------------------------------------------------------#

cf = CoordinateFactory()
cs = NumericCoordinateSystem(BasisFunctionType.TENSOR_DEGREE, verbose=False)

coord_x = cf.createUniformCoordinate(cf.newCoordinateID(), "x", dict(a=-1, b=1), max_monomial_dof=2)
coord_y = cf.createUniformCoordinate(cf.newCoordinateID(), "y", dict(a=-1, b=1), max_monomial_dof=2)

cs.addCoordinateAxis(coord_x)
cs.addCoordinateAxis(coord_y)

cs.initialize()

print(f"Num basis functions = {cs.getNumBasisFunctions()}")

#---------------------------------------------------------------------#
# 2. Define operator and RHS
#---------------------------------------------------------------------#

# Operator: identity weight (Gram matrix)
gram_op = PolyFunction([(1.0, Counter())], coordinates=cs.coordinates)

# RHS: g(x,y) = 1 + 2x + 3y
rhs_fn = PolyFunction([
    (1.0, Counter()),
    (2.0, Counter({coord_x.id: 1})),
    (3.0, Counter({coord_y.id: 1}))
], coordinates=cs.coordinates)

#---------------------------------------------------------------------#
# 3. Build and assemble StateEquation
#---------------------------------------------------------------------#

eq = StateEquation("reconstruction", gram_op, rhs_fn, cs)
eq.assemble(mode=InnerProductMode.NUMERICAL, sparse=True)

print("\n[Operator matrix G_phi]")
print(np.round(eq.operator_matrix, 3))

print("\n[RHS vector]")
print(np.round(eq.rhs_vector, 3))

#---------------------------------------------------------------------#
# 4. Precondition and solve
#---------------------------------------------------------------------#

eq.precondition(method="cholesky")
a_phi = eq.solve()

print("\n[Whitened operator condition number]")
print(np.linalg.cond(eq.operator_whitened))

print("\n[Solved coefficients a_phi]")
print(np.round(a_phi, 6))

# Verify reconstruction consistency
residual = eq.operator_matrix @ a_phi - eq.rhs_vector
print("‖Residual‖₂ =", np.linalg.norm(residual))
