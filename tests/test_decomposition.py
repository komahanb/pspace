#=====================================================================#
# Randomized Decomposition Tests
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import random
import sympy as sp
import pytest
from collections import Counter

from pspace.core import (
    CoordinateFactory,
    CoordinateSystem,
    BasisFunctionType
)

#=====================================================================#
# Helper : randomized coordinate factory with logging
#=====================================================================#

def random_coordinate(cf, cid):
    coord_type = random.choice(["normal", "uniform", "exponential"])
    name       = f"y{cid}"

    if coord_type == "normal":
        mu    = random.uniform(-2.0,  2.0)
        sigma = random.uniform( 0.5,  2.0)
        deg   = random.randint(1, 4)
        coord = cf.createNormalCoordinate(cf.newCoordinateID(), name,
                                          dict(mu=mu, sigma=sigma), deg)
        print(f"[Coord {cid}] NORMAL(mu={mu:.3f}, sigma={sigma:.3f}), "
              f"deg={deg}")
        return coord

    elif coord_type == "uniform":
        a     = random.uniform(-3.0,  0.0)
        b     = a + random.uniform(1.0, 5.0)
        deg   = random.randint(1, 4)
        coord = cf.createUniformCoordinate(cf.newCoordinateID(), name,
                                           dict(a=a, b=b), deg)
        print(f"[Coord {cid}] UNIFORM(a={a:.3f}, b={b:.3f}), "
              f"deg={deg}")
        return coord

    elif coord_type == "exponential":
        mu    = random.uniform(0.0, 2.0)
        beta  = random.uniform(0.5, 2.0)
        deg   = random.randint(1, 4)
        coord = cf.createExponentialCoordinate(cf.newCoordinateID(), name,
                                               dict(mu=mu, beta=beta), deg)
        print(f"[Coord {cid}] EXPONENTIAL(mu={mu:.3f}, beta={beta:.3f}), "
              f"deg={deg}")
        return coord

#=====================================================================#
# Helper : build random polynomial (with cross terms)
#=====================================================================#

def random_polynomial(cs, max_deg=2, max_terms=3):
    coords  = list(cs.coordinates.keys())
    symbols = {cid : cs.coordinates[cid].symbol for cid in coords}
    fdeg    = Counter()
    terms   = []

    #---------------------------------------------------------------#
    # Individual terms
    #---------------------------------------------------------------#
    for cid in coords:
        deg        = random.randint(0, max_deg)
        coeff      = random.randint(1, 3)
        terms.append(coeff * symbols[cid]**deg)
        fdeg[cid]  = max(fdeg.get(cid, 0), deg)

    #---------------------------------------------------------------#
    # Cross terms
    #---------------------------------------------------------------#
    if len(coords) >= 2:
        for _ in range(random.randint(0, max_terms)):
            cids       = random.sample(coords, k=random.randint(2, len(coords)))
            coeff      = random.randint(1, 3)
            term       = coeff
            for cid in cids:
                d        = random.randint(1, max_deg)
                term    *= symbols[cid]**d
                fdeg[cid] = max(fdeg.get(cid, 0), d)
            terms.append(term)

    fexpr  = sum(terms)

    # numeric callable : Y is dict(cid -> float)
    fnum   = sp.lambdify([list(symbols.values())], fexpr, "numpy")
    dfunc  = lambda Y: fnum([Y[cid] for cid in coords])

    print(f"[Polynomial] f(y) = {fexpr}, degrees={dict(fdeg)}")
    return fexpr, dfunc, fdeg

#=====================================================================#
# Problem setup
#=====================================================================#

def get_coordinate_system_type(basis_type):
    cf = CoordinateFactory()
    cs = CoordinateSystem(basis_type)

    ncoords = random.randint(1, 3)
    print(f"[Setup] Using {ncoords} coordinates")
    for cid in range(ncoords):
        coord = random_coordinate(cf, cid)
        cs.addCoordinateAxis(coord)

    cs.initialize()

    return cs

#=====================================================================#
# Tests 1 B: Tensor Degree Basis (sparse vs full assembly)
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_tensor_basis_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Tensor Degree Basis (sparse vs full assembly)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE)

    fexpr, dfunc, fdeg = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(dfunc, fdeg,
                                                             tol=1e-6,
                                                             verbose=True)
    assert ok

#=====================================================================#
# Tests 2 B: Total Degree Basis (sparse vs full assembly)
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_total_basis_sparse_full(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Total Degree Basis (sparse vs full assembly)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE)

    fexpr, dfunc, fdeg = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_sparse_full(dfunc, fdeg,
                                                             tol=1e-6,
                                                             verbose=True)
    assert ok

#=====================================================================#
# Tests 1 A : Tensor Basis (numerical vs symbolic)
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_tensor_numerical_symbolic(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Tensor Degree Basis (numerical vs symbolic)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE)

    fexpr, dfunc, fdeg = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_symbolic(dfunc, fdeg,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok

#=====================================================================#
# Tests 2 A : Total Degree Basis (numerical vs symbolic)
#=====================================================================#

@pytest.mark.parametrize("trial", range(5))
def test_randomized_total_numerical_symbolic(trial):
    random.seed(trial)

    print(f"\n=== Trial {trial} : Total Degree Basis (numerical vs symbolic)  ===")

    cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE)

    fexpr, dfunc, fdeg = random_polynomial(cs)

    ok, diffs = cs.check_decomposition_numerical_symbolic(dfunc, fdeg,
                                                          tol=1e-6,
                                                          verbose=True)
    assert ok
