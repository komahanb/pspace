#=====================================================================#
# Shared helpers for randomized test setups
#=====================================================================#

import random
from collections import Counter

import sympy as sp

from pspace.core import BasisFunctionType, PolyFunction
from pspace.numeric import CoordinateFactory
from pspace.verify import NumericCoordinateSystem

#=====================================================================#
# Global test flags
#=====================================================================#

ENABLE_ANALYTIC_TESTS = False

#=====================================================================#
# Helper : randomized coordinate factory with logging
#=====================================================================#

def random_coordinate(cf, cid, max_deg=4):
    coord_type = random.choice(["normal", "uniform", "exponential"])
    name = f"y{cid}"

    if coord_type == "normal":
        mu = random.uniform(-2.0, 2.0)
        sigma = random.uniform(0.5, 2.0)
        deg = random.randint(1, max_deg)
        coord = cf.createNormalCoordinate(
            cf.newCoordinateID(), name, dict(mu=mu, sigma=sigma), deg
        )
        print(f"[Coord {cid}] NORMAL(mu={mu:.3f}, sigma={sigma:.3f}), deg={deg}")
        return coord

    if coord_type == "uniform":
        a = random.uniform(-3.0, 0.0)
        b = a + random.uniform(1.0, 5.0)
        deg = random.randint(1, max_deg)
        coord = cf.createUniformCoordinate(
            cf.newCoordinateID(), name, dict(a=a, b=b), deg
        )
        print(f"[Coord {cid}] UNIFORM(a={a:.3f}, b={b:.3f}), deg={deg}")
        return coord

    # exponential branch
    mu = random.uniform(0.0, 2.0)
    beta = random.uniform(0.5, 2.0)
    deg = random.randint(1, max_deg)
    coord = cf.createExponentialCoordinate(
        cf.newCoordinateID(), name, dict(mu=mu, beta=beta), deg
    )
    print(f"[Coord {cid}] EXPONENTIAL(mu={mu:.3f}, beta={beta:.3f}), deg={deg}")
    return coord

#=====================================================================#
# Helper : build random polynomial (with cross terms)
#=====================================================================#

def random_polynomial(cs, max_deg=2, max_cross_terms=3):
    coords = list(cs.coordinates.keys())
    symbols = {cid: cs.coordinates[cid].symbol for cid in coords}
    fdeg = Counter()
    terms = []

    # Individual terms
    for cid in coords:
        deg = random.randint(0, max_deg)
        coeff = random.randint(1, 3)
        terms.append((coeff, Counter({cid: deg})))
        fdeg[cid] = max(fdeg.get(cid, 0), deg)

    # Cross terms
    if len(coords) >= 2:
        for _ in range(random.randint(0, max_cross_terms)):
            cids = random.sample(coords, k=random.randint(2, len(coords)))
            coeff = random.randint(1, 3)
            degs = Counter()
            for cid in cids:
                d = random.randint(1, max_deg)
                degs[cid] = d
                fdeg[cid] = max(fdeg.get(cid, 0), d)
            terms.append((coeff, degs))

    polyf = PolyFunction(terms, coordinates=cs.coordinates)

    fexpr = polyf(symbols)
    print(f"[Polynomial] f(y) = {sp.simplify(fexpr)}, degrees={dict(fdeg)}")

    return polyf

#=====================================================================#
# Common coordinate system setup
#=====================================================================#

def get_coordinate_system_type(basis_type, max_deg=4, max_coords=3):
    cf = CoordinateFactory()
    cs = NumericCoordinateSystem(basis_type)

    ncoords = random.randint(1, max_coords)
    print(f"[Setup] Using {ncoords} coordinates")
    for cid in range(ncoords):
        coord = random_coordinate(cf, cid, max_deg)
        cs.addCoordinateAxis(coord)

    cs.initialize()

    return cs
