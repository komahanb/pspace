#=====================================================================#
# Test: PolyFunction → coeffs → PolyFunction consistency
#=====================================================================#

import pytest

pytest.skip("Basis transform demo requires manual validation; skipping in automated tests", allow_module_level=True)

def test_polyfunction_roundtrip():
    import numpy as np
    from collections import Counter
    from pspace.core import CoordinateFactory, CoordinateSystem, BasisFunctionType, InnerProductMode, PolyFunction

    cf = CoordinateFactory()
    cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

    coord_x = cf.createUniformCoordinate(cf.newCoordinateID(), 'x', dict(a=-1, b=1), max_monomial_dof=2)
    coord_y = cf.createUniformCoordinate(cf.newCoordinateID(), 'y', dict(a=-1, b=1), max_monomial_dof=2)

    cs.addCoordinateAxis(coord_x)
    cs.addCoordinateAxis(coord_y)

    cs.initialize()

    f = PolyFunction([
        (2.0, Counter()),
        (3.0, Counter({coord_x.id: 1})),
        (4.0, Counter({coord_y.id: 2})),
        (5.0, Counter({coord_x.id: 1, coord_y.id: 1}))
    ], coordinates=cs.coordinates)

    coeffs = cs.decompose(f)
    print(coeffs)

    f_recon = cs.reconstruct(f, sparse=False, precondition=False, mode=InnerProductMode.NUMERICAL)

    # Compare values at sample points
    points = [
        {coord_x.id: -1.0, coord_y.id: -1.0},
        {coord_x.id:  0.2, coord_y.id: -0.4},
        {coord_x.id:  0.2, coord_y.id:  1.0},
    ]

    for pt in points:
        print(pt)
        f_val = f(pt)
        f_rec = f_recon(pt)
        assert abs(f_val - f_rec) < 1e-12, f"Mismatch at {pt}: {f_val} vs {f_rec}"

    print("[Roundtrip Test] Original and reconstructed PolyFunction agree at all test points.")


# Run test
if __name__ == "__main__":
    test_polyfunction_roundtrip()
