from collections import Counter

from pspace.core import BasisFunctionType, CoordinateFactory, CoordinateSystem


def test_total_degree_sparse_vector_axis_cutoff():
    factory = CoordinateFactory()
    cs      = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)

    y0 = factory.createNormalCoordinate(factory.newCoordinateID(), 'y0', dict(mu=0.0, sigma=1.0), 4)
    y1 = factory.createUniformCoordinate(factory.newCoordinateID(), 'y1', dict(a=-1.0, b=1.0), 4)

    cs.addCoordinateAxis(y0)
    cs.addCoordinateAxis(y1)
    cs.initialize()

    exceeds_axis = Counter({0: 4, 1: 0})
    function_deg = Counter({0: 2, 1: 2})

    assert cs.sparse_vector(exceeds_axis, function_deg) is False


def test_total_degree_sparse_vector_within_axis():
    factory = CoordinateFactory()
    cs      = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)

    y0 = factory.createNormalCoordinate(factory.newCoordinateID(), 'y0', dict(mu=0.0, sigma=1.0), 3)
    y1 = factory.createUniformCoordinate(factory.newCoordinateID(), 'y1', dict(a=-1.0, b=1.0), 3)

    cs.addCoordinateAxis(y0)
    cs.addCoordinateAxis(y1)
    cs.initialize()

    within_axis   = Counter({0: 2, 1: 1})
    function_deg  = Counter({0: 2, 1: 2})

    assert cs.sparse_vector(within_axis, function_deg) is True
