import numpy as np

import pspace.cwrap as uq


def _build_parameter_container():
    pfactory = uq.PyParameterFactory()
    y1 = pfactory.createNormalParameter(mu=1.0, sigma=0.1, dmax=2)
    y2 = pfactory.createUniformParameter(a=1.0, b=0.1, dmax=2)
    y3 = pfactory.createExponentialParameter(mu=1.0, beta=0.1, dmax=2)

    pc = uq.PyParameterContainer(1)
    pc.addParameter(y1)
    pc.addParameter(y2)
    pc.addParameter(y3)
    pc.initialize()
    return pc


def test_quadrature_shapes_and_weight():
    pc = _build_parameter_container()

    zq, yq = pc.quadrature(0)
    assert isinstance(zq, np.ndarray)
    assert isinstance(yq, np.ndarray)
    assert zq.shape == yq.shape == (pc.getNumParameters(),)

    wq, zq_w, yq_w = pc.quadrature(0, with_weight=True)
    assert np.isscalar(wq)
    assert np.allclose(zq_w, zq)
    assert np.allclose(yq_w, yq)


def test_basis_evaluation_returns_scalar():
    pc = _build_parameter_container()
    zq, _ = pc.quadrature(0)
    value = pc.basis(0, zq)
    assert np.isscalar(value)


def test_num_basis_terms_positive():
    pc = _build_parameter_container()
    assert pc.getNumBasisTerms() > 0
