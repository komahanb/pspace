import numpy as np

from pspace.core import CoordinateSystem, CoordinateFactory, BasisFunctionType
from pspace.geometry import SpectralGeometry, CylindricalGeometry

cf = CoordinateFactory()
cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)

# Create simple coordinates
cs.addCoordinateAxis(cf.createUniformCoordinate(0, 'r', {'a':0, 'b':1}, 2))
cs.addCoordinateAxis(cf.createUniformCoordinate(1, 'θ', {'a':0, 'b':np.pi/2}, 2))
cs.addCoordinateAxis(cf.createUniformCoordinate(2, 'z', {'a':0, 'b':1}, 2))
cs.initialize()

# Set geometry
geom = SpectralGeometry("spectral_cylinder")
geom.InitializeFromBounds(cs)
cs.SetGeometry(geom)

# Represent
cs.represent(degree=2, color='cyan')
