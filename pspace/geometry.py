#!/usr/bin/env python
#=====================================================================#
# GEOMETRY MODULE FOR PSPACE
# Author: Komahan Boopathy
#=====================================================================#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

#=====================================================================#
# Base Geometry Class
#=====================================================================#

class Geometry:
    """Abstract base class for geometry mappings."""

    def __init__(self, name="geometry"):
        self.name = name
        self.coeffs = None   # used by spectral geometry

    def Transform(self, coords, csystem=None):
        """Map logical → physical coordinates (to be overridden)."""
        raise NotImplementedError

    def represent(self, csystem, degree=1, color='gray', alpha=0.4):
        """Generate geometry points using basis-driven representation."""
        qmap = csystem.build_quadrature(Counter({
            cid: degree for cid in csystem.coordinates.keys()
        }))
        pts = np.array([
            list(self.Transform(q['Y'], csystem).values())
            for q in qmap.values()
        ])
        self.show(pts, color=color, alpha=alpha)

    def show(self, pts, color='gray', alpha=0.4):
        """Plot geometry points in 3D."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, alpha=alpha, s=20)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(self.name)
        plt.show()


class IndependentGeometry(Geometry):
    """Simple Cartesian geometry — coordinates are independent."""

    def Transform(self, coords, csystem=None):
        return {'x': coords.get(0, 0.0),
                'y': coords.get(1, 0.0),
                'z': coords.get(2, 0.0)}

class CylindricalGeometry(Geometry):
    """Cylindrical mapping: (r,θ,z) → (x,y,z)."""

    def Transform(self, coords, csystem=None):
        # Expect coords = {0:r, 1:θ, 2:z}
        r  = float(coords.get(0, 0.0))
        th = float(coords.get(1, 0.0))
        z  = float(coords.get(2, 0.0))
        return {'x': r*np.cos(th), 'y': r*np.sin(th), 'z': z}

class SpectralGeometry(Geometry):
    """
    Geometry represented as a spectral field:
        x(ξ,η,ζ) = Σ a_k^x ψ_k(ξ,η,ζ)
        y(ξ,η,ζ) = Σ a_k^y ψ_k(ξ,η,ζ)
        z(ξ,η,ζ) = Σ a_k^z ψ_k(ξ,η,ζ)
    """

    def __init__(self, name="spectral_geometry", coeffs=None):
        super().__init__(name)
        self.coeffs = coeffs or {}

    def InitializeFromBounds(self, csystem):
        """Seed modal coefficients using coordinate bounds."""
        self.coeffs = {}
        for k in csystem.basis.keys():
            # simplest start: constant mode gets midpoints, others = 0
            self.coeffs[k] = np.zeros(3)

        # populate lowest mode
        bounds = []
        for cid, coord in csystem.coordinates.items():
            a, b = coord.dist_coords.get('a', 0), coord.dist_coords.get('b', 1)
            bounds.append(float(a))
            bounds.append(float(b))
        avg = np.mean(bounds)
        self.coeffs[0] = np.array([avg, avg, avg])

    def Transform(self, coords, csystem):
        """Evaluate geometry field at logical coordinates."""
        x, y, z = 0.0, 0.0, 0.0
        for k, degs in csystem.basis.items():
            ψ = csystem.evaluateBasisDegreesY(coords, degs)
            ax, ay, az = self.coeffs[k]
            x += ax * ψ; y += ay * ψ; z += az * ψ
        return {'x': x, 'y': y, 'z': z}
