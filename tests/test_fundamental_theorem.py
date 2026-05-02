#=====================================================================#
# test_fundamental_theorem.py
#
# Tests for the Fundamental Theorem of PCE:
#
#   residual := reconstruct(decompose(f)) - f  ≡  0   (full basis)
#
# This is the continuous analogue of the adaptive Completeness law:
#
#   Completeness:  grow(S, exhausted) == universe  (discrete / index set)
#   Fundamental:   reconstruct(decompose(f)) == f  (continuous / L²)
#
# Both are statements about lossless round-trip recovery.  The difference
# is the space: the adaptive law operates on the index set (combinatorial),
# the PCE law operates on the function space (analytic).
#
# On a truncated (adaptive) basis the residual is the projection error —
# the natural error metric for adaptive basis quality.
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
import random
from collections import Counter

from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType,
                         PolyFunction,
                         Operation)
from pspace.adaptive import (LevelByLevelStrategy,
                              LevelStarting,
                              MaxIterationsStopping)

from .test_utils import (random_coordinate,
                          random_polynomial,
                          get_coordinate_system_type)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

TOL = 1e-8

def _sample_points(cs, n=20, seed=0):
    """Draw n random sample points Y = {cid: value} inside each coordinate's domain.
    For unbounded axes (Normal: ±∞, Exponential: half-line) we sample within
    a ±3σ / [lower, lower+4·mean] window to stay in high-probability mass."""
    import math
    rng = random.Random(seed)
    points = []
    for _ in range(n):
        Y = {}
        for cid, coord in cs.coordinates.items():
            mu  = float(coord.mean())
            std = math.sqrt(float(coord.variance()))
            a, b = coord.domain()
            a = float(a) if not math.isinf(float(a)) else mu - 3.0 * std
            b = float(b) if not math.isinf(float(b)) else mu + 3.0 * std
            Y[cid] = rng.uniform(a, b)
        points.append(Y)
    return points


def _representable_polynomial(cs, seed=0):
    """Build a random polynomial guaranteed to lie in cs's representable space.

    Generates pure single-axis terms only, each with degree ≤ that axis's
    max_monomial_dof.  No cross terms, so total degree = per-axis degree and
    the basis always covers every monomial.  This guarantees roundtrip exactness.
    """
    rng = random.Random(seed)
    terms = []
    for cid, coord in cs.coordinates.items():
        for d in range(1, coord.degree + 1):
            c = rng.uniform(-3.0, 3.0)
            terms.append((c, Counter({cid: d})))
    terms.append((rng.uniform(-3.0, 3.0), Counter()))
    return PolyFunction(terms)


def _max_residual(residual, points):
    """‖residual‖∞ over the sample points."""
    return max(abs(residual(Y)) for Y in points)


# ──────────────────────────────────────────────────────────────────────────────
# Fundamental Theorem: full basis  →  residual ≡ 0
# ──────────────────────────────────────────────────────────────────────────────

class TestFundamentalTheorem:
    """
    reconstruct(decompose(f)) - f  ≡  0  when the basis is complete for f.

    Dual of TestCompleteness in test_adaptive_basis.py:
      Completeness:  grow(S, exhausted) == universe   (index set)
      Fundamental:   roundtrip residual == 0          (function space)
    """

    @pytest.mark.parametrize("trial", range(5))
    def test_roundtrip_residual_zero_total_degree(self, trial):
        """Randomized total-degree basis: residual must vanish at all sample points."""
        random.seed(trial)
        cs = get_coordinate_system_type(BasisFunctionType.TOTAL_DEGREE,
                                        max_deg=3, max_coords=2)
        f        = _representable_polynomial(cs, seed=trial)
        coeffs   = cs.decompose(f)
        f_approx = cs.reconstruct(coeffs)
        residual = f_approx - f
        assert _max_residual(residual, _sample_points(cs)) < TOL, (
            f"trial {trial}: roundtrip residual > {TOL}")

    @pytest.mark.parametrize("trial", range(5))
    def test_roundtrip_residual_zero_tensor_degree(self, trial):
        """Randomized tensor-degree basis: same law, different index set."""
        random.seed(trial + 10)
        cs = get_coordinate_system_type(BasisFunctionType.TENSOR_DEGREE,
                                        max_deg=3, max_coords=2)
        f        = _representable_polynomial(cs, seed=trial + 10)
        coeffs   = cs.decompose(f)
        f_approx = cs.reconstruct(coeffs)
        residual = f_approx - f
        assert _max_residual(residual, _sample_points(cs)) < TOL, (
            f"trial {trial}: roundtrip residual > {TOL}")

    def test_residual_is_polyfunctions_arithmetic(self):
        """
        f_approx - f must be computable via arithmetic, not lambda hacks.
        Verify that OrthoPolyFunction - PolyFunction returns a callable.
        """
        cf = CoordinateFactory()
        k  = cf.createUniformCoordinate(cf.newCoordinateID(), 'k',
                                         dict(a=0.0, b=1.0), max_monomial_dof=3)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k)
        cs.initialize()

        f        = PolyFunction([(2.0, Counter()), (3.0, Counter({k.id: 1}))])
        coeffs   = cs.decompose(f)
        f_approx = cs.reconstruct(coeffs)
        residual = f_approx - f          # arithmetic, not lambda
        Y = {k.id: 0.5}
        assert abs(residual(Y)) < TOL


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive basis: residual is the projection error
# ──────────────────────────────────────────────────────────────────────────────

class TestAdaptiveResidual:
    """
    On a truncated basis the residual is nonzero — it measures how well
    the adaptive subspace captures f.

    The discrete dual: an adaptive CS that has not exhausted the pool
    has missing modes.  The continuous dual: the
    PCE residual ||f - f_approx|| > 0 for the same truncation.

    Monotonicity law: enriching the adaptive basis can only decrease
    the residual (or keep it equal), never increase it.
    """

    def _make_cs2(self):
        cf = CoordinateFactory()
        k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                         dict(a=0.9, b=1.1), max_monomial_dof=3)
        k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                         dict(a=0.4, b=0.6), max_monomial_dof=3)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k1)
        cs.addCoordinateAxis(k2)
        cs.initialize()
        return cs, k1, k2

    def test_full_basis_zero_residual(self):
        """Full basis (degree 3) must satisfy the fundamental theorem exactly."""
        cs, k1, k2 = self._make_cs2()
        cs_full = cs.make_cs(3)
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 2})),
            (0.5, Counter({k1.id: 1, k2.id: 1})),
        ])
        coeffs   = cs_full.decompose(f)
        residual = cs_full.reconstruct(coeffs) - f
        assert _max_residual(residual, _sample_points(cs_full)) < TOL

    def test_truncated_basis_nonzero_residual(self):
        """A degree-1 basis cannot represent a degree-2 term: residual must be nonzero."""
        cs, k1, k2 = self._make_cs2()
        cs_trunc = cs.make_cs(1)        # only mean + linear modes
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 2})),  # degree-2 term — not in basis
        ])
        coeffs   = cs_trunc.decompose(f)
        residual = cs_trunc.reconstruct(coeffs) - f
        assert _max_residual(residual, _sample_points(cs_trunc)) > TOL

    def test_residual_decreases_monotonically_with_enrichment(self):
        """
        Residuals at level 1, 2, 3 must be non-increasing.
        Mirrors: an enriched active set can only add modes, never lose them.
        """
        cs, k1, k2 = self._make_cs2()
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 2})),
            (0.5, Counter({k1.id: 1, k2.id: 1})),
            (0.3, Counter({k1.id: 2, k2.id: 1})),
        ])
        pts = _sample_points(cs)
        prev_err = float('inf')
        for deg in range(1, 4):
            cs_d     = cs.make_cs(deg)
            coeffs   = cs_d.decompose(f)
            residual = cs_d.reconstruct(coeffs) - f
            err = _max_residual(residual, pts)
            assert err <= prev_err + TOL, (
                f"degree {deg}: residual {err:.2e} > previous {prev_err:.2e} "
                f"(monotonicity violated)")
            prev_err = err

    def test_adaptive_residual_le_full_truncated(self):
        """
        An adaptive basis grown by LevelByLevel to the same max_degree as
        a full total-degree basis must have the same (zero) residual on f
        that the full basis represents exactly.
        """
        cs, k1, k2 = self._make_cs2()
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 1})),
        ])
        pts = _sample_points(cs)
        cs_full     = cs.make_cs(3)
        cs_adaptive = cs.make_adaptive_cs(LevelByLevelStrategy(3))

        r_full     = _max_residual(cs_full.residual(f), pts)
        r_adaptive = _max_residual(cs_adaptive.residual(f), pts)
        assert r_adaptive <= r_full + TOL


# ──────────────────────────────────────────────────────────────────────────────
# residual_norm: L² norm of the PCE residual
# ──────────────────────────────────────────────────────────────────────────────

class TestResidualNorm:
    r"""
    cs.residual_norm(coeffs, f) = sqrt(<r, r>)  where r = reconstruct(coeffs) - f

    Function-space error metric dual to |pool \ active| in index space.
    Three laws mirror TestCompleteness and TestAdaptiveResidual:

      Full basis:    residual_norm == 0      (Fundamental Theorem in L2 norm)
      Truncation:    residual_norm > 0       (projection error)
      Monotonicity:  norm(degree k+1) <= norm(degree k)
    """

    def _make_cs2(self):
        cf = CoordinateFactory()
        k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                         dict(a=0.9, b=1.1), max_monomial_dof=3)
        k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                         dict(a=0.4, b=0.6), max_monomial_dof=3)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k1)
        cs.addCoordinateAxis(k2)
        cs.initialize()
        return cs, k1, k2

    def test_full_basis_norm_is_zero(self):
        """Full basis: residual_norm must be 0 (Fundamental Theorem in L²)."""
        cs, k1, k2 = self._make_cs2()
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 2})),
            (0.5, Counter({k1.id: 1, k2.id: 1})),
        ])
        cs_full = cs.make_cs(3)
        norm = cs_full.residual_norm(f)
        assert norm < 1e-8, f"Full basis residual_norm = {norm:.2e}, expected ~0"

    def test_truncated_basis_norm_is_positive(self):
        """Degree-1 basis cannot represent degree-2 terms: norm must be > 0."""
        cs, k1, k2 = self._make_cs2()
        f = PolyFunction([
            (1.0, Counter()),
            (3.0, Counter({k1.id: 2})),   # degree-2 term missing from basis
        ])
        cs_trunc = cs.make_cs(1)
        norm = cs_trunc.residual_norm(f)
        assert norm > 1e-8, f"Truncated basis residual_norm = {norm:.2e}, expected > 0"

    def test_norm_decreases_monotonically(self):
        """residual_norm at degree k+1 must be <= norm at degree k."""
        cs, k1, k2 = self._make_cs2()
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 2})),
            (0.5, Counter({k1.id: 1, k2.id: 1})),
            (0.3, Counter({k1.id: 2, k2.id: 1})),
        ])
        prev = float('inf')
        for deg in range(1, 4):
            cs_d = cs.make_cs(deg)
            norm = cs_d.residual_norm(f)
            assert norm <= prev + 1e-8, (
                f"degree {deg}: norm {norm:.2e} > prev {prev:.2e} "
                f"(monotonicity violated)")
            prev = norm

    def test_adaptive_norm_equals_full_for_representable_f(self):
        """LevelByLevel to max_degree must give same (zero) norm as full basis."""
        cs, k1, k2 = self._make_cs2()
        f = PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 1})),
        ])
        cs_full     = cs.make_cs(3)
        cs_adaptive = cs.make_adaptive_cs(LevelByLevelStrategy(3))
        assert cs_adaptive.residual_norm(f) <= cs_full.residual_norm(f) + 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# Operation contract on CoordinateSystem
#
# CoordinateSystem implements the Operation ABC:
#   forward(f)    = decompose(f)
#   inverse(c)    = reconstruct(c)
#   residual(f)   = inverse(forward(f)) - f  (via default Operation.residual)
#
# These tests pin that forward/inverse are true aliases (not copies) and that
# residual(f) evaluates identically to reconstruct(decompose(f)) - f.
# ──────────────────────────────────────────────────────────────────────────────

class TestOperationContract:
    """
    CoordinateSystem must satisfy the Operation interface contract.
    forward/inverse are aliases; residual inherits the default implementation.
    """

    def _make_cs2(self):
        cf = CoordinateFactory()
        k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                         dict(a=0.9, b=1.1), max_monomial_dof=3)
        k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                         dict(a=0.4, b=0.6), max_monomial_dof=3)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k1)
        cs.addCoordinateAxis(k2)
        cs.initialize()
        return cs, k1, k2

    def _make_f(self, k1, k2):
        return PolyFunction([
            (1.0, Counter()),
            (2.0, Counter({k1.id: 1})),
            (1.5, Counter({k2.id: 1})),
        ])

    def test_cs_is_operation_instance(self):
        cs, _, _ = self._make_cs2()
        assert isinstance(cs.make_cs(2), Operation)

    def test_forward_equals_decompose(self):
        """forward(f) must return the same coefficients as decompose(f)."""
        cs, k1, k2 = self._make_cs2()
        cs2 = cs.make_cs(2)
        f   = self._make_f(k1, k2)
        assert cs2.forward(f) == cs2.decompose(f)

    def test_inverse_equals_reconstruct_at_sample_points(self):
        """inverse(c) must evaluate identically to reconstruct(c)."""
        cs, k1, k2 = self._make_cs2()
        cs2 = cs.make_cs(2)
        f   = self._make_f(k1, k2)
        c   = cs2.decompose(f)
        pts = _sample_points(cs2)
        rec = cs2.reconstruct(c)
        inv = cs2.inverse(c)
        for Y in pts:
            assert abs(rec(Y) - inv(Y)) < TOL, (
                f"inverse({Y}) = {inv(Y):.6g} != reconstruct = {rec(Y):.6g}")

    def test_residual_matches_manual_at_sample_points(self):
        """cs.residual(f) must agree with reconstruct(decompose(f)) - f pointwise."""
        cs, k1, k2 = self._make_cs2()
        cs1 = cs.make_cs(1)
        f   = self._make_f(k1, k2)   # degree-1: fully representable at degree 1
        pts = _sample_points(cs1)
        res_method = cs1.residual(f)
        res_manual = cs1.reconstruct(cs1.decompose(f))
        for Y in pts:
            method_val = float(res_method(Y))
            manual_val = float(res_manual(Y)) - float(f(Y))
            assert abs(method_val - manual_val) < TOL, (
                f"residual mismatch at {Y}: method={method_val:.6g}, "
                f"manual={manual_val:.6g}")

    def test_residual_zero_for_representable_f(self):
        """For a degree-1 f at a degree-1 CS, residual must be ~0 everywhere."""
        cs, k1, k2 = self._make_cs2()
        cs1 = cs.make_cs(1)
        f   = self._make_f(k1, k2)
        pts = _sample_points(cs1)
        res = cs1.residual(f)
        for Y in pts:
            assert abs(float(res(Y))) < TOL, (
                f"residual non-zero at {Y}: {float(res(Y)):.2e}")

    def test_forward_inverse_roundtrip_at_sample_points(self):
        """inverse(forward(f)) must reproduce f exactly for representable f."""
        cs, k1, k2 = self._make_cs2()
        cs2 = cs.make_cs(2)
        f   = self._make_f(k1, k2)
        pts = _sample_points(cs2)
        recovered = cs2.inverse(cs2.forward(f))
        for Y in pts:
            assert abs(float(recovered(Y)) - float(f(Y))) < TOL, (
                f"roundtrip mismatch at {Y}")

