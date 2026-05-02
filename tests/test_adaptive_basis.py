#=====================================================================#
# test_adaptive_basis.py
#
# Structural tests for adaptive basis selection.
#
# Design philosophy: test algebraic laws, not behaviors.
# The three governing laws for any strategy S and its reverse:
#
#   Completeness:  grow(S, exhausted)            == universe
#   Zero:          decay(S.reverse(), exhausted) == ∅
#   Involution:    S.reverse().reverse()  is  S
#
# Corner cases: DC guard, contract (type, IDs, find_modes), 36-combo smoke.
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
from collections import Counter

from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType)
from pspace.adaptive import (
    MeanOnlyStarting,
    LevelStarting,
    SensitivityStarting,
    FixedModeSetStarting,
    LevelByLevelStrategy,
    SensitivityDrivenStrategy,
    DownwardClosedStrategy,
    CandidatePoolExhaustedStopping,
    MaxIterationsStopping,
    RelativeGrowthStopping,
    _ReversedStrategy,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures and helpers
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cs2():
    """2-parameter Uniform coordinate system (k1, k2)."""
    cf = CoordinateFactory()
    k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                     dict(a=0.9, b=1.1), max_monomial_dof=3)
    k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                     dict(a=0.4, b=0.6), max_monomial_dof=3)
    cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
    cs.addCoordinateAxis(k1)
    cs.addCoordinateAxis(k2)
    cs.initialize()
    return cs


@pytest.fixture(scope="module")
def cs3():
    """3-parameter coordinate system (Normal, Uniform, Exponential)."""
    cf = CoordinateFactory()
    p0 = cf.createNormalCoordinate(cf.newCoordinateID(), 'p0',
                                    dict(mu=1.0, sigma=0.2), max_monomial_dof=3)
    p1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'p1',
                                     dict(a=0.8, b=1.2), max_monomial_dof=3)
    p2 = cf.createExponentialCoordinate(cf.newCoordinateID(), 'p2',
                                         dict(mu=0.5, beta=1.0), max_monomial_dof=3)
    cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
    cs.addCoordinateAxis(p0)
    cs.addCoordinateAxis(p1)
    cs.addCoordinateAxis(p2)
    cs.initialize()
    return cs


def _total(degs):
    return sum(degs.values())


def _strategies(cs, max_degree):
    """All three concrete strategies for a given coordinate system and degree."""
    v = [c.variance() for c in cs.coordinates.values()]
    return [
        LevelByLevelStrategy(max_degree),
        SensitivityDrivenStrategy(max_degree, v, batch_size=1),
        DownwardClosedStrategy(max_degree, batch_size=1),
    ]


def _all_startings(cs, max_degree):
    """All four starting criteria, using a downward-closed seed for FixedModeSet."""
    v  = [c.variance() for c in cs.coordinates.values()]
    dc = set(cs.make_cs(max_degree).find_modes(total_degree=max_degree - 1).keys())
    return [
        MeanOnlyStarting(),
        LevelStarting(max_degree - 1),
        SensitivityStarting(v),
        FixedModeSetStarting(dc),
    ]


def _active_levels(cs_a):
    """Return set of total degrees present in cs_a's basis."""
    return {_total(degs) for degs in cs_a.basis.values()}


# ──────────────────────────────────────────────────────────────────────────────
# Law 1 — Completeness:  grow(S, exhausted) == universe
# ──────────────────────────────────────────────────────────────────────────────

class TestCompleteness:
    """
    Exhausting the candidate pool must always recover the full reference basis,
    regardless of strategy, starting criterion, or min_degree.

    This is the discrete analogue of the fundamental theorem of calculus:
    integrating a decomposition (filtration) recovers the original set.
    """

    def test_all_strategies_2param(self, cs2):
        n = cs2.make_cs(2).getNumBasisFunctions()
        for s in _strategies(cs2, 2):
            assert cs2.make_adaptive_cs(s).getNumBasisFunctions() == n, (
                f"{type(s).__name__} failed completeness")

    def test_all_startings(self, cs2):
        """Any warm start must reach the same full basis after pool exhaustion."""
        n = cs2.make_cs(2).getNumBasisFunctions()
        s = LevelByLevelStrategy(2)
        for sc in _all_startings(cs2, 2):
            assert cs2.make_adaptive_cs(s, starting=sc).getNumBasisFunctions() == n, (
                f"{type(sc).__name__} starting failed completeness")

    def test_all_min_degrees(self, cs2):
        """Completeness holds for all min_degree values across all strategies."""
        n = cs2.make_cs(3).getNumBasisFunctions()
        v = [c.variance() for c in cs2.coordinates.values()]
        for k in range(4):
            for s in [LevelByLevelStrategy(3, min_degree=k),
                      SensitivityDrivenStrategy(3, v, min_degree=k, batch_size=1),
                      DownwardClosedStrategy(3, min_degree=k, batch_size=1)]:
                cs_a = cs2.make_adaptive_cs(s)
                assert cs_a.getNumBasisFunctions() == n, (
                    f"{type(s).__name__}(min_degree={k}) failed completeness")

    def test_3param_coordinate_system(self, cs3):
        n = cs3.make_cs(2).getNumBasisFunctions()
        for s in _strategies(cs3, 2):
            assert cs3.make_adaptive_cs(s).getNumBasisFunctions() == n, (
                f"{type(s).__name__} 3-param failed completeness")


# ──────────────────────────────────────────────────────────────────────────────
# Law 2 — Zero:  decay(S.reverse(), exhausted) == empty
# ──────────────────────────────────────────────────────────────────────────────

class TestZero:
    """
    Running a reversed strategy from a fully-enriched starting point must
    empty the active set entirely.

    LevelStarting(max_degree) is the ideal decay input: active IS the full
    reference basis and the pool starts empty.  The 'while active' guard in
    the adaptive loop drives decay to completion.
    """

    def test_all_strategies_full_decay(self, cs2):
        for s in _strategies(cs2, 2):
            cs_a = cs2.make_adaptive_cs(s.reverse(), starting=LevelStarting(2))
            assert cs_a.getNumBasisFunctions() == 0, (
                f"{type(s).__name__}.reverse() did not empty the basis")

    def test_partial_decay_reduces_basis(self, cs2):
        """One decay step must strictly shrink the active set."""
        n_full = cs2.make_cs(2).getNumBasisFunctions()
        for s in _strategies(cs2, 2):
            cs_a = cs2.make_adaptive_cs(
                s.reverse(),
                starting=LevelStarting(2),
                stopping=MaxIterationsStopping(1),
            )
            assert cs_a.getNumBasisFunctions() < n_full, (
                f"{type(s).__name__}.reverse() + MaxIter(1) did not reduce basis")


# ──────────────────────────────────────────────────────────────────────────────
# Law 3 — Involution:  S.reverse().reverse() is S
# ──────────────────────────────────────────────────────────────────────────────

class TestInvolution:
    """
    The reverse operation is an involution: applying it twice returns the
    original object.  This guarantees grow and decay are exact duals with
    no information loss across double-reversal.
    """

    def test_double_reverse_is_identity(self):
        v = [1.0, 0.5]
        for s in [LevelByLevelStrategy(3),
                  SensitivityDrivenStrategy(3, v),
                  DownwardClosedStrategy(3)]:
            assert s.reverse().reverse() is s

    def test_direction_property_toggles(self):
        """direction starts as 'grow'; each .reverse() flips it."""
        v = [1.0, 0.5]
        for s in [LevelByLevelStrategy(3),
                  SensitivityDrivenStrategy(3, v),
                  DownwardClosedStrategy(3)]:
            assert s.direction == 'grow'
            assert isinstance(s.reverse(), _ReversedStrategy)
            assert s.reverse().direction == 'decay'
            assert s.reverse().reverse().direction == 'grow'


# ──────────────────────────────────────────────────────────────────────────────
# Structural properties of DownwardClosedStrategy
# ──────────────────────────────────────────────────────────────────────────────

class TestDownwardClosed:
    """
    DownwardClosedStrategy enforces a downward-closed (monotone) index set
    at every step.  This is the Smolyak admissibility condition: a multi-index
    alpha is eligible only if alpha - e_i is already active for all i with
    alpha_i > 0.
    """

    @staticmethod
    def _dc_key(d):
        return tuple(sorted((k, v) for k, v in d.items() if v > 0))

    def test_active_set_always_downward_closed(self, cs2):
        cs_a = cs2.make_adaptive_cs(DownwardClosedStrategy(3, batch_size=1))
        active_degs = {self._dc_key(degs) for degs in cs_a.basis.values()}
        for degs in cs_a.basis.values():
            for axis, d in degs.items():
                if d > 0:
                    reduced = Counter({k: v for k, v in degs.items() if v > 0})
                    reduced[axis] -= 1
                    if reduced[axis] == 0:
                        del reduced[axis]
                    assert self._dc_key(reduced) in active_degs, (
                        f"Not downward-closed: {dict(degs)} active "
                        f"but predecessor {dict(reduced)} is not.")

    def test_admissible_rejects_skipped_level(self, cs2):
        """A level-2 mode is not admissible when only the mean is active."""
        cs_ref    = cs2.make_cs(3)
        active    = {0: cs_ref.basis[0]}
        level2_id = next(mid for mid, d in cs_ref.basis.items() if _total(d) == 2)
        assert not DownwardClosedStrategy._admissible(cs_ref.basis[level2_id], active)


# ──────────────────────────────────────────────────────────────────────────────
# DC guard: DownwardClosedStrategy rejects non-downward-closed seeds
# ──────────────────────────────────────────────────────────────────────────────

class TestDCGuard:

    def test_non_dc_seed_raises_value_error(self, cs2):
        """A level-2 seed (missing its level-1 predecessor) must raise ValueError."""
        cs_ref2   = cs2.make_cs(2)
        level2_id = next(iter(cs_ref2.find_modes(total_degree=2).keys()))
        with pytest.raises(ValueError, match="downward-closed"):
            cs2.make_adaptive_cs(
                DownwardClosedStrategy(2, batch_size=1),
                starting=FixedModeSetStarting({level2_id}),
            )

    def test_valid_seeds_accepted(self, cs2):
        """MeanOnly, LevelStarting(k), SensitivityStarting, and a DC FixedModeSet
        must all be accepted without error."""
        v  = [c.variance() for c in cs2.coordinates.values()]
        dc = set(cs2.make_cs(2).find_modes(total_degree=1).keys())
        for sc in [MeanOnlyStarting(),
                   LevelStarting(0), LevelStarting(1), LevelStarting(2),
                   SensitivityStarting(v),
                   FixedModeSetStarting(dc)]:
            cs2.make_adaptive_cs(DownwardClosedStrategy(2, batch_size=1), starting=sc)


# ──────────────────────────────────────────────────────────────────────────────
# API contract and 36-combination smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestContract:
    """
    Structural invariants every make_adaptive_cs result must satisfy:
      - basis_construction == ADAPTIVE_DEGREE
      - basis IDs are contiguous from 0
      - the mean (degree-0) mode is always present
      - find_modes() is consistent with getNumBasisFunctions()
    """

    def test_structural_invariants(self, cs2):
        for s in _strategies(cs2, 2):
            cs_a = cs2.make_adaptive_cs(s)
            n = cs_a.getNumBasisFunctions()
            assert cs_a.basis_construction == BasisFunctionType.ADAPTIVE_DEGREE
            assert set(cs_a.basis.keys()) == set(range(n))
            assert any(sum(d.values()) == 0 for d in cs_a.basis.values())
            assert len(cs_a.find_modes()) == n

    def test_all_36_combinations_no_exception(self, cs2):
        """Every (starting × strategy × stopping) triple must complete without error."""
        v  = [c.variance() for c in cs2.coordinates.values()]
        dc = set(cs2.make_cs(2).find_modes(total_degree=1).keys())
        startings  = [MeanOnlyStarting(), LevelStarting(1),
                      SensitivityStarting(v), FixedModeSetStarting(dc)]
        strategies = [LevelByLevelStrategy(2),
                      SensitivityDrivenStrategy(2, v),
                      DownwardClosedStrategy(2, batch_size=1)]
        for sc in startings:
            for tc in strategies:
                # Recreate stateful stopping criteria each iteration
                for oc in [CandidatePoolExhaustedStopping(),
                           MaxIterationsStopping(2),
                           RelativeGrowthStopping(tol=0.01)]:
                    label = (f"{type(sc).__name__}+"
                             f"{type(tc).__name__}+"
                             f"{type(oc).__name__}")
                    try:
                        cs2.make_adaptive_cs(tc, stopping=oc, starting=sc)
                    except Exception as e:
                        pytest.fail(f"[{label}] raised {type(e).__name__}: {e}")
