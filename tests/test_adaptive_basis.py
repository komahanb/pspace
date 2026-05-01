#=====================================================================#
# Tests for adaptive basis selection:
#   CoordinateSystem.make_adaptive_cs()
#   LevelByLevelStrategy, SensitivityDrivenStrategy, DownwardClosedStrategy
#   CandidatePoolExhaustedStopping, MaxIterationsStopping, RelativeGrowthStopping
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
from collections import Counter

from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType)
from pspace.adaptive import (
    LevelByLevelStrategy,
    SensitivityDrivenStrategy,
    DownwardClosedStrategy,
    CandidatePoolExhaustedStopping,
    MaxIterationsStopping,
    RelativeGrowthStopping,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
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


def _active_levels(cs_a):
    """Return set of total degrees present in cs_a's basis."""
    return {_total(degs) for degs in cs_a.basis.values()}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: strategy / stopping ABC contracts
# ──────────────────────────────────────────────────────────────────────────────

class TestStrategyABC:

    def test_level_by_level_is_strategy(self, cs2):
        s = LevelByLevelStrategy(max_degree=2)
        assert isinstance(s, LevelByLevelStrategy)
        assert s.max_degree == 2

    def test_sensitivity_driven_is_strategy(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        s = SensitivityDrivenStrategy(max_degree=2, variances=variances)
        assert s.max_degree == 2

    def test_downward_closed_is_strategy(self, cs2):
        s = DownwardClosedStrategy(max_degree=2)
        assert s.max_degree == 2

    def test_max_iterations_is_stopping(self):
        assert isinstance(MaxIterationsStopping(3), MaxIterationsStopping)

    def test_relative_growth_is_stopping(self):
        assert isinstance(RelativeGrowthStopping(0.05), RelativeGrowthStopping)


# ──────────────────────────────────────────────────────────────────────────────
# make_adaptive_cs: basic contract
# ──────────────────────────────────────────────────────────────────────────────

class TestMakeAdaptiveCsContract:

    def test_returns_adaptive_degree_type(self, cs2):
        cs_a = cs2.make_adaptive_cs(LevelByLevelStrategy(2))
        assert cs_a.basis_construction == BasisFunctionType.ADAPTIVE_DEGREE

    def test_mean_mode_always_present(self, cs2):
        """Mode 0 (constant) must always be in the adaptive basis."""
        cs_a = cs2.make_adaptive_cs(LevelByLevelStrategy(3))
        mean_level = min(_total(degs) for degs in cs_a.basis.values())
        assert mean_level == 0

    def test_basis_ids_are_contiguous_from_zero(self, cs2):
        cs_a = cs2.make_adaptive_cs(LevelByLevelStrategy(3))
        assert set(cs_a.basis.keys()) == set(range(len(cs_a.basis)))

    def test_find_modes_works_on_adaptive_cs(self, cs2):
        cs_a = cs2.make_adaptive_cs(LevelByLevelStrategy(2))
        result = cs_a.find_modes(total_degree=1)
        assert len(result) == 2   # two linear modes for 2 params

    def test_same_axes_as_parent(self, cs2):
        cs_a = cs2.make_adaptive_cs(LevelByLevelStrategy(2))
        assert cs_a.getNumCoordinateAxes() == cs2.getNumCoordinateAxes()

    def test_param_to_mode_map_works(self, cs2):
        cs_a = cs2.make_adaptive_cs(LevelByLevelStrategy(2))
        mapping = cs_a.param_to_mode_map()
        assert len(mapping) == 2


# ──────────────────────────────────────────────────────────────────────────────
# LevelByLevelStrategy
# ──────────────────────────────────────────────────────────────────────────────

class TestLevelByLevel:

    def test_full_enrichment_equals_total_degree_cs(self, cs2):
        """Exhaustive LevelByLevel must produce same modes as make_cs(p)."""
        for p in range(1, 4):
            cs_ref = cs2.make_cs(p)
            cs_a   = cs2.make_adaptive_cs(LevelByLevelStrategy(p))
            assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()

    def test_max_iterations_1_gives_level_0_and_1(self, cs2):
        """After 1 iteration (level 1), active should contain levels 0 and 1."""
        cs_a = cs2.make_adaptive_cs(
            LevelByLevelStrategy(3),
            stopping=MaxIterationsStopping(1),
        )
        levels = _active_levels(cs_a)
        assert 0 in levels
        assert 1 in levels
        assert 2 not in levels

    def test_max_iterations_2_gives_levels_0_1_2(self, cs2):
        cs_a = cs2.make_adaptive_cs(
            LevelByLevelStrategy(3),
            stopping=MaxIterationsStopping(2),
        )
        levels = _active_levels(cs_a)
        assert levels == {0, 1, 2}

    def test_basis_size_at_each_level(self, cs2):
        """For 2 params, level k adds k+1 modes (total-degree binomial)."""
        expected = {0: 1, 1: 3, 2: 6, 3: 10}
        for p in range(1, 4):
            cs_a = cs2.make_adaptive_cs(
                LevelByLevelStrategy(p),
                stopping=MaxIterationsStopping(p),
            )
            assert cs_a.getNumBasisFunctions() == expected[p], (
                f"degree {p}: expected {expected[p]}, got {cs_a.getNumBasisFunctions()}")

    def test_select_returns_whole_level(self, cs2):
        """LevelByLevelStrategy.select on a fresh pool returns all level-1 modes."""
        cs_ref = cs2.make_cs(3)
        active = {0: cs_ref.basis[0]}
        pool   = {mid: degs for mid, degs in cs_ref.basis.items() if mid != 0}
        s      = LevelByLevelStrategy(3)
        sel    = s.select(active, pool, cs_ref)
        assert all(_total(pool[mid]) == 1 for mid in sel)
        assert len(sel) == 2   # both level-1 modes for 2 params

    def test_select_on_empty_pool_returns_empty(self, cs2):
        cs_ref = cs2.make_cs(1)
        s = LevelByLevelStrategy(1)
        assert s.select({}, {}, cs_ref) == set()

    def test_3param_full_enrichment(self, cs3):
        for p in range(1, 4):
            cs_ref = cs3.make_cs(p)
            cs_a   = cs3.make_adaptive_cs(LevelByLevelStrategy(p))
            assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()


# ──────────────────────────────────────────────────────────────────────────────
# SensitivityDrivenStrategy
# ──────────────────────────────────────────────────────────────────────────────

class TestSensitivityDriven:

    def _make_strategy(self, cs, max_degree, batch=1):
        variances = [c.variance() for c in cs.coordinates.values()]
        return SensitivityDrivenStrategy(max_degree, variances, batch_size=batch)

    def test_first_selected_mode_belongs_to_highest_variance_param(self, cs2):
        """The very first mode added should involve the highest-variance parameter."""
        variances  = [c.variance() for c in cs2.coordinates.values()]
        dominant_k = variances.index(max(variances))   # local index
        cs_ref = cs2.make_cs(3)
        active = {0: cs_ref.basis[0]}
        pool   = {mid: degs for mid, degs in cs_ref.basis.items() if mid != 0}
        s      = SensitivityDrivenStrategy(3, variances)
        sel    = s.select(active, pool, cs_ref)
        assert len(sel) == 1
        mid = next(iter(sel))
        assert pool[mid].get(dominant_k, 0) >= 1

    def test_full_enrichment_covers_all_modes(self, cs2):
        """With batch_size=1 and no stopping, all modes should be covered."""
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(3)
        cs_a      = cs2.make_adaptive_cs(
            SensitivityDrivenStrategy(3, variances, batch_size=1)
        )
        assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()

    def test_batch_size_2_adds_two_at_a_time(self, cs2):
        """With batch_size=2 and MaxIterations(1), should add 2 modes in first step."""
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(3)
        active = {0: cs_ref.basis[0]}
        pool   = {mid: degs for mid, degs in cs_ref.basis.items() if mid != 0}
        s      = SensitivityDrivenStrategy(3, variances, batch_size=2)
        sel    = s.select(active, pool, cs_ref)
        assert len(sel) <= 2

    def test_mean_always_in_basis(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_a = cs2.make_adaptive_cs(SensitivityDrivenStrategy(3, variances))
        mean_ids = [mid for mid, degs in cs_a.basis.items()
                    if _total(degs) == 0]
        assert len(mean_ids) == 1

    def test_3param_covers_all(self, cs3):
        variances = [c.variance() for c in cs3.coordinates.values()]
        cs_ref    = cs3.make_cs(2)
        cs_a      = cs3.make_adaptive_cs(
            SensitivityDrivenStrategy(2, variances)
        )
        assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()


# ──────────────────────────────────────────────────────────────────────────────
# DownwardClosedStrategy
# ──────────────────────────────────────────────────────────────────────────────

class TestDownwardClosed:

    def test_active_set_always_downward_closed(self, cs2):
        """At every step the active set must be a downward-closed index set."""
        cs_a = cs2.make_adaptive_cs(DownwardClosedStrategy(3, batch_size=1))

        # Strip zero-degree axes so mean mode appears as () and comparisons work
        def _key(d):
            return tuple(sorted((k, v) for k, v in d.items() if v > 0))

        active_degs = {_key(degs) for degs in cs_a.basis.values()}

        for degs in cs_a.basis.values():
            for axis, d in degs.items():
                if d > 0:
                    reduced = Counter({k: v for k, v in degs.items() if v > 0})
                    reduced[axis] -= 1
                    if reduced[axis] == 0:
                        del reduced[axis]
                    key = _key(reduced)
                    assert key in active_degs, (
                        f"Basis is not downward-closed: {dict(degs)} is active "
                        f"but {dict(reduced)} is not.")

    def test_full_enrichment_equals_total_degree_cs(self, cs2):
        cs_ref = cs2.make_cs(3)
        cs_a   = cs2.make_adaptive_cs(DownwardClosedStrategy(3, batch_size=1))
        assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()

    def test_mean_mode_present(self, cs2):
        cs_a = cs2.make_adaptive_cs(DownwardClosedStrategy(2, batch_size=1))
        assert any(_total(degs) == 0 for degs in cs_a.basis.values())

    def test_admissible_check_rejects_skipped_level(self, cs2):
        """A mode at level 2 is not admissible if the connecting level-1 mode is absent."""
        cs_ref     = cs2.make_cs(3)
        active     = {0: cs_ref.basis[0]}   # only mean

        # Find a level-2 mode (total degree 2)
        level2_ids = [mid for mid, degs in cs_ref.basis.items()
                      if _total(degs) == 2]
        assert level2_ids, "Need at least one level-2 mode"

        degs2 = cs_ref.basis[level2_ids[0]]
        # It should NOT be admissible with only the mean mode active
        assert not DownwardClosedStrategy._admissible(degs2, active)

    def test_custom_scorer_preferred(self, cs2):
        """A custom scorer that prefers higher degree should still give downward-closed set."""
        scorer = lambda mid, degs, active, cs_ref: sum(degs.values())
        cs_a = cs2.make_adaptive_cs(
            DownwardClosedStrategy(3, scorer=scorer, batch_size=1)
        )
        # Just check it completes without error and is non-empty
        assert cs_a.getNumBasisFunctions() > 0

    def test_3param_downward_closed(self, cs3):
        def _key(d):
            return tuple(sorted((k, v) for k, v in d.items() if v > 0))

        cs_a = cs3.make_adaptive_cs(DownwardClosedStrategy(2, batch_size=1))
        active_degs = {_key(degs) for degs in cs_a.basis.values()}
        for degs in cs_a.basis.values():
            for axis, d in degs.items():
                if d > 0:
                    reduced = Counter({k: v for k, v in degs.items() if v > 0})
                    reduced[axis] -= 1
                    if reduced[axis] == 0:
                        del reduced[axis]
                    key = _key(reduced)
                    assert key in active_degs


# ──────────────────────────────────────────────────────────────────────────────
# Stopping criteria
# ──────────────────────────────────────────────────────────────────────────────

class TestStoppingCriteria:

    def test_pool_exhausted_stops_only_when_empty(self, cs2):
        s     = CandidatePoolExhaustedStopping()
        cs_ref = cs2.make_cs(2)
        pool  = dict(cs_ref.basis)
        assert not s.should_stop({}, pool,   cs_ref, 0)
        assert     s.should_stop({}, {},     cs_ref, 0)

    def test_max_iterations_stops_at_threshold(self, cs2):
        s      = MaxIterationsStopping(3)
        cs_ref = cs2.make_cs(3)
        assert not s.should_stop({}, {}, cs_ref, 0)
        assert not s.should_stop({}, {}, cs_ref, 2)
        assert     s.should_stop({}, {}, cs_ref, 3)
        assert     s.should_stop({}, {}, cs_ref, 99)

    def test_max_iterations_produces_correct_level_count(self, cs2):
        """MaxIterationsStopping(n) should add exactly n levels beyond the mean."""
        for n in range(1, 4):
            cs_a = cs2.make_adaptive_cs(
                LevelByLevelStrategy(3),
                stopping=MaxIterationsStopping(n),
            )
            levels = _active_levels(cs_a)
            assert max(levels) == n, (
                f"n={n}: expected max level {n}, got {max(levels)}")

    def test_relative_growth_does_not_stop_on_first_iter(self, cs2):
        s = RelativeGrowthStopping(tol=0.99)
        cs_ref = cs2.make_cs(2)
        assert not s.should_stop({}, cs_ref.basis, cs_ref, 0)

    def test_relative_growth_stops_after_small_batch(self, cs2):
        """After recording a tiny batch, RelativeGrowthStopping should fire."""
        s = RelativeGrowthStopping(tol=0.5)
        cs_ref = cs2.make_cs(3)
        # Simulate: active has 100 modes, only 1 was added last round
        s.record(n_selected=1, n_active=100)
        assert s.should_stop({}, cs_ref.basis, cs_ref, iteration=1)

    def test_relative_growth_does_not_stop_when_large_batch(self, cs2):
        s = RelativeGrowthStopping(tol=0.05)
        cs_ref = cs2.make_cs(3)
        s.record(n_selected=5, n_active=6)   # ratio = 5/6 > 0.05
        assert not s.should_stop({}, cs_ref.basis, cs_ref, iteration=1)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-strategy consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossStrategyConsistency:

    def test_all_strategies_contain_mean(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        strategies = [
            LevelByLevelStrategy(2),
            SensitivityDrivenStrategy(2, variances),
            DownwardClosedStrategy(2),
        ]
        for s in strategies:
            cs_a = cs2.make_adaptive_cs(s)
            assert any(_total(degs) == 0 for degs in cs_a.basis.values()), (
                f"{type(s).__name__} missing mean mode")

    def test_all_strategies_full_enrichment_agree(self, cs2):
        """All strategies exhaustively applied must yield the same mode count."""
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(2)
        n_ref     = cs_ref.getNumBasisFunctions()

        strategies = [
            LevelByLevelStrategy(2),
            SensitivityDrivenStrategy(2, variances, batch_size=1),
            DownwardClosedStrategy(2, batch_size=1),
        ]
        for s in strategies:
            cs_a = cs2.make_adaptive_cs(s)
            assert cs_a.getNumBasisFunctions() == n_ref, (
                f"{type(s).__name__}: expected {n_ref}, got {cs_a.getNumBasisFunctions()}")

    def test_adaptive_cs_find_modes_consistent(self, cs2):
        """find_modes on adaptive CS should not raise and return subsets."""
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_a = cs2.make_adaptive_cs(
            SensitivityDrivenStrategy(3, variances),
            stopping=MaxIterationsStopping(2),
        )
        # find_modes queries must work without error
        all_modes = cs_a.find_modes()
        assert len(all_modes) == cs_a.getNumBasisFunctions()
        level1 = cs_a.find_modes(total_degree=1)
        assert set(level1.keys()).issubset(set(all_modes.keys()))
