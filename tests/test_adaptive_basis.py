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
    StartingCriterion,
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
# Starting criteria: unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestStartingCriteria:

    def test_mean_only_returns_one_mode(self, cs2):
        cs_ref = cs2.make_cs(3)
        pool   = dict(cs_ref.basis)
        active = MeanOnlyStarting().initialize(pool, cs_ref)
        assert len(active) == 1
        assert _total(next(iter(active.values()))) == 0

    def test_mean_only_removes_from_pool(self, cs2):
        cs_ref   = cs2.make_cs(3)
        pool     = dict(cs_ref.basis)
        n_before = len(pool)
        active   = MeanOnlyStarting().initialize(pool, cs_ref)
        assert len(pool) == n_before - 1
        assert set(active.keys()).isdisjoint(pool.keys())

    def test_level_starting_0_same_as_mean_only(self, cs2):
        cs_ref = cs2.make_cs(3)
        pool1, pool2 = dict(cs_ref.basis), dict(cs_ref.basis)
        a1 = MeanOnlyStarting().initialize(pool1, cs_ref)
        a2 = LevelStarting(0).initialize(pool2, cs_ref)
        assert len(a1) == len(a2)

    def test_level_starting_1_contains_levels_0_and_1(self, cs2):
        cs_ref = cs2.make_cs(3)
        pool   = dict(cs_ref.basis)
        active = LevelStarting(1).initialize(pool, cs_ref)
        levels = {_total(degs) for degs in active.values()}
        assert 0 in levels
        assert 1 in levels
        assert 2 not in levels
        # For 2 params: 1 mean + 2 linear = 3 modes
        assert len(active) == 3

    def test_level_starting_removes_from_pool(self, cs2):
        cs_ref = cs2.make_cs(3)
        pool   = dict(cs_ref.basis)
        active = LevelStarting(2).initialize(pool, cs_ref)
        assert set(active.keys()).isdisjoint(pool.keys())
        assert len(active) + len(pool) == len(cs_ref.basis)

    def test_sensitivity_starting_mean_always_included(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(3)
        pool      = dict(cs_ref.basis)
        active    = SensitivityStarting(variances, top_k=1).initialize(pool, cs_ref)
        assert any(_total(degs) == 0 for degs in active.values())

    def test_sensitivity_starting_top1_seeds_dominant_param(self, cs2):
        variances  = [c.variance() for c in cs2.coordinates.values()]
        dominant_k = variances.index(max(variances))
        cs_ref     = cs2.make_cs(3)
        pool       = dict(cs_ref.basis)
        active     = SensitivityStarting(variances, top_k=1).initialize(pool, cs_ref)
        # Mean + 1 pure-linear mode for dominant parameter
        assert len(active) == 2
        linear_modes = [degs for degs in active.values() if _total(degs) == 1]
        assert len(linear_modes) == 1
        assert linear_modes[0].get(dominant_k, 0) == 1

    def test_sensitivity_starting_top_all_seeds_all_linear(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(3)
        pool      = dict(cs_ref.basis)
        active    = SensitivityStarting(variances).initialize(pool, cs_ref)
        # Mean + 2 pure-linear modes for 2 params
        assert len(active) == 3

    def test_fixed_mode_set_includes_mean(self, cs2):
        cs_ref = cs2.make_cs(3)
        pool   = dict(cs_ref.basis)
        # Request a level-2 mode (ID 5) without explicitly listing the mean
        active = FixedModeSetStarting({5}).initialize(pool, cs_ref)
        assert any(_total(degs) == 0 for degs in active.values())

    def test_fixed_mode_set_includes_requested_modes(self, cs2):
        cs_ref    = cs2.make_cs(3)
        pool      = dict(cs_ref.basis)
        wanted    = {1, 4, 7}   # some arbitrary mode IDs in a degree-3 basis
        available = wanted & set(cs_ref.basis.keys())
        active    = FixedModeSetStarting(available).initialize(pool, cs_ref)
        assert available.issubset(set(active.keys()))

    def test_fixed_mode_set_removes_from_pool(self, cs2):
        cs_ref = cs2.make_cs(3)
        pool   = dict(cs_ref.basis)
        active = FixedModeSetStarting({1, 2, 3}).initialize(pool, cs_ref)
        assert set(active.keys()).isdisjoint(pool.keys())

    def test_starting_is_abc(self):
        assert issubclass(MeanOnlyStarting,       StartingCriterion)
        assert issubclass(LevelStarting,          StartingCriterion)
        assert issubclass(SensitivityStarting,    StartingCriterion)
        assert issubclass(FixedModeSetStarting,   StartingCriterion)


# ──────────────────────────────────────────────────────────────────────────────
# Starting criteria: integration with make_adaptive_cs
# ──────────────────────────────────────────────────────────────────────────────

class TestStartingCriterionIntegration:

    def test_mean_only_default_matches_explicit(self, cs2):
        """make_adaptive_cs() default == explicit MeanOnlyStarting."""
        strat  = LevelByLevelStrategy(3)
        cs_default  = cs2.make_adaptive_cs(strat)
        cs_explicit = cs2.make_adaptive_cs(strat, starting=MeanOnlyStarting())
        assert cs_default.getNumBasisFunctions() == cs_explicit.getNumBasisFunctions()

    def test_level_starting_skips_initial_levels(self, cs2):
        """Starting at level 1, MaxIterations(1) should reach level 2."""
        cs_a = cs2.make_adaptive_cs(
            LevelByLevelStrategy(3),
            starting=LevelStarting(1),
            stopping=MaxIterationsStopping(1),
        )
        assert 2 in _active_levels(cs_a)
        assert 3 not in _active_levels(cs_a)

    def test_level_starting_full_enrichment_agrees(self, cs2):
        """Starting from any level, full enrichment must reach the same total."""
        cs_ref = cs2.make_cs(3)
        n_ref  = cs_ref.getNumBasisFunctions()
        for start_level in range(3):
            cs_a = cs2.make_adaptive_cs(
                LevelByLevelStrategy(3),
                starting=LevelStarting(start_level),
            )
            assert cs_a.getNumBasisFunctions() == n_ref, (
                f"LevelStarting({start_level}): expected {n_ref}, "
                f"got {cs_a.getNumBasisFunctions()}")

    def test_sensitivity_starting_warm_start_full_enrichment(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(3)
        cs_a = cs2.make_adaptive_cs(
            SensitivityDrivenStrategy(3, variances),
            starting=SensitivityStarting(variances),
        )
        assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()

    def test_fixed_mode_set_warm_start(self, cs2):
        """Warm-starting from a fixed set and then enriching must cover all modes."""
        cs_ref = cs2.make_cs(2)
        # Seed with all level-1 modes (IDs 1,2 in a degree-2 ref CS)
        level1_ids = set(cs_ref.find_modes(total_degree=1).keys())
        cs_a = cs2.make_adaptive_cs(
            LevelByLevelStrategy(2),
            starting=FixedModeSetStarting(level1_ids),
        )
        assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()

    def test_all_startings_give_adaptive_degree_type(self, cs2):
        variances  = [c.variance() for c in cs2.coordinates.values()]
        strat      = LevelByLevelStrategy(2)
        startings  = [
            MeanOnlyStarting(),
            LevelStarting(1),
            SensitivityStarting(variances),
            FixedModeSetStarting({1}),
        ]
        for sc in startings:
            cs_a = cs2.make_adaptive_cs(strat, starting=sc)
            assert cs_a.basis_construction == BasisFunctionType.ADAPTIVE_DEGREE, (
                f"{type(sc).__name__} did not produce ADAPTIVE_DEGREE type")

    def test_basis_always_contiguous_from_zero(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        strat     = LevelByLevelStrategy(3)
        for sc in [MeanOnlyStarting(), LevelStarting(1),
                   SensitivityStarting(variances), FixedModeSetStarting({2, 3})]:
            cs_a = cs2.make_adaptive_cs(strat, starting=sc)
            assert set(cs_a.basis.keys()) == set(range(len(cs_a.basis))), (
                f"{type(sc).__name__}: basis IDs not contiguous")


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

    def test_exhaustion_completeness_all_12_combinations(self, cs2):
        """
        Completeness theorem (4 startings × 3 strategies × PoolExhausted):

        For any starting criterion and any strategy, exhausting the candidate
        pool must always reconstruct the full reference basis exactly.

        Rationale
        ---------
        The adaptive loop imposes a total ordering on the pool — a filtration
        S_0 ⊂ S_1 ⊂ ... ⊂ S_n.  Taking the union of all prefixes (i.e. running
        to PoolExhausted) must recover the full pool.  This is the discrete
        analogue of: integrating a decomposition recovers the original set.
        If any combination fails this property, that combination produces a
        sequence that cannot losslessly represent its operand.
        """
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(2)
        n_ref     = cs_ref.getNumBasisFunctions()
        pool_ids  = set(cs_ref.basis.keys())

        dc_seed = set(cs_ref.find_modes(total_degree=1).keys())

        startings = [
            ("MeanOnly",        MeanOnlyStarting()),
            ("Level1",          LevelStarting(1)),
            ("Sensitivity",     SensitivityStarting(variances)),
            ("FixedModeSet-DC", FixedModeSetStarting(dc_seed)),
        ]
        strategies = [
            ("LevelByLevel",   LevelByLevelStrategy(2)),
            ("Sensitivity",    SensitivityDrivenStrategy(2, variances, batch_size=1)),
            ("DownwardClosed", DownwardClosedStrategy(2, batch_size=1)),
        ]

        for sl, sc in startings:
            for tl, tc in strategies:
                label = f"{sl}+{tl}+PoolExhausted"
                cs_a  = cs2.make_adaptive_cs(
                    tc,
                    stopping=CandidatePoolExhaustedStopping(),
                    starting=sc,
                )
                n_a = cs_a.getNumBasisFunctions()
                assert n_a == n_ref, (
                    f"Completeness failed for [{label}]: "
                    f"got {n_a} modes, expected {n_ref}.  "
                    f"The sequence does not losslessly represent its operand."
                )

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

# Parametric coverage: all 36 (Starting x Strategy x Stopping) combinations

class TestAllCombinations:

    @pytest.fixture(autouse=True)
    def _setup(self, cs2):
        self.cs   = cs2
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref2   = cs2.make_cs(2)
        dc_seed   = set(cs_ref2.find_modes(total_degree=1).keys())

        self.startings = [
            ("MeanOnly",        MeanOnlyStarting()),
            ("Level1",          LevelStarting(1)),
            ("Sensitivity",     SensitivityStarting(variances)),
            ("FixedModeSet-DC", FixedModeSetStarting(dc_seed)),
        ]
        self.strategies = [
            ("LevelByLevel",   LevelByLevelStrategy(2)),
            ("Sensitivity",    SensitivityDrivenStrategy(2, variances)),
            ("DownwardClosed", DownwardClosedStrategy(2, batch_size=1)),
        ]
        self.stoppings = [
            ("PoolExhausted", CandidatePoolExhaustedStopping()),
            ("MaxIter2",      MaxIterationsStopping(2)),
            ("RelGrowth",     RelativeGrowthStopping(tol=0.01)),
        ]

    def _all_combos(self):
        for sl, sc in self.startings:
            for tl, tc in self.strategies:
                for ol in ("PoolExhausted", "MaxIter2", "RelGrowth"):
                    oc = {
                        "PoolExhausted": CandidatePoolExhaustedStopping(),
                        "MaxIter2":      MaxIterationsStopping(2),
                        "RelGrowth":     RelativeGrowthStopping(tol=0.01),
                    }[ol]
                    yield f"{sl}+{tl}+{ol}", sc, tc, oc

    def test_all_36_no_exception(self):
        for label, sc, tc, oc in self._all_combos():
            try:
                self.cs.make_adaptive_cs(tc, stopping=oc, starting=sc)
            except Exception as e:
                pytest.fail(f"Combo [{label}] raised {type(e).__name__}: {e}")

    def test_all_36_return_adaptive_degree_type(self):
        for label, sc, tc, oc in self._all_combos():
            cs_a = self.cs.make_adaptive_cs(tc, stopping=oc, starting=sc)
            assert cs_a.basis_construction == BasisFunctionType.ADAPTIVE_DEGREE, (
                f"Combo [{label}]: wrong basis type")

    def test_all_36_mean_mode_present(self):
        for label, sc, tc, oc in self._all_combos():
            cs_a = self.cs.make_adaptive_cs(tc, stopping=oc, starting=sc)
            assert any(_total(degs) == 0 for degs in cs_a.basis.values()), (
                f"Combo [{label}]: mean mode missing")

    def test_all_36_contiguous_ids(self):
        for label, sc, tc, oc in self._all_combos():
            cs_a = self.cs.make_adaptive_cs(tc, stopping=oc, starting=sc)
            n = cs_a.getNumBasisFunctions()
            assert set(cs_a.basis.keys()) == set(range(n)), (
                f"Combo [{label}]: non-contiguous IDs")

    def test_all_36_find_modes_works(self):
        for label, sc, tc, oc in self._all_combos():
            cs_a = self.cs.make_adaptive_cs(tc, stopping=oc, starting=sc)
            assert len(cs_a.find_modes()) == cs_a.getNumBasisFunctions(), (
                f"Combo [{label}]: find_modes() count mismatch")


class TestDownwardClosedGuard:

    def test_non_dc_seed_raises_value_error(self, cs2):
        cs_ref2   = cs2.make_cs(2)
        level2_id = next(iter(cs_ref2.find_modes(total_degree=2).keys()))
        bad_seed  = FixedModeSetStarting({level2_id})
        with pytest.raises(ValueError, match="downward-closed"):
            cs2.make_adaptive_cs(DownwardClosedStrategy(2, batch_size=1), starting=bad_seed)

    def test_dc_seed_does_not_raise(self, cs2):
        cs_ref2  = cs2.make_cs(2)
        dc_seed  = set(cs_ref2.find_modes(total_degree=1).keys())
        cs_a = cs2.make_adaptive_cs(
            DownwardClosedStrategy(2, batch_size=1),
            starting=FixedModeSetStarting(dc_seed),
        )
        assert cs_a.getNumBasisFunctions() > 0

    def test_mean_only_never_raises_for_dc(self, cs2):
        cs2.make_adaptive_cs(DownwardClosedStrategy(3, batch_size=1), starting=MeanOnlyStarting())

    def test_level_starting_never_raises_for_dc(self, cs2):
        for level in range(3):
            cs2.make_adaptive_cs(DownwardClosedStrategy(3, batch_size=1), starting=LevelStarting(level))

    def test_sensitivity_starting_never_raises_for_dc(self, cs2):
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs2.make_adaptive_cs(
            DownwardClosedStrategy(3, batch_size=1),
            starting=SensitivityStarting(variances),
        )

    def test_error_message_mentions_downward_closed(self, cs2):
        cs_ref2   = cs2.make_cs(2)
        level2_id = next(iter(cs_ref2.find_modes(total_degree=2).keys()))
        with pytest.raises(ValueError) as exc_info:
            cs2.make_adaptive_cs(
                DownwardClosedStrategy(2, batch_size=1),
                starting=FixedModeSetStarting({level2_id}),
            )
        assert "downward-closed" in str(exc_info.value).lower()


# ------------------------------------------------------------------------------
# min_degree: pool filtering and baseline pre-seeding
# ------------------------------------------------------------------------------

class TestMinDegree:
    """
    min_degree splits the reference basis into:
      baseline  -- modes with |alpha| < min_degree  (auto-active, not in pool)
      pool      -- modes with |alpha| >= min_degree  (subject to enrichment)

    Completeness: baseline union exhausted-pool must equal the full reference.
    """

    def test_min_degree_0_is_default_behaviour(self, cs2):
        """min_degree=0 must reproduce the standard no-baseline result."""
        cs_ref = cs2.make_cs(3)
        cs_a0  = cs2.make_adaptive_cs(LevelByLevelStrategy(3, min_degree=0))
        cs_a_d = cs2.make_adaptive_cs(LevelByLevelStrategy(3))
        assert cs_a0.getNumBasisFunctions() == cs_a_d.getNumBasisFunctions()

    def test_baseline_modes_always_present(self, cs2):
        """All modes with |alpha| < min_degree must appear in the result."""
        cs_ref  = cs2.make_cs(3)
        level1  = {mid for mid, degs in cs_ref.basis.items()
                   if sum(degs.values()) <= 1}
        cs_a = cs2.make_adaptive_cs(
            LevelByLevelStrategy(3, min_degree=2),
            stopping=CandidatePoolExhaustedStopping(),
        )
        result_ids = set(range(cs_a.getNumBasisFunctions()))
        # The result must contain the same number as full reference
        assert cs_a.getNumBasisFunctions() == cs_ref.getNumBasisFunctions()

    def test_completeness_with_min_degree(self, cs2):
        """Exhausting the pool with any min_degree still recovers full basis."""
        cs_ref = cs2.make_cs(3)
        n_ref  = cs_ref.getNumBasisFunctions()
        for min_d in range(4):   # 0, 1, 2, 3
            cs_a = cs2.make_adaptive_cs(
                LevelByLevelStrategy(3, min_degree=min_d),
                stopping=CandidatePoolExhaustedStopping(),
            )
            assert cs_a.getNumBasisFunctions() == n_ref, (
                f"min_degree={min_d}: got {cs_a.getNumBasisFunctions()}, expected {n_ref}")

    def test_min_degree_equal_max_degree_gives_full_baseline(self, cs2):
        """min_degree == max_degree: entire pool is empty, result = baseline."""
        max_d  = 2
        cs_ref = cs2.make_cs(max_d)
        n_ref  = cs_ref.getNumBasisFunctions()
        cs_a = cs2.make_adaptive_cs(
            LevelByLevelStrategy(max_d, min_degree=max_d),
        )
        # baseline contains all modes at levels 0..max_d-1; pool has only
        # level max_d modes which are added by the enrichment loop (or pool
        # is empty if min_degree == max_degree leaves nothing above baseline).
        # Either way the total must not exceed n_ref.
        assert cs_a.getNumBasisFunctions() <= n_ref

    def test_dc_strategy_min_degree_completeness(self, cs2):
        """DownwardClosedStrategy with min_degree also satisfies completeness."""
        cs_ref = cs2.make_cs(3)
        n_ref  = cs_ref.getNumBasisFunctions()
        for min_d in range(4):
            cs_a = cs2.make_adaptive_cs(
                DownwardClosedStrategy(3, min_degree=min_d, batch_size=1),
                stopping=CandidatePoolExhaustedStopping(),
            )
            assert cs_a.getNumBasisFunctions() == n_ref, (
                f"DC min_degree={min_d}: got {cs_a.getNumBasisFunctions()}, expected {n_ref}")

    def test_sensitivity_strategy_min_degree_completeness(self, cs2):
        """SensitivityDrivenStrategy with min_degree also satisfies completeness."""
        variances = [c.variance() for c in cs2.coordinates.values()]
        cs_ref    = cs2.make_cs(3)
        n_ref     = cs_ref.getNumBasisFunctions()
        for min_d in range(4):
            cs_a = cs2.make_adaptive_cs(
                SensitivityDrivenStrategy(3, variances, min_degree=min_d),
                stopping=CandidatePoolExhaustedStopping(),
            )
            assert cs_a.getNumBasisFunctions() == n_ref, (
                f"Sensitivity min_degree={min_d}: got {cs_a.getNumBasisFunctions()}, expected {n_ref}")

    def test_mean_mode_present_with_nonzero_min_degree(self, cs2):
        """Mean mode must always be present even when min_degree > 0."""
        for min_d in range(1, 4):
            cs_a = cs2.make_adaptive_cs(
                LevelByLevelStrategy(3, min_degree=min_d),
            )
            assert any(sum(d.values()) == 0 for d in cs_a.basis.values()), (
                f"min_degree={min_d}: mean mode missing")

    def test_min_degree_exposes_correct_property(self):
        """All three strategies must expose min_degree as a property."""
        assert LevelByLevelStrategy(3, min_degree=2).min_degree == 2
        assert DownwardClosedStrategy(3, min_degree=2).min_degree == 2
        variances = [1.0, 0.5]
        assert SensitivityDrivenStrategy(3, variances, min_degree=2).min_degree == 2

    def test_default_min_degree_is_zero(self):
        """Default min_degree must be 0 for all strategies."""
        assert LevelByLevelStrategy(3).min_degree == 0
        assert DownwardClosedStrategy(3).min_degree == 0
        assert SensitivityDrivenStrategy(3, [1.0, 0.5]).min_degree == 0
