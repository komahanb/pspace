#=====================================================================#
# Tests for CoordinateSystem.find_modes() and param_to_mode_map()
#
# find_modes(param_degrees, total_degree, exact) is a composable query
# over the multi-index basis set.  param_to_mode_map() is a thin
# wrapper that identifies the single "pure linear" mode for each
# parameter.
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

import pytest
from collections import Counter

from pspace.core import (CoordinateFactory,
                         CoordinateSystem,
                         BasisFunctionType)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cs2_total():
    """2-parameter TOTAL_DEGREE coordinate system (k1 Uniform, k2 Uniform).

    Local parameter indices (as assigned by make_cs):
      0 -> k1,  1 -> k2

    Basis (degree 3, 10 modes):
      mode 0: (0,0)  mode 1: (0,1)  mode 2: (0,2)  mode 3: (0,3)
      mode 4: (1,0)  mode 5: (1,1)  mode 6: (1,2)
      mode 7: (2,0)  mode 8: (2,1)
      mode 9: (3,0)
    """
    cf = CoordinateFactory()
    k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                     dict(a=0.9, b=1.1), max_monomial_dof=3)
    k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                     dict(a=0.4, b=0.6), max_monomial_dof=3)
    cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
    cs.addCoordinateAxis(k1)
    cs.addCoordinateAxis(k2)
    cs.initialize()
    return cs.make_cs(3)


@pytest.fixture(scope="module")
def cs3_total():
    """3-parameter TOTAL_DEGREE coordinate system (degree 2).

    Local indices: 0->p0 (Normal), 1->p1 (Uniform), 2->p2 (Exponential).
    """
    cf = CoordinateFactory()
    p0 = cf.createNormalCoordinate(cf.newCoordinateID(), 'p0',
                                    dict(mu=1.0, sigma=0.2), max_monomial_dof=2)
    p1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'p1',
                                     dict(a=0.8, b=1.2), max_monomial_dof=2)
    p2 = cf.createExponentialCoordinate(cf.newCoordinateID(), 'p2',
                                         dict(mu=0.5, beta=1.0), max_monomial_dof=2)
    cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
    cs.addCoordinateAxis(p0)
    cs.addCoordinateAxis(p1)
    cs.addCoordinateAxis(p2)
    cs.initialize()
    return cs.make_cs(2)


@pytest.fixture(scope="module")
def cs1_total():
    """Single-parameter TOTAL_DEGREE (degree 3)."""
    cf = CoordinateFactory()
    p = cf.createUniformCoordinate(cf.newCoordinateID(), 'p',
                                    dict(a=0.0, b=1.0), max_monomial_dof=3)
    cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
    cs.addCoordinateAxis(p)
    cs.initialize()
    return cs.make_cs(3)


@pytest.fixture(scope="module")
def cs2_tensor():
    """2-parameter TENSOR_DEGREE coordinate system (degree 2 each)."""
    cf = CoordinateFactory()
    k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                     dict(a=0.9, b=1.1), max_monomial_dof=2)
    k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                     dict(a=0.4, b=0.6), max_monomial_dof=2)
    cs = CoordinateSystem(BasisFunctionType.TENSOR_DEGREE)
    cs.addCoordinateAxis(k1)
    cs.addCoordinateAxis(k2)
    cs.initialize()
    return cs.make_cs(2)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _degree_of(degs, param):
    """Return the degree of parameter `param` in a Counter `degs`."""
    return degs.get(param, 0)


def _total(degs):
    return sum(degs.values())


# ──────────────────────────────────────────────────────────────────────────────
# 1.  find_modes() with no arguments — return ALL modes
# ──────────────────────────────────────────────────────────────────────────────

class TestFindModesNoConstraint:

    def test_returns_all_modes_2param_deg3(self, cs2_total):
        result = cs2_total.find_modes()
        expected_size = cs2_total.getNumBasisFunctions()
        assert len(result) == expected_size, (
            f"find_modes() should return all {expected_size} modes, got {len(result)}")

    def test_mode_ids_cover_full_basis(self, cs2_total):
        result = cs2_total.find_modes()
        assert set(result.keys()) == set(cs2_total.basis.keys())

    def test_returns_all_modes_3param_deg2(self, cs3_total):
        result = cs3_total.find_modes()
        assert len(result) == cs3_total.getNumBasisFunctions()

    def test_returns_all_modes_tensor(self, cs2_tensor):
        result = cs2_tensor.find_modes()
        assert len(result) == cs2_tensor.getNumBasisFunctions()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  find_modes(total_degree=p) — filter by total polynomial degree
# ──────────────────────────────────────────────────────────────────────────────

class TestFindModesTotalDegree:

    def test_total_degree_0_is_mean_only(self, cs2_total):
        result = cs2_total.find_modes(total_degree=0)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _total(degs) == 0

    def test_total_degree_1_gives_linear_modes(self, cs2_total):
        result = cs2_total.find_modes(total_degree=1)
        # 2 params -> 2 linear modes for TOTAL_DEGREE
        assert len(result) == 2
        for mid, degs in result.items():
            assert _total(degs) == 1

    def test_total_degree_2_gives_3_modes(self, cs2_total):
        result = cs2_total.find_modes(total_degree=2)
        # (2,0), (1,1), (0,2) -> 3 modes
        assert len(result) == 3
        for mid, degs in result.items():
            assert _total(degs) == 2

    def test_total_degree_3_gives_4_modes(self, cs2_total):
        result = cs2_total.find_modes(total_degree=3)
        # (3,0),(2,1),(1,2),(0,3) -> 4 modes
        assert len(result) == 4
        for mid, degs in result.items():
            assert _total(degs) == 3

    def test_all_levels_partition_basis(self, cs2_total):
        """Modes at levels 0..3 should be disjoint and cover the full basis."""
        all_ids = set()
        for p in range(4):
            level_ids = set(cs2_total.find_modes(total_degree=p).keys())
            assert level_ids.isdisjoint(all_ids), f"Level {p} overlaps previous levels"
            all_ids |= level_ids
        assert all_ids == set(cs2_total.basis.keys())

    def test_total_degree_1_in_3param(self, cs3_total):
        result = cs3_total.find_modes(total_degree=1)
        assert len(result) == 3   # one linear mode per parameter
        for mid, degs in result.items():
            assert _total(degs) == 1

    def test_total_degree_single_param(self, cs1_total):
        for p in range(4):
            result = cs1_total.find_modes(total_degree=p)
            assert len(result) == 1, f"1 param should have exactly 1 mode per level, got {len(result)} at level {p}"
            assert _total(next(iter(result.values()))) == p

    def test_total_degree_tensor_basis(self, cs2_tensor):
        # TENSOR_DEGREE p=2: all 9 modes (0..2) x (0..2)
        # Level 0: (0,0), Level 1: (0,1),(1,0), Level 2: (0,2),(1,1),(2,0)
        result = cs2_tensor.find_modes(total_degree=2)
        assert len(result) == 3
        for mid, degs in result.items():
            assert _total(degs) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 3.  find_modes({k: d}, exact=True) — pure modes in a single parameter
# ──────────────────────────────────────────────────────────────────────────────

class TestFindModesPure:

    def test_pure_linear_k1(self, cs2_total):
        result = cs2_total.find_modes({0: 1}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 0) == 1
        assert _degree_of(degs, 1) == 0

    def test_pure_linear_k2(self, cs2_total):
        result = cs2_total.find_modes({1: 1}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 1) == 1
        assert _degree_of(degs, 0) == 0

    def test_pure_quadratic_k1(self, cs2_total):
        result = cs2_total.find_modes({0: 2}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 0) == 2
        assert _degree_of(degs, 1) == 0

    def test_pure_quadratic_k2(self, cs2_total):
        result = cs2_total.find_modes({1: 2}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 1) == 2
        assert _degree_of(degs, 0) == 0

    def test_mean_mode_via_empty_dict_exact(self, cs2_total):
        """find_modes({}, exact=True) must return only the mean mode."""
        result = cs2_total.find_modes({}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _total(degs) == 0

    def test_cross_term_exact(self, cs2_total):
        result = cs2_total.find_modes({0: 1, 1: 1}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 0) == 1
        assert _degree_of(degs, 1) == 1

    def test_degree_exceeds_basis_returns_empty(self, cs2_total):
        result = cs2_total.find_modes({0: 5}, exact=True)
        assert len(result) == 0

    def test_cross_term_exact_3param(self, cs3_total):
        # (1,1,0) — linear in first two params, constant in p2
        result = cs3_total.find_modes({0: 1, 1: 1}, exact=True)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 0) == 1
        assert _degree_of(degs, 1) == 1
        assert _degree_of(degs, 2) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 4.  find_modes({k: d}, exact=False) — modes containing at least this degree
# ──────────────────────────────────────────────────────────────────────────────

class TestFindModesImpure:

    def test_linear_in_k1_any_k2(self, cs2_total):
        result = cs2_total.find_modes({0: 1}, exact=False)
        # modes (1,0), (1,1), (1,2) -> 3 modes in a degree-3 basis
        assert len(result) == 3
        for mid, degs in result.items():
            assert _degree_of(degs, 0) == 1

    def test_linear_in_k2_any_k1(self, cs2_total):
        result = cs2_total.find_modes({1: 1}, exact=False)
        # modes (0,1),(1,1),(2,1) -> 3 modes
        assert len(result) == 3
        for mid, degs in result.items():
            assert _degree_of(degs, 1) == 1

    def test_quadratic_in_k1_any_k2(self, cs2_total):
        result = cs2_total.find_modes({0: 2}, exact=False)
        # modes (2,0),(2,1) -> 2 modes
        assert len(result) == 2
        for mid, degs in result.items():
            assert _degree_of(degs, 0) == 2

    def test_inexact_superset_of_exact(self, cs2_total):
        """Inexact result must always contain the exact result."""
        exact   = set(cs2_total.find_modes({0: 2}, exact=True).keys())
        inexact = set(cs2_total.find_modes({0: 2}, exact=False).keys())
        assert exact.issubset(inexact)

    def test_inexact_3param_linear_first(self, cs3_total):
        result = cs3_total.find_modes({0: 1}, exact=False)
        for mid, degs in result.items():
            assert _degree_of(degs, 0) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Composing total_degree + param_degrees
# ──────────────────────────────────────────────────────────────────────────────

class TestFindModesComposed:

    def test_linear_in_k1_at_total_degree_2(self, cs2_total):
        # The only level-2 mode with k1-degree=1 is (1,1)
        result = cs2_total.find_modes({0: 1}, total_degree=2, exact=False)
        assert len(result) == 1
        mid, degs = next(iter(result.items()))
        assert _degree_of(degs, 0) == 1
        assert _total(degs) == 2

    def test_pure_linear_at_level_1_exact(self, cs2_total):
        # Both pure-linear modes are at total_degree=1
        result = cs2_total.find_modes({0: 1}, total_degree=1, exact=True)
        assert len(result) == 1
        result2 = cs2_total.find_modes({1: 1}, total_degree=1, exact=True)
        assert len(result2) == 1

    def test_mean_at_total_degree_0(self, cs2_total):
        result = cs2_total.find_modes({}, exact=True, total_degree=0)
        assert len(result) == 1

    def test_composed_returns_subset_of_total_degree(self, cs2_total):
        for p in range(4):
            full  = set(cs2_total.find_modes(total_degree=p).keys())
            filt  = set(cs2_total.find_modes({0: 1}, total_degree=p, exact=False).keys())
            assert filt.issubset(full)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  param_to_mode_map()  — thin wrapper over find_modes
# ──────────────────────────────────────────────────────────────────────────────

class TestParamToModeMap:

    def test_2param_deg1_correct_mapping(self):
        """For degree-1 total basis with 2 params, each param maps to a unique mode."""
        cf = CoordinateFactory()
        k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                         dict(a=0.9, b=1.1), max_monomial_dof=1)
        k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                         dict(a=0.4, b=0.6), max_monomial_dof=1)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k1)
        cs.addCoordinateAxis(k2)
        cs.initialize()
        cs1 = cs.make_cs(1)

        mapping = cs1.param_to_mode_map()

        assert set(mapping.keys()) == {0, 1}, "Should have one entry per parameter"
        assert mapping[0] != mapping[1], "Each parameter must map to a distinct mode"
        # Verify both mapped modes are linear (total degree == 1)
        for k, mid in mapping.items():
            assert cs1.basis[mid].get(k, 0) == 1, f"Param {k} should map to a mode linear in param {k}"
            assert sum(cs1.basis[mid].values()) == 1, f"Mapped mode should be purely linear"

    def test_2param_deg3_each_param_has_unique_mode(self, cs2_total):
        mapping = cs2_total.param_to_mode_map()
        assert len(mapping) == 2
        # Values are distinct mode IDs
        assert len(set(mapping.values())) == 2
        # Each mode is purely linear in the named param
        for k, mid in mapping.items():
            assert cs2_total.basis[mid].get(k, 0) == 1
            assert sum(cs2_total.basis[mid].values()) == 1

    def test_3param_deg2_three_entries(self, cs3_total):
        mapping = cs3_total.param_to_mode_map()
        assert len(mapping) == 3
        assert len(set(mapping.values())) == 3
        for k, mid in mapping.items():
            assert cs3_total.basis[mid].get(k, 0) == 1
            assert sum(cs3_total.basis[mid].values()) == 1

    def test_degree_0_returns_empty(self):
        """At degree 0 there are no linear modes, so map is empty."""
        cf = CoordinateFactory()
        k1 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k1',
                                         dict(a=0.9, b=1.1), max_monomial_dof=1)
        k2 = cf.createUniformCoordinate(cf.newCoordinateID(), 'k2',
                                         dict(a=0.4, b=0.6), max_monomial_dof=1)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k1)
        cs.addCoordinateAxis(k2)
        cs.initialize()
        cs0 = cs.make_cs(0)
        assert cs0.param_to_mode_map() == {}

    def test_single_param_deg3_maps_param_0(self, cs1_total):
        mapping = cs1_total.param_to_mode_map()
        assert set(mapping.keys()) == {0}
        mid = mapping[0]
        assert cs1_total.basis[mid].get(0, 0) == 1
        assert sum(cs1_total.basis[mid].values()) == 1

    def test_param_to_mode_map_consistent_with_find_modes(self, cs2_total):
        """param_to_mode_map must return modes identical to find_modes({k:1}, exact=True)."""
        mapping = cs2_total.param_to_mode_map()
        for k, mid in mapping.items():
            fm_result = cs2_total.find_modes({k: 1}, exact=True)
            assert len(fm_result) == 1
            assert mid in fm_result, (
                f"param_to_mode_map gave mode {mid} for param {k}, "
                f"but find_modes returned {set(fm_result.keys())}")

    def test_tensor_basis_param_to_mode_map(self, cs2_tensor):
        mapping = cs2_tensor.param_to_mode_map()
        assert len(mapping) == 2
        for k, mid in mapping.items():
            assert cs2_tensor.basis[mid].get(k, 0) == 1
            assert sum(cs2_tensor.basis[mid].values()) == 1


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Edge cases and robustness
# ──────────────────────────────────────────────────────────────────────────────

class TestFindModesEdgeCases:

    def test_impossible_total_degree_returns_empty(self, cs2_total):
        """Total degree beyond basis max returns empty dict."""
        result = cs2_total.find_modes(total_degree=99)
        assert result == {}

    def test_conflicting_constraints_returns_empty(self, cs2_total):
        """Requesting k1-degree=2 AND total_degree=1 is impossible."""
        result = cs2_total.find_modes({0: 2}, total_degree=1, exact=True)
        assert result == {}

    def test_uninitialized_raises(self):
        """find_modes on an uninitialized CoordinateSystem must raise RuntimeError."""
        cf = CoordinateFactory()
        k  = cf.createUniformCoordinate(cf.newCoordinateID(), 'k',
                                         dict(a=0.0, b=1.0), max_monomial_dof=2)
        cs = CoordinateSystem(BasisFunctionType.TOTAL_DEGREE)
        cs.addCoordinateAxis(k)
        # Do NOT call cs.initialize() or make_cs()
        with pytest.raises(RuntimeError):
            cs.find_modes()

    def test_find_modes_does_not_mutate_basis(self, cs2_total):
        """Repeated calls must always return the same result."""
        r1 = cs2_total.find_modes({0: 1}, exact=True)
        r2 = cs2_total.find_modes({0: 1}, exact=True)
        assert set(r1.keys()) == set(r2.keys())

    def test_total_degree_0_tensor(self, cs2_tensor):
        result = cs2_tensor.find_modes(total_degree=0)
        assert len(result) == 1
        assert _total(next(iter(result.values()))) == 0

    def test_find_modes_returns_counters(self, cs2_total):
        """Returned degree dicts should behave like Counters."""
        result = cs2_total.find_modes(total_degree=1)
        for mid, degs in result.items():
            # Must support .get()
            _ = degs.get(0, 0)
            # Total degree must match
            assert sum(degs.values()) == 1
