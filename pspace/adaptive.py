#=====================================================================#
# Adaptive basis selection for CoordinateSystem.make_adaptive_cs().
#
# Separation of concerns — three orthogonal axes:
# ─────────────────────────────────────────────────────────────────────
#   StartingCriterion      — *where* to begin (initial active set)
#   AdaptiveBasisStrategy  — *what* modes to add at each step
#   StoppingCriterion      — *when* to stop adding modes
#
# The adaptive loop in make_adaptive_cs() orchestrates:
#
#   active  = starting.initialize(pool, cs_ref)
#   pool    = all candidate modes up to strategy.max_degree
#   while pool and not stopping(active, pool, cs_ref, iter):
#       selected = strategy.select(active, pool, cs_ref)
#       active  += selected;  pool -= selected
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

from __future__ import annotations

from abc        import ABC, abstractmethod
from collections import Counter
from typing     import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CoordinateSystem   # only for type hints; no circular import


# ──────────────────────────────────────────────────────────────────────────────
# StartingCriterion ABC and concrete implementations
# ──────────────────────────────────────────────────────────────────────────────

class StartingCriterion(ABC):
    """
    Pluggable policy that determines the **initial active set** before the
    enrichment loop begins.

    Subclasses must implement :meth:`initialize`, which consumes entries
    from *pool* and returns them as the starting active set.
    """

    @abstractmethod
    def initialize(self, pool: dict, cs_ref) -> dict:
        """
        Build the initial active set from the candidate pool.

        This method is called once by ``make_adaptive_cs`` before the
        enrichment loop.  It must:

        * Select a subset of *pool* entries for the initial active set.
        * **Remove** those entries from *pool* in-place.
        * Return the initial active set as a new ``dict[mode_id, Counter]``.

        Parameters
        ----------
        pool : dict[mode_id, Counter]
            All candidate modes (mutable; modify in-place).
        cs_ref : CoordinateSystem
            The full reference coordinate system at ``strategy.max_degree``.

        Returns
        -------
        dict[int, Counter]
            Initial active set.
        """


class MeanOnlyStarting(StartingCriterion):
    """
    Start with just the mean mode (total degree 0).

    This is the default — the minimal possible starting point, equivalent
    to "start from scratch."  It is the starting criterion implicitly used
    by the ISQS hierarchy.
    """

    def initialize(self, pool, cs_ref) -> dict:
        mean_id = next(mid for mid, degs in pool.items()
                       if sum(degs.values()) == 0)
        return {mean_id: pool.pop(mean_id)}


class LevelStarting(StartingCriterion):
    """
    Start with all modes up to and including a given total degree.

    Useful when a coarse solution (e.g. a deterministic run) is already
    available and you want to enrich *beyond* a known baseline level.

    Parameters
    ----------
    level : int
        All modes with ``sum(alpha) <= level`` are placed in the initial
        active set.  ``level=0`` is equivalent to :class:`MeanOnlyStarting`.
    """

    def __init__(self, level: int):
        self._level = level

    def initialize(self, pool, cs_ref) -> dict:
        selected = {mid: degs for mid, degs in pool.items()
                    if sum(degs.values()) <= self._level}
        for mid in selected:
            pool.pop(mid)
        return selected


class SensitivityStarting(StartingCriterion):
    """
    Start with the mean mode plus the pure-linear mode for each of the
    top-*k* parameters ranked by variance.

    This encodes the BSF prior: the directions of largest variance are the
    most likely to matter, so seed the active set with their linear modes
    before letting the strategy decide what comes next.

    Parameters
    ----------
    variances : list[float]
        Per-parameter variances in **local** index order (0-based).
    top_k : int, optional
        Number of parameters to seed.  Defaults to all parameters.
    """

    def __init__(self, variances: list, top_k: int | None = None):
        self._variances = list(variances)
        self._top_k     = top_k if top_k is not None else len(variances)

    def initialize(self, pool, cs_ref) -> dict:
        # Mean mode first
        mean_id = next(mid for mid, degs in pool.items()
                       if sum(degs.values()) == 0)
        active = {mean_id: pool.pop(mean_id)}

        # Rank params by descending variance and seed their pure-linear modes
        ranked = sorted(range(len(self._variances)),
                        key=lambda k: self._variances[k], reverse=True)
        for k in ranked[:self._top_k]:
            pure = cs_ref.find_modes({k: 1}, exact=True)
            for mid in pure:
                if mid in pool:
                    active[mid] = pool.pop(mid)

        return active


class FixedModeSetStarting(StartingCriterion):
    """
    Start with a user-supplied set of mode IDs.

    Gives full control over the initial active set — useful for warm-starts
    from a previous adaptive run or for restart scenarios.

    Parameters
    ----------
    mode_ids : iterable[int]
        Mode IDs (from the reference CS at ``strategy.max_degree``) to
        place in the initial active set.  The mean mode (ID 0) is always
        included even if not listed.
    """

    def __init__(self, mode_ids):
        self._mode_ids = set(mode_ids)

    def initialize(self, pool, cs_ref) -> dict:
        # Always include the mean mode
        mean_ids = {mid for mid, degs in pool.items()
                    if sum(degs.values()) == 0}
        wanted = self._mode_ids | mean_ids

        active = {}
        for mid in list(pool):
            if mid in wanted:
                active[mid] = pool.pop(mid)
        return active


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base classes
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveBasisStrategy(ABC):
    """
    Pluggable policy that selects a subset of candidate modes to add at
    each enrichment step.

    Subclasses must implement :meth:`select` and expose :attr:`max_degree`.

    Parameters are passed as ``dict[mode_id, Counter]`` throughout.
    """

    @property
    @abstractmethod
    def max_degree(self) -> int:
        """
        Maximum total polynomial degree of the candidate pool.
        The reference CS used for candidate generation is built at this degree.
        """

    @abstractmethod
    def select(
        self,
        active: dict,
        pool:   dict,
        cs_ref,   # CoordinateSystem
    ) -> set:
        """
        Choose a subset of *pool* to add to *active*.

        Parameters
        ----------
        active : dict[mode_id, Counter]
            Modes currently in the active basis (read-only; do not modify).
        pool : dict[mode_id, Counter]
            Remaining candidate modes not yet in the active set.
        cs_ref : CoordinateSystem
            The full reference coordinate system (built at ``max_degree``).
            Use ``cs_ref.find_modes(...)`` to query structural information.

        Returns
        -------
        set[int]
            Mode IDs from *pool* to activate.  An empty set signals that the
            strategy has nothing more to add (the loop will terminate).
        """

    def validate_initial_set(self, active: dict, cs_ref) -> None:
        """
        Validate the initial active set produced by the starting criterion.

        Called by ``make_adaptive_cs`` once, immediately after
        ``starting.initialize()``, before the enrichment loop begins.
        The default implementation is a no-op; subclasses with structural
        requirements on the active set (e.g. downward-closedness) should
        override this to raise ``ValueError`` with a descriptive message
        when the constraint is violated.

        Parameters
        ----------
        active : dict[mode_id, Counter]
            Initial active set as returned by ``starting.initialize()``.
        cs_ref : CoordinateSystem
            Full reference coordinate system.

        Raises
        ------
        ValueError
            If the initial active set is incompatible with this strategy.
        """


class StoppingCriterion(ABC):
    """
    Pluggable policy that decides when the adaptive enrichment loop should
    terminate.
    """

    @abstractmethod
    def should_stop(
        self,
        active:    dict,
        pool:      dict,
        cs_ref,         # CoordinateSystem
        iteration: int,
    ) -> bool:
        """
        Return ``True`` to terminate the adaptive loop *before* the next
        ``strategy.select`` call.

        Parameters
        ----------
        active : dict[mode_id, Counter]
            Modes currently in the active basis.
        pool : dict[mode_id, Counter]
            Remaining candidates.
        cs_ref : CoordinateSystem
            Full reference coordinate system.
        iteration : int
            Zero-based enrichment step count (0 before any modes added).
        """


# ──────────────────────────────────────────────────────────────────────────────
# Concrete strategies
# ──────────────────────────────────────────────────────────────────────────────

class LevelByLevelStrategy(AdaptiveBasisStrategy):
    """
    Add all modes whose total polynomial degree equals the next level.

    This is the strategy implicitly used by the ISQS hierarchy:
    level 0 (mean) → level 1 (linear) → level 2 (quadratic) → …

    At each call to :meth:`select`, the minimum total degree present in the
    pool is chosen and *all* modes at that degree are returned.

    Parameters
    ----------
    max_degree : int
        Maximum total degree to include in the candidate pool.
    """

    def __init__(self, max_degree: int):
        self._max_degree = max_degree

    @property
    def max_degree(self) -> int:
        return self._max_degree

    def select(self, active, pool, cs_ref) -> set:
        if not pool:
            return set()
        next_level = min(sum(degs.values()) for degs in pool.values())
        return {mid for mid, degs in pool.items()
                if sum(degs.values()) == next_level}


class SensitivityDrivenStrategy(AdaptiveBasisStrategy):
    """
    Add modes in decreasing order of parameter variance contribution.

    At each call the parameter with the highest variance that does **not**
    yet have its pure-linear mode in the active set is enriched first.
    Once all pure-linear modes are active, cross-term enrichment proceeds
    by adding the mode with the smallest total degree that involves the
    dominant parameter.

    This strategy encodes the BSF insight: parameters with large
    ``alpha_k = sqrt(Var(k))`` dominate the sensitivity and should be
    enriched first.

    Parameters
    ----------
    max_degree : int
        Maximum total degree of the candidate pool.
    variances : list[float]
        Per-parameter variances in **local** index order (0-based).
        Determines enrichment priority.
    batch_size : int, default 1
        Number of modes to return per :meth:`select` call.
    """

    def __init__(self, max_degree: int, variances: list, batch_size: int = 1):
        self._max_degree = max_degree
        self._variances  = list(variances)
        self._batch_size = batch_size
        # Params ranked by descending variance (local 0-based indices)
        self._ranked = sorted(range(len(variances)),
                               key=lambda k: variances[k], reverse=True)

    @property
    def max_degree(self) -> int:
        return self._max_degree

    def select(self, active, pool, cs_ref) -> set:
        if not pool:
            return set()

        selected  = set()
        remaining = dict(pool)

        for k in self._ranked:
            if len(selected) >= self._batch_size:
                break
            # Find the lowest-total-degree mode in the pool that involves
            # parameter k with at least degree 1
            candidates = {mid: degs for mid, degs in remaining.items()
                          if degs.get(k, 0) >= 1}
            if not candidates:
                continue
            # Pick the one with smallest total degree (break ties by mode ID)
            best = min(candidates, key=lambda m: (sum(candidates[m].values()), m))
            selected.add(best)
            remaining.pop(best)

        return selected


class DownwardClosedStrategy(AdaptiveBasisStrategy):
    """
    Maintain the downward-closed (lower set / monotone) admissibility
    constraint and add the highest-scoring admissible candidate.

    A mode α is **admissible** if for every axis i where α_i > 0, the
    index α − e_i (axis i decremented by 1) is already in the active set.
    This guarantees the active set is always a downward-closed multi-index
    set, which is required for the Smolyak/sparse-grid correctness proofs
    and for the nested quadrature rules used in hp-refinement.

    Parameters
    ----------
    max_degree : int
        Maximum total degree of the candidate pool.
    scorer : callable, optional
        ``scorer(mode_id, degs, active, cs_ref) -> float``
        Higher score = higher priority.  Defaults to ``-total_degree``
        (i.e., prefer lower-degree modes — same as level-by-level but
        restricted to admissible candidates).
    batch_size : int, default 1
        Number of admissible modes to return per :meth:`select` call.
    """

    def __init__(self, max_degree: int, scorer=None, batch_size: int = 1):
        self._max_degree = max_degree
        self._scorer     = scorer or (lambda mid, degs, active, cs_ref:
                                       -sum(degs.values()))
        self._batch_size = batch_size

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @staticmethod
    def _admissible(degs: Counter, active: dict) -> bool:
        """True iff every α − e_i (for axes i where α_i > 0) is in active."""
        # Normalise: keys must exclude zero-degree axes so that the mean
        # mode (all zeros) and any reduced index compare correctly.
        def _key(d):
            return tuple(sorted((k, v) for k, v in d.items() if v > 0))

        active_degs = {_key(d) for d in active.values()}
        for axis, d in degs.items():
            if d > 0:
                reduced = Counter({k: v for k, v in degs.items() if v > 0})
                reduced[axis] -= 1
                if reduced[axis] == 0:
                    del reduced[axis]
                if _key(reduced) not in active_degs:
                    return False
        return True

    def select(self, active, pool, cs_ref) -> set:
        if not pool:
            return set()

        admissible = {mid: degs for mid, degs in pool.items()
                      if self._admissible(degs, active)}
        if not admissible:
            return set()

        # Sort by score (descending), break ties by mode ID
        ranked = sorted(admissible,
                         key=lambda m: (-self._scorer(m, admissible[m],
                                                       active, cs_ref), m))
        return set(ranked[:self._batch_size])

    def validate_initial_set(self, active: dict, cs_ref) -> None:
        """
        Verify the initial active set is downward-closed.

        Raises
        ------
        ValueError
            If any mode α in *active* has a predecessor α − e_i that is
            absent from *active*.  A non-downward-closed seed will cause
            the strategy to permanently block admissible candidates,
            producing a mathematically unsound basis.

        Note
        ----
        To fix a non-DC seed, use :class:`LevelStarting` or
        :class:`MeanOnlyStarting` instead of :class:`FixedModeSetStarting`,
        or ensure your fixed set is itself downward-closed.
        """
        def _key(d):
            return tuple(sorted((k, v) for k, v in d.items() if v > 0))

        active_keys = {_key(degs) for degs in active.values()}
        for mid, degs in active.items():
            for axis, d in degs.items():
                if d > 0:
                    reduced = Counter({k: v for k, v in degs.items() if v > 0})
                    reduced[axis] -= 1
                    if reduced[axis] == 0:
                        del reduced[axis]
                    if _key(reduced) not in active_keys:
                        raise ValueError(
                            f"DownwardClosedStrategy: initial active set is not "
                            f"downward-closed.  Mode {mid} with degrees "
                            f"{dict(degs)} is active, but its predecessor "
                            f"{dict(reduced)} (axis {axis} decremented) is not.  "
                            f"Use LevelStarting or MeanOnlyStarting, or supply a "
                            f"downward-closed FixedModeSetStarting seed."
                        )


# ──────────────────────────────────────────────────────────────────────────────
# Concrete stopping criteria
# ──────────────────────────────────────────────────────────────────────────────

class CandidatePoolExhaustedStopping(StoppingCriterion):
    """
    Stop only when the candidate pool is empty (default behaviour).

    This is the most permissive criterion: the loop runs until the strategy
    has nothing more to add or the pool is exhausted.
    """

    def should_stop(self, active, pool, cs_ref, iteration) -> bool:
        return not pool


class MaxIterationsStopping(StoppingCriterion):
    """
    Stop after a fixed number of enrichment iterations.

    Parameters
    ----------
    max_iterations : int
        Maximum number of times :meth:`~AdaptiveBasisStrategy.select` may
        be called.  At ``iteration == max_iterations`` the loop terminates.
    """

    def __init__(self, max_iterations: int):
        self._max = max_iterations

    def should_stop(self, active, pool, cs_ref, iteration) -> bool:
        return iteration >= self._max


class RelativeGrowthStopping(StoppingCriterion):
    """
    Stop when the relative size of the last enrichment batch falls below a
    threshold.

    Specifically, tracks ``|selected| / |active|`` at each step and stops
    when this ratio is below *tol*.  Useful for anisotropic problems where
    enrichment naturally stalls along inactive directions.

    Parameters
    ----------
    tol : float
        Relative growth threshold.  Typical values: 0.05–0.20.
    """

    def __init__(self, tol: float):
        self._tol   = tol
        self._ratio = 1.0   # initialised large so first iteration always runs

    def record(self, n_selected: int, n_active: int) -> None:
        """Called by make_adaptive_cs after each select() call."""
        self._ratio = n_selected / max(n_active, 1)

    def should_stop(self, active, pool, cs_ref, iteration) -> bool:
        return iteration > 0 and self._ratio < self._tol
