#=====================================================================#
# Adaptive basis selection for CoordinateSystem.make_adaptive_cs().
#
# Separation of concerns — three orthogonal axes:
# ─────────────────────────────────────────────────────────────────────
#   StartingCriterion      — *where* to begin (initial active set)
#   AdaptiveBasisStrategy  — *what* modes to add at each step
#   Convergence            — *when* to stop; same operation over two operand spaces:
#
#       Operand: index set     error() = pool size, iteration budget, growth ratio
#       Operand: domain        error() = L2 residual norm  (ResidualNormConvergence)
#
# Contextual Reflection of Abstract Interfaces (CRAI):
#   AdaptiveContext carries all loop state uniformly.  Each Convergence
#   implementation extracts the fields it needs; the loop is indifferent
#   to which operand space is in use.
#
# The adaptive loop in make_adaptive_cs() orchestrates:
#
#   active  = starting.initialize(pool, cs_ref)
#   pool    = all candidate modes up to strategy.max_degree
#   while pool and not convergence.is_met():
#       selected = strategy.select(active, pool, cs_ref)
#       active  += selected;  pool -= selected
#
# Author : Komahan Boopathy (komahan@gatech.edu)
#=====================================================================#

from __future__ import annotations

from abc        import ABC, abstractmethod
from collections import Counter
from typing     import TYPE_CHECKING, NamedTuple, Optional

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
        mean_candidates = [mid for mid, degs in pool.items()
                           if sum(degs.values()) == 0]
        if not mean_candidates:
            # mean is already in the baseline (min_degree > 0); nothing to do
            return {}
        mean_id = mean_candidates[0]
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

    Direction
    ---------
    Every strategy has a :attr:`direction` property (``'grow'`` by default).
    Calling :meth:`reverse` returns a ``_ReversedStrategy`` wrapper that
    swaps ``(active, pool)`` in every :meth:`select` call, turning a growth
    rule into a **decay rule** without touching any concrete subclass:

    .. code-block:: python

        # grow: level-by-level enrichment
        cs_a = cs.make_adaptive_cs(LevelByLevelStrategy(3))

        # decay: level-by-level pruning (remove highest-degree modes first)
        cs_a = cs.make_adaptive_cs(LevelByLevelStrategy(3).reverse(),
                                   starting=LevelStarting(3))
    """

    @property
    @abstractmethod
    def max_degree(self) -> int:
        """
        Maximum total polynomial degree of the candidate pool.
        The reference CS used for candidate generation is built at this degree.
        """

    @property
    def min_degree(self) -> int:
        """
        Minimum total polynomial degree of the candidate pool (default 0).

        Modes with ``|α| < min_degree`` are automatically placed in the
        **baseline** active set by ``make_adaptive_cs`` and are never offered
        to the starting criterion or the enrichment loop.  This enables
        warm-restarts: set ``min_degree = k`` to continue enriching a basis
        that already covers levels 0 … k−1, without redundantly re-adding
        those modes to the pool.

        The baseline is always a complete level set and therefore trivially
        satisfies the downward-closed constraint, so admissibility checks on
        new candidates remain well-defined regardless of the starting criterion.

        Override in subclasses that expose a user-facing ``min_degree``
        parameter.  The default returns 0 (no baseline pre-seeding, full pool).
        """
        return 0

    @property
    def direction(self) -> str:
        """
        ``'grow'`` (default) or ``'decay'``.

        * ``'grow'`` — :meth:`select` returns modes to move from *pool* into
          *active* (standard enrichment).
        * ``'decay'`` — :meth:`select` returns modes to evict from *active*
          back into *pool* (pruning / basis reduction).

        Set automatically by :meth:`reverse`; do not override manually.
        """
        return 'grow'

    def reverse(self) -> '_ReversedStrategy':
        """
        Return a decay-mode wrapper around this strategy.

        The wrapper swaps ``(active, pool)`` in every :meth:`select` call so
        that the same scoring/selection logic that drives *enrichment* now
        drives *pruning*.  All concrete strategy subclasses inherit this for
        free — no subclass needs to be modified.

        ``make_adaptive_cs`` inspects :attr:`direction` and reverses the
        transfer direction (evict from active → pool) when ``'decay'``.

        Examples
        --------
        Prune a full degree-3 basis down to its most important modes:

        >>> cs_full = cs.make_adaptive_cs(LevelByLevelStrategy(3),
        ...                               starting=LevelStarting(3))
        >>> cs_pruned = cs.make_adaptive_cs(
        ...     SensitivityDrivenStrategy(3, variances).reverse(),
        ...     starting=LevelStarting(3),          # start full
        ...     stopping=MaxIterationsStopping(4),  # prune 4 modes
        ... )
        """
        return _ReversedStrategy(self)

    @abstractmethod
    def select(
        self,
        active: dict,
        pool:   dict,
        cs_ref,          # CoordinateSystem
        coeffs: Optional[dict] = None,
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
        coeffs : dict[mode_id, float] or None
            PCE coefficients for the current active set, keyed by the same
            mode IDs as *active*.  Non-None when the caller supplied *f* to
            ``make_adaptive_cs``.  Coefficient-aware strategies (e.g.
            :class:`CoefficientDecayScorer`) use this to score candidates.

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


class _ReversedStrategy(AdaptiveBasisStrategy):
    """
    Decay-mode wrapper produced by :meth:`AdaptiveBasisStrategy.reverse`.

    Swaps ``(active, pool)`` in every :meth:`select` call so that the inner
    strategy's scoring logic drives *pruning* instead of *enrichment*:

    * In grow mode:  ``inner.select(active, pool)``  → modes to add
    * In decay mode: ``inner.select(pool, active)``  → modes to evict

    All other properties (``max_degree``, ``min_degree``,
    ``validate_initial_set``) are delegated to the inner strategy unchanged.
    Calling ``.reverse()`` on a ``_ReversedStrategy`` returns the original
    strategy (double reversal = identity).
    """

    def __init__(self, inner: AdaptiveBasisStrategy):
        self._inner = inner

    @property
    def max_degree(self) -> int:
        return self._inner.max_degree

    @property
    def min_degree(self) -> int:
        return self._inner.min_degree

    @property
    def direction(self) -> str:
        return 'decay'

    def reverse(self) -> AdaptiveBasisStrategy:
        """Double reversal returns the original strategy (identity)."""
        return self._inner

    def select(self, active: dict, pool: dict, cs_ref, coeffs=None) -> set:
        # Swap: inner sees (pool → active), returns modes to evict from active
        return self._inner.select(pool, active, cs_ref, coeffs=coeffs)

    def validate_initial_set(self, active: dict, cs_ref) -> None:
        # In decay mode the initial active set is the full pool, which is
        # always a valid lower set; delegate to inner for completeness.
        self._inner.validate_initial_set(active, cs_ref)


# ──────────────────────────────────────────────────────────────────────────────
# AdaptiveContext — uniform context carrier for the adaptive enrichment loop
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveContext(NamedTuple):
    """
    All loop state in one object.

    Both index-space and function-space Convergence implementations receive
    the same context; each extracts the fields relevant to its operand type.

    Fields
    ------
    active, pool : dict
        Current active basis and candidate pool (index-space operands).
    cs_ref : CoordinateSystem
        Full reference coordinate system.
    iteration : int
        Zero-based enrichment step count.
    direction : str
        'grow' or 'decay' — the strategy's current direction.
    n_selected : int
        Number of modes moved in the *previous* iteration (for growth-ratio
        computation).  Zero on the first iteration.
    coeffs : dict or None
        PCE coefficients from decompose(f) on the current active basis.
        Non-None only when the caller provides a function for residual-norm
        convergence.
    """
    active:     dict
    pool:       dict
    cs_ref:     'CoordinateSystem'
    iteration:  int
    direction:  str            = 'grow'
    n_selected: int            = 0
    coeffs:     Optional[dict] = None


# ──────────────────────────────────────────────────────────────────────────────
# Convergence ABC  (CRAI — same operation over index-space or domain operand)
# ──────────────────────────────────────────────────────────────────────────────

class Convergence(ABC):
    """
    Abstract operation: measure convergence over an operand.

    The same protocol serves two operand spaces:

      Index space (discrete)   — operand is (active, pool, iteration, …)
      Function space (continuous) — operand is (cs_ref, coeffs, f)

    Implementing a new criterion means:
      1. Override ``update(ctx)``  — ingest the current AdaptiveContext.
      2. Override ``error()``      — return a non-negative float; 0 = converged.
      3. Override ``is_met()``     — if a custom tolerance is needed.

    The adaptive loop calls ``update(ctx)`` then ``is_met()`` each iteration,
    indifferent to which operand space the criterion lives in.
    """

    @abstractmethod
    def update(self, ctx: AdaptiveContext) -> None:
        """Ingest the current loop context before querying error()."""

    @abstractmethod
    def error(self) -> float:
        """Current convergence error. Returns 0.0 when fully converged."""

    def is_met(self) -> bool:
        """True when convergence is achieved (error <= 0)."""
        return self.error() <= 0.0


# Backward-compat bridge: old code that subclasses StoppingCriterion and
# implements should_stop() continues to work via the Convergence protocol.
class StoppingCriterion(Convergence):
    """
    Backward-compatible base for pre-CRAI stopping criteria.
    New code should extend Convergence and implement update()/error() directly.
    """

    def __init__(self):
        self._ctx = AdaptiveContext({}, {}, None, 0)

    def update(self, ctx: AdaptiveContext) -> None:
        self._ctx = ctx

    def error(self) -> float:
        ctx = self._ctx
        return 0.0 if self.should_stop(
            ctx.active, ctx.pool, ctx.cs_ref, ctx.iteration
        ) else 1.0

    @abstractmethod
    def should_stop(self, active, pool, cs_ref, iteration) -> bool:
        """Return True to terminate the adaptive loop."""


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
    min_degree : int, default 0
        Minimum total degree to include in the candidate pool.  Modes with
        ``|α| < min_degree`` are pre-seeded as the baseline by
        ``make_adaptive_cs`` and are never placed in the pool.
    """

    def __init__(self, max_degree: int, min_degree: int = 0):
        self._max_degree = max_degree
        self._min_degree = min_degree

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @property
    def min_degree(self) -> int:
        return self._min_degree

    def select(self, active, pool, cs_ref, coeffs=None) -> set:
        if not pool:
            return set()
        next_level = min(sum(degs.values()) for degs in pool.values())
        return {mid for mid, degs in pool.items()
                if sum(degs.values()) == next_level}


class _LevelByLevelDecayStrategy(AdaptiveBasisStrategy):
    """
    Max-degree-first decay for use by AdaptiveOperator.inverse.

    Removes all modes at the **maximum** total degree present in the
    active set, one level at a time.  This is the correct inverse of
    LevelByLevelStrategy: grow adds levels min→max ascending, so decay
    should remove them max→min descending, restoring the original active
    set (Involution law holds).

    This class is intentionally NOT returned by
    ``LevelByLevelStrategy.reverse()`` (which keeps the public
    ``_ReversedStrategy`` contract for backward compatibility); it is
    used directly by ``AdaptiveOperator.inverse``.
    """

    def __init__(self, max_degree: int, min_degree: int = 0):
        self._max_degree = max_degree
        self._min_degree = min_degree

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @property
    def min_degree(self) -> int:
        return self._min_degree

    @property
    def direction(self) -> str:
        return 'decay'

    def select(self, active: dict, pool: dict, cs_ref, coeffs=None) -> set:
        if not active:
            return set()
        max_deg = max(sum(d.values()) for d in active.values())
        if max_deg <= self._min_degree:
            return set()
        return {mid for mid, degs in active.items()
                if sum(degs.values()) == max_deg}

    def validate_initial_set(self, active, cs_ref) -> None:
        pass

    def reverse(self) -> 'LevelByLevelStrategy':
        return LevelByLevelStrategy(self._max_degree, self._min_degree)


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
    min_degree : int, default 0
        Minimum total degree to include in the candidate pool.  Modes with
        ``|α| < min_degree`` are pre-seeded as the baseline by
        ``make_adaptive_cs`` and are never placed in the pool.
    """

    def __init__(self, max_degree: int, variances: list,
                 batch_size: int = 1, min_degree: int = 0):
        self._max_degree = max_degree
        self._min_degree = min_degree
        self._variances  = list(variances)
        self._batch_size = batch_size
        # Params ranked by descending variance (local 0-based indices)
        self._ranked = sorted(range(len(variances)),
                               key=lambda k: variances[k], reverse=True)

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @property
    def min_degree(self) -> int:
        return self._min_degree

    def select(self, active, pool, cs_ref, coeffs=None) -> set:
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

        # Fallback: if no sensitivity-based candidate was found but the pool
        # still has modes (e.g. the all-zero mean mode in decay mode), pick the
        # lowest-degree one.  This ensures the Zero law holds under reversal.
        if not selected and remaining:
            best = min(remaining, key=lambda m: (sum(remaining[m].values()), m))
            selected.add(best)

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
    min_degree : int, default 0
        Minimum total degree to include in the candidate pool.  Modes with
        ``|α| < min_degree`` are pre-seeded as the **baseline** active set
        by ``make_adaptive_cs`` before the enrichment loop starts.  This
        enables warm-restarts — set ``min_degree = k`` to enrich a basis
        that already covers levels 0 … k−1.

        Invariant: the baseline is always a complete level set and therefore
        trivially downward-closed, so the DC constraint remains valid for all
        new candidates regardless of the starting criterion.
    """

    def __init__(self, max_degree: int, scorer=None,
                 batch_size: int = 1, min_degree: int = 0):
        self._max_degree = max_degree
        self._min_degree = min_degree
        self._scorer     = scorer or (lambda mid, degs, active, cs_ref,
                                       coeffs=None: -sum(degs.values()))
        self._batch_size = batch_size

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @property
    def min_degree(self) -> int:
        return self._min_degree

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

    def select(self, active: dict, pool: dict, cs_ref, coeffs=None) -> set:
        if not pool:
            return set()

        admissible = {mid: degs for mid, degs in pool.items()
                      if self._admissible(degs, active)}
        if not admissible:
            return set()

        # Sort by score (descending), break ties by mode ID
        ranked = sorted(admissible,
                         key=lambda m: (-self._scorer(m, admissible[m],
                                                       active, cs_ref, coeffs), m))
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


class CoefficientDecayScorer:
    """
    Natural error indicator for :class:`DownwardClosedStrategy`.

    Scores a candidate mode ``α`` by the **maximum absolute PCE coefficient
    among its immediate predecessors** in the current active set:

        score(α) = max{ |c_{α − eᵢ}| : αᵢ > 0, α − eᵢ ∈ active }

    The intuition (BSF / ISQS framework): if a boundary coefficient is large,
    the polynomial energy is still flowing in that direction — adding the next
    mode is likely to contribute significant energy.  Conversely, if all
    predecessors have decayed to near zero, the candidate can be deferred.

    When ``coeffs`` is ``None`` (i.e. ``f`` was not passed to
    ``make_adaptive_cs``), falls back to ``−total_degree`` (level-by-level
    order) so the scorer is always safe to use.

    Usage
    -----
    ::

        scorer = CoefficientDecayScorer()
        cs_a   = cs.make_adaptive_cs(
            DownwardClosedStrategy(max_degree=4, scorer=scorer, batch_size=1),
            f=f,    # <-- enables coefficient supply to ctx and scorer
        )
    """

    @staticmethod
    def _key(degs):
        """Normalised degree key: frozenset of (axis, degree) pairs, zeros dropped."""
        return frozenset((k, v) for k, v in degs.items() if v > 0)

    def __call__(self, mode_id, degs, active, cs_ref, coeffs=None):
        if not coeffs:
            return -sum(degs.values())          # fallback: level-by-level order

        # Build reverse map: normalised degree key → coefficient
        key_to_coeff = {self._key(d): coeffs[mid]
                        for mid, d in active.items()
                        if mid in coeffs}

        max_pred = 0.0
        for axis, d in degs.items():
            if d > 0:
                pred = Counter({k: v for k, v in degs.items() if v > 0})
                pred[axis] -= 1
                if pred[axis] == 0:
                    del pred[axis]
                c = key_to_coeff.get(self._key(pred))
                if c is not None:
                    max_pred = max(max_pred, abs(c))

        # Return negative so that DownwardClosedStrategy's sort (ascending)
        # picks the candidate with the LARGEST predecessor coefficient first.
        # Tie-break handled externally by mode_id.
        return max_pred


# ──────────────────────────────────────────────────────────────────────────────
# Concrete Convergence implementations
# ──────────────────────────────────────────────────────────────────────────────

class CandidatePoolExhaustedConvergence(Convergence):
    """
    Index-space convergence: stop when the *source* of the enrichment loop
    is exhausted.

    * **grow** mode — error = |pool|;  converged when pool is empty.
    * **decay** mode — error = |active|; converged when active is empty.

    Direction flows via ``AdaptiveContext.direction``; no external injection
    needed.
    """

    def __init__(self):
        self._size = 1   # non-zero until first update

    def update(self, ctx: AdaptiveContext) -> None:
        source = ctx.pool if ctx.direction == 'grow' else ctx.active
        self._size = len(source)

    def error(self) -> float:
        return float(self._size)


class MaxIterationsConvergence(Convergence):
    """
    Index-space convergence: stop after a fixed number of enrichment steps.

    error() = remaining iterations budget (0 when budget is exhausted).

    Parameters
    ----------
    max_iterations : int
        Maximum number of :meth:`~AdaptiveBasisStrategy.select` calls.
    """

    def __init__(self, max_iterations: int):
        self._max       = max_iterations
        self._remaining = float(max_iterations)

    def update(self, ctx: AdaptiveContext) -> None:
        self._remaining = float(max(0, self._max - ctx.iteration))

    def error(self) -> float:
        return self._remaining


class RelativeGrowthConvergence(Convergence):
    """
    Index-space convergence: stop when the relative enrichment batch size
    falls below a threshold.

    error() = |selected_{k-1}| / |active_k|  (growth ratio from the previous
    step).  Initialised to 1.0 so the first iteration always runs.

    Parameters
    ----------
    tol : float
        Relative growth threshold.  Typical values: 0.05–0.20.
    """

    def __init__(self, tol: float):
        self._tol   = tol
        self._ratio = 1.0   # large initial value prevents premature stopping

    def update(self, ctx: AdaptiveContext) -> None:
        if ctx.iteration > 0:
            self._ratio = ctx.n_selected / max(len(ctx.active), 1)

    def error(self) -> float:
        return self._ratio

    def is_met(self) -> bool:
        return self._ratio < self._tol


class ResidualNormConvergence(Convergence):
    """
    Function-space convergence: stop when the L2 PCE residual norm is below
    a tolerance.

        error() = ||reconstruct(coeffs) - f||_2

    This is the continuous (domain) operand dual of the discrete
    (index-space) CandidatePoolExhaustedConvergence:

        Index space:    error() = |pool|        -> 0 when basis is complete
        Function space: error() = residual_norm -> 0 when f is represented exactly

    CRAI — same abstract operation (convergence measurement), different operand.

    Parameters
    ----------
    f : PolyFunction
        The function being approximated.
    tol : float
        L2 residual tolerance.  Convergence when error() <= tol.

    Notes
    -----
    ``ctx.coeffs`` must be non-None for ``update()`` to compute the norm.
    Pass ``f`` to ``make_adaptive_cs()`` to have the loop supply coefficients.
    """

    def __init__(self, f, tol: float):
        self._f    = f
        self._tol  = tol
        self._norm = float('inf')

    def update(self, ctx: AdaptiveContext) -> None:
        if ctx.coeffs is not None:
            self._norm = ctx.cs_ref.residual_norm(ctx.coeffs, self._f)

    def error(self) -> float:
        return self._norm

    def is_met(self) -> bool:
        return self._norm <= self._tol


# Backward-compat aliases (old names still importable)
CandidatePoolExhaustedStopping = CandidatePoolExhaustedConvergence
MaxIterationsStopping          = MaxIterationsConvergence
RelativeGrowthStopping         = RelativeGrowthConvergence


# ──────────────────────────────────────────────────────────────────────────────
# AdaptiveOperator — Operation contract for the index space
# ──────────────────────────────────────────────────────────────────────────────

if TYPE_CHECKING:
    from .core import Operation  # only for type hints


class AdaptiveOperator:
    """
    Index-space implementation of the Operation contract:

        forward(cs)       = grow:  cs.make_adaptive_cs(strategy)
        inverse(grown_cs) = decay: grown_cs.make_adaptive_cs(strategy.reverse())
        residual(cs)      = symmetric difference of basis after round-trip

    The Involution law (index-space Fundamental Theorem):

        residual(cs) == frozenset()   iff   decay(grow(cs)).basis == cs.basis

    This is the index-space analogue of CoordinateSystem.residual_norm == 0.
    CRAI: same Operation, different operand (index set vs function domain).

    Parameters
    ----------
    strategy : AdaptiveBasisStrategy
        The growth strategy.  Decay uses strategy.reverse() automatically.
    stopping : Convergence, optional
        Convergence criterion.  Defaults to CandidatePoolExhaustedConvergence.
    starting : StartingCriterion, optional
        Initial active set policy.
    """

    def __init__(self, strategy, stopping=None, starting=None):
        self._strategy = strategy
        self._stopping = stopping
        self._starting = starting

    def forward(self, cs: 'CoordinateSystem') -> 'CoordinateSystem':
        """Grow: enrich cs using the strategy."""
        return cs.make_adaptive_cs(self._strategy, self._stopping, self._starting)

    def inverse(self, grown_cs: 'CoordinateSystem') -> 'CoordinateSystem':
        """Decay: prune grown_cs back using max-degree-first removal."""
        return grown_cs.make_adaptive_cs(
            _LevelByLevelDecayStrategy(self._strategy.max_degree,
                                       self._strategy.min_degree),
            starting=LevelStarting(self._strategy.max_degree),
        )

    def residual(self, cs: 'CoordinateSystem') -> frozenset:
        """
        Symmetric difference of the basis after a grow/decay round-trip.

        Returns frozenset() (empty) iff the Involution law holds:
            decay(grow(cs)).basis == cs.basis
        """
        recovered  = self.inverse(self.forward(cs))
        orig_modes = frozenset(frozenset(d.items()) for d in cs.basis.values())
        recv_modes = frozenset(frozenset(d.items()) for d in recovered.basis.values())
        return orig_modes ^ recv_modes   # symmetric difference


