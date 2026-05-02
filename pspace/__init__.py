"""
pspace — Polynomial Chaos Expansion library
============================================
Public API
----------
CoordinateFactory     : create Normal, Uniform, or Exponential coordinate axes
CoordinateSystem      : assemble multi-dimensional probability space with chosen basis
BasisFunctionType     : TENSOR_DEGREE | TOTAL_DEGREE
PolyFunction          : polynomial f(y) as list of (coeff, Counter{axis:degree}) terms
OrthoPolyFunction     : polynomial expressed in the orthonormal PCE basis
StateEquation         : assemble, precondition, and solve a linear system in the PCE basis
DistributionType      : NORMAL | UNIFORM | EXPONENTIAL (enum, for introspection)
PointSampler          : abstract sampler interface — iterate (point, weight) pairs
QuadratureSampler     : tensor-product Gauss quadrature sampler
MonteCarloSampler     : pseudo-random Monte Carlo sampler
"""

from .core import (
    CoordinateFactory,
    CoordinateSystem,
    BasisFunctionType,
    DistributionType,
    CoordinateType,
    Projection,
    PolyFunction,
    OrthoPolyFunction,
    StateEquation,
    PointSampler,
    QuadratureSampler,
    MonteCarloSampler,
)

from .adaptive import (
    StartingCriterion,
    MeanOnlyStarting,
    LevelStarting,
    SensitivityStarting,
    FixedModeSetStarting,
    AdaptiveBasisStrategy,
    AdaptiveContext,
    Convergence,
    StoppingCriterion,
    LevelByLevelStrategy,
    SensitivityDrivenStrategy,
    DownwardClosedStrategy,
    CandidatePoolExhaustedConvergence,
    MaxIterationsConvergence,
    RelativeGrowthConvergence,
    ResidualNormConvergence,
    CandidatePoolExhaustedStopping,
    MaxIterationsStopping,
    RelativeGrowthStopping,
)

__all__ = [
    "CoordinateFactory",
    "CoordinateSystem",
    "BasisFunctionType",
    "DistributionType",
    "CoordinateType",
    "PolyFunction",
    "OrthoPolyFunction",
    "StateEquation",
    "PointSampler",
    "QuadratureSampler",
    "MonteCarloSampler",
    # adaptive — starting
    "StartingCriterion",
    "MeanOnlyStarting",
    "LevelStarting",
    "SensitivityStarting",
    "FixedModeSetStarting",
    # adaptive — strategy
    "AdaptiveBasisStrategy",
    "LevelByLevelStrategy",
    "SensitivityDrivenStrategy",
    "DownwardClosedStrategy",
    # adaptive — stopping
    "StoppingCriterion",
    "CandidatePoolExhaustedStopping",
    "MaxIterationsStopping",
    "RelativeGrowthStopping",
]
