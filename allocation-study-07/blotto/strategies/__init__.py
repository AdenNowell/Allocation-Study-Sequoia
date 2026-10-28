"""Strategy definitions, empirical archetypes, and stochastic field generators."""

from blotto.strategies.archetypes import (
    ALL_ARCHETYPES,
    GAME1_CANDIDATES,
    GAME1_LLM_BASELINES,
    GAME1_SHARK_VECTORS,
    GAME2_CANDIDATES,
    GAME2_LLM_BASELINES,
    GAME2_SHARK_VECTORS,
     build_realistic_field_distribution,
)
from blotto.strategies.base import FixedStrategy, StochasticStrategy, Strategy
from blotto.strategies.generative import (
    DirichletQuantalGenerator,
    IntegerTransferPerturber,
    UniformSimplexGenerator,
)

__all__ = [
    "Strategy",
    "FixedStrategy",
    "StochasticStrategy",
    "GAME1_LLM_BASELINES",
    "GAME2_LLM_BASELINES",
    "GAME1_CANDIDATES",
    "GAME2_CANDIDATES",
    "GAME1_SHARK_VECTORS",
    "GAME2_SHARK_VECTORS",
    "ALL_ARCHETYPES",
    "build_realistic_field_distribution",
    "DirichletQuantalGenerator",
    "IntegerTransferPerturber",
    "UniformSimplexGenerator",
]
