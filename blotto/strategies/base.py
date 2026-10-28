"""Abstract base classes for deterministic and stochastic Colonel Blotto strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from blotto.core.rules import validate_allocation


class Strategy(ABC):
    """Abstract base class for all 5-castle Colonel Blotto strategies."""

    def __init__(self, name: str, category: str = "general", description: str = "") -> None:
        self.name = name
        self.category = category
        self.description = description

    @abstractmethod
    def get_allocation(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        """Return a single valid 5-castle integer allocation summing to 100."""

    def sample_batch(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        """Sample `n` valid allocations of shape (n, 5)."""
        gen = rng or np.random.default_rng()
        return np.stack([self.get_allocation(gen) for _ in range(n)], axis=0)


class FixedStrategy(Strategy):
    """Deterministic strategy that always submits a fixed 5-castle integer vector."""

    def __init__(
        self,
        name: str,
        allocation: Sequence[int] | NDArray[np.integer],
        category: str = "fixed",
        description: str = "",
    ) -> None:
        super().__init__(name=name, category=category, description=description)
        self._allocation: NDArray[np.int64] = validate_allocation(allocation)

    @property
    def vector(self) -> NDArray[np.int64]:
        """Return a copy of the validated 5-element allocation vector."""
        return self._allocation.copy()

    def get_allocation(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        return self._allocation.copy()

    def sample_batch(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        return np.tile(self._allocation, (n, 1))

    def __repr__(self) -> str:
        return f"FixedStrategy(name={self.name!r}, vector={self._allocation.tolist()})"


class StochasticStrategy(Strategy):
    """Base class for randomized/generative strategies."""

    @abstractmethod
    def get_allocation(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        """Sample a single valid allocation vector."""
