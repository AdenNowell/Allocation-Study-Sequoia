"""
Generative and noisy opponent models:
- UniformSimplexGenerator: Exact uniform random integer compositions summing to 100 via stars-and-bars.
- DirichletQuantalGenerator: Dirichlet-multinomial generator around castle weights or prior concentrations.
- IntegerTransferPerturber: Local soldier-transfer perturbation around an anchor vector.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from blotto.core.rules import NUM_CASTLES, TOTAL_SOLDIERS, validate_allocation
from blotto.strategies.base import StochasticStrategy


class UniformSimplexGenerator(StochasticStrategy):
    """
    Samples uniformly over all (100 + 5 - 1 choose 4) = 4,598,126 legal integer allocations
    using exact stars-and-bars order statistics.
    """

    def __init__(self, name: str = "UniformIntegerSimplex") -> None:
        super().__init__(
            name=name,
            category="generative",
            description="Exact uniform sampler over all 4,598,126 integer compositions of 100 into 5 parts",
        )

    def get_allocation(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        return self.sample_batch(1, rng=rng)[0]

    def sample_batch(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        gen = rng or np.random.default_rng()
        total_positions = TOTAL_SOLDIERS + NUM_CASTLES - 1  # 104
        out = np.zeros((n, NUM_CASTLES), dtype=np.int64)
        for i in range(n):
            bars = np.sort(gen.choice(total_positions, size=NUM_CASTLES - 1, replace=False))
            out[i, 0] = bars[0]
            out[i, 1] = bars[1] - bars[0] - 1
            out[i, 2] = bars[2] - bars[1] - 1
            out[i, 3] = bars[3] - bars[2] - 1
            out[i, 4] = total_positions - 1 - bars[3]
        return out


class DirichletQuantalGenerator(StochasticStrategy):
    """
    Samples continuous probabilities p ~ Dirichlet(alpha) and converts them into an exact
    100-soldier integer allocation via largest-remainder rounding or multinomial draws.
    """

    def __init__(
        self,
        alpha: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0),
        use_multinomial: bool = True,
        name: str = "DirichletGenerator",
    ) -> None:
        super().__init__(
            name=name,
            category="generative",
            description=f"Dirichlet({list(alpha)}) stochastic strategy",
        )
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if alpha_arr.shape != (NUM_CASTLES,) or np.any(alpha_arr <= 0):
            raise ValueError("alpha must be 5 positive floats.")
        self.alpha = alpha_arr
        self.use_multinomial = use_multinomial

    def get_allocation(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        return self.sample_batch(1, rng=rng)[0]

    def sample_batch(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        gen = rng or np.random.default_rng()
        probs = gen.dirichlet(self.alpha, size=n)
        if self.use_multinomial:
            out = np.zeros((n, NUM_CASTLES), dtype=np.int64)
            for i in range(n):
                out[i] = gen.multinomial(TOTAL_SOLDIERS, probs[i])
            return out

        scaled = probs * TOTAL_SOLDIERS
        floored = np.floor(scaled).astype(np.int64)
        remainders = scaled - floored
        deficits = TOTAL_SOLDIERS - np.sum(floored, axis=1)
        order = np.argsort(-remainders, axis=1)
        for i in range(n):
            d = int(deficits[i])
            if d > 0:
                floored[i, order[i, :d]] += 1
        return floored


class IntegerTransferPerturber(StochasticStrategy):
    """
    Perturbs a base integer allocation by executing `steps` random unit transfers
    between distinct castles while preserving non-negativity and sum == 100.
    """

    def __init__(
        self,
        name: str,
        base_allocation: Sequence[int] | NDArray[np.integer],
        steps: int = 6,
    ) -> None:
        super().__init__(
            name=name,
            category="perturbed",
            description=f"{steps}-step random integer transfer perturbation around {list(base_allocation)}",
        )
        self.base_allocation = validate_allocation(base_allocation)
        self.steps = int(steps)

    @staticmethod
    def perturb_vector(
        base: NDArray[np.int64],
        steps: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        vec = base.copy()
        for _ in range(steps):
            src = int(rng.integers(0, NUM_CASTLES))
            dst = int(rng.integers(0, NUM_CASTLES))
            if src != dst and vec[src] > 0:
                vec[src] -= 1
                vec[dst] += 1
        return vec

    def get_allocation(self, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
        gen = rng or np.random.default_rng()
        return self.perturb_vector(self.base_allocation, self.steps, gen)
