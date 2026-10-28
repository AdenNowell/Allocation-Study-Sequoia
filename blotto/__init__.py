"""
blotto-simulation-engine: Production simulation and mathematical analysis package
for the 5-castle weighted Colonel Blotto tournament.
"""

from blotto.core.rules import (
    CASTLE_VALUES,
    GAME2_RANK_PAYOUTS,
    NUM_CASTLES,
    TOTAL_SOLDIERS,
    validate_allocation,
    validate_population_matrix,
)
from blotto.core.scoring import (
    exact_game1_dp_best_response,
    score_1v1_matrix,
    score_1v1_pair,
    score_4way_pod,
    score_4way_pods_batch,
)

__version__ = "1.0.0"

__all__ = [
    "CASTLE_VALUES",
    "GAME2_RANK_PAYOUTS",
    "NUM_CASTLES",
    "TOTAL_SOLDIERS",
    "validate_allocation",
    "validate_population_matrix",
    "score_1v1_pair",
    "score_1v1_matrix",
    "exact_game1_dp_best_response",
    "score_4way_pod",
    "score_4way_pods_batch",
]
