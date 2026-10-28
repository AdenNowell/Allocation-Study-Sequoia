"""
Vectorized NumPy evaluation engines for Game 1 (1v1 head-to-head marginals + exact budget DP)
and Game 2 (4-way battle pods with two-layer integer tie-splitting and rank payouts).

Mathematical Notes:
- Game 1 pairwise score:
    S(x, y) = sum_{c=1}^5 c * (1[x_c > y_c] + 0.5 * 1[x_c == y_c])
  Notice S(x, y) + S(y, x) = 15.0 for all (x, y).
- Game 2 pod scoring:
    Layer 1 (Castle ties): Each castle c in {1..5} is won by argmax_{p in {0..3}} x_{p, c}.
    If k players tie for the maximum bid on castle c, each receives c / k points.
    Because k in {1, 2, 3, 4} and lcm(1, 2, 3, 4) = 12, multiplying castle values by 12
    ([12, 24, 36, 48, 60]) makes all Layer-1 castle point totals exact integers!
    Layer 2 (Rank ties): Players are ranked by their total castle points. Rank payouts are
    [9, 4, 2, 0]. If m players tie at rank index r (0-indexed), each receives the average
    of payouts[r : r + m]. Sum of payouts in every pod is identically 15.0.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from blotto.core.rules import (
    CASTLE_VALUES,
    CASTLE_VALUES_INT12,
    GAME2_RANK_PAYOUTS,
    NUM_CASTLES,
    TOTAL_SOLDIERS,
    validate_allocation,
    validate_population_matrix,
)


def score_1v1_pair(
    hero: Sequence[int] | NDArray[np.integer],
    opponent: Sequence[int] | NDArray[np.integer],
) -> tuple[float, float]:
    """
    Compute exact head-to-head Game 1 castle points for (hero, opponent).

    Parameters
    ----------
    hero : Sequence[int] | NDArray
        Hero's 5-castle allocation.
    opponent : Sequence[int] | NDArray
        Opponent's 5-castle allocation.

    Returns
    -------
    tuple[float, float]
        (hero_points, opponent_points), always summing to 15.0.
    """
    h = validate_allocation(hero)
    o = validate_allocation(opponent)

    wins = (h > o).astype(np.float64)
    ties = (h == o).astype(np.float64) * 0.5

    hero_pts = float(np.dot(wins + ties, CASTLE_VALUES))
    opp_pts = 15.0 - hero_pts
    return hero_pts, opp_pts


def score_1v1_matrix(
    candidates: NDArray[np.integer] | Sequence[Sequence[int]],
    field: NDArray[np.integer] | Sequence[Sequence[int]],
    field_weights: NDArray[np.floating] | Sequence[float] | None = None,
) -> NDArray[np.float64]:
    """
    Vectorized pairwise Game 1 evaluation of M candidates against N field opponents.

    Computes the full (M, N) pairwise score matrix using broadcasting over the 5 castles:
        scores[m, n] = S(candidates[m], field[n])
    Also returns weighted mean scores if `field_weights` is provided, or returns the full
    (M, N) matrix when `field_weights` is None.

    Parameters
    ----------
    candidates : NDArray of shape (M, 5)
        Candidate strategies to evaluate.
    field : NDArray of shape (N, 5)
        Opponent population strategies.
    field_weights : Optional 1D array of length N
        If provided, returns the 1D array of weighted average scores of shape (M,).
        If None, returns the full 2D pairwise matrix of shape (M, N).

    Returns
    -------
    NDArray[np.float64]
        Shape (M, N) if field_weights is None, else shape (M,).
    """
    cand_mat = validate_population_matrix(candidates)
    field_mat = validate_population_matrix(field)

    # Broadcast shapes: (M, 1, 5) vs (1, N, 5)
    c_exp = cand_mat[:, None, :]
    f_exp = field_mat[None, :, :]

    wins = (c_exp > f_exp).astype(np.float64)
    ties = (c_exp == f_exp).astype(np.float64) * 0.5

    pairwise = np.tensordot(wins + ties, CASTLE_VALUES, axes=([2], [0]))  # shape (M, N)

    if field_weights is None:
        return pairwise

    w = np.asarray(field_weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != field_mat.shape[0]:
        raise ValueError(f"field_weights must have shape ({field_mat.shape[0]},), got {w.shape}.")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("field_weights must sum to a positive number.")
    w_norm = w / w_sum
    return pairwise @ w_norm


def exact_game1_dp_best_response(
    field: NDArray[np.integer] | Sequence[Sequence[int]],
    field_weights: NDArray[np.floating] | Sequence[float] | None = None,
    fixed_castle_bids: dict[int, int] | None = None,
) -> tuple[NDArray[np.int64], float]:
    """
    Solve the EXACT global best-response allocation in Game 1 over all 4,598,126 legal allocations
    using per-castle marginal distributions and exact budget dynamic programming.

    Because Game 1's payoff is additive across the 5 castles:
        E[S(x)] = sum_{c=0}^4 v_c * (P(Y_c < x_c) + 0.5 * P(Y_c == x_c))
    this 5-stage knapsack DP runs in O(5 * 100^2) operations (< 1 ms) and is certified globally optimal.

    Parameters
    ----------
    field : NDArray of shape (N, 5)
        Opponent field strategies.
    field_weights : Optional 1D array of length N
        Probability weights for each opponent in `field`. Defaults to uniform weights.
    fixed_castle_bids : Optional dict[int, int]
        Optional constraint dictionary mapping 0-indexed castle index (0..4) to a forced bid
        (e.g., `{4: 0}` to force 0 soldiers on Castle 5, replicating Finding 5's opportunity cost).

    Returns
    -------
    tuple[NDArray[np.int64], float]
        (optimal_allocation, optimal_expected_points)
    """
    field_mat = validate_population_matrix(field)
    n_opp = field_mat.shape[0]

    if field_weights is None:
        w = np.full(n_opp, 1.0 / n_opp, dtype=np.float64)
    else:
        w_raw = np.asarray(field_weights, dtype=np.float64)
        w = w_raw / np.sum(w_raw)

    # Precompute marginal reward table reward[c, b] for castle c in 0..4 and bid b in 0..100
    reward = np.zeros((NUM_CASTLES, TOTAL_SOLDIERS + 1), dtype=np.float64)
    for c in range(NUM_CASTLES):
        pmf = np.bincount(field_mat[:, c], weights=w, minlength=TOTAL_SOLDIERS + 1)[: TOTAL_SOLDIERS + 1]
        cdf_strict = np.zeros(TOTAL_SOLDIERS + 1, dtype=np.float64)
        cdf_strict[1:] = np.cumsum(pmf[:-1])
        reward[c, :] = CASTLE_VALUES[c] * (cdf_strict + 0.5 * pmf)

    constraints = fixed_castle_bids or {}

    # DP state: dp[c, s] = max expected points from castles 0..c using total budget s
    dp = np.full((NUM_CASTLES, TOTAL_SOLDIERS + 1), -np.inf, dtype=np.float64)
    parent = np.zeros((NUM_CASTLES, TOTAL_SOLDIERS + 1), dtype=np.int64)

    # Base case: Castle 0
    if 0 in constraints:
        b0 = constraints[0]
        dp[0, b0] = reward[0, b0]
        parent[0, b0] = b0
    else:
        for b in range(TOTAL_SOLDIERS + 1):
            dp[0, b] = reward[0, b]
            parent[0, b] = b

    # Transitions: Castles 1..4
    for c in range(1, NUM_CASTLES):
        allowed_bids = [constraints[c]] if c in constraints else range(TOTAL_SOLDIERS + 1)
        for s in range(TOTAL_SOLDIERS + 1):
            best_val = -np.inf
            best_b = 0
            for b in allowed_bids:
                if b <= s and dp[c - 1, s - b] > -1e9:
                    val = dp[c - 1, s - b] + reward[c, b]
                    if val > best_val:
                        best_val = val
                        best_b = b
            dp[c, s] = best_val
            parent[c, s] = best_b

    best_score = float(dp[NUM_CASTLES - 1, TOTAL_SOLDIERS])
    if not np.isfinite(best_score):
        raise ValueError("Infeasible fixed_castle_bids constraints: budget cannot sum to 100.")

    alloc = np.zeros(NUM_CASTLES, dtype=np.int64)
    rem = TOTAL_SOLDIERS
    for c in range(NUM_CASTLES - 1, -1, -1):
        b = int(parent[c, rem])
        alloc[c] = b
        rem -= b

    return alloc, best_score


def score_4way_pod(
    p0: Sequence[int] | NDArray[np.integer],
    p1: Sequence[int] | NDArray[np.integer],
    p2: Sequence[int] | NDArray[np.integer],
    p3: Sequence[int] | NDArray[np.integer],
) -> NDArray[np.float64]:
    """
    Scalar reference scorer for a single 4-player Game 2 pod.

    Uses exact integer units (12x castle points) to eliminate floating-point representation bugs
    on 3-way splits (e.g., 1/3 + 2/3 == 1.0 or 4/3 + 5/3 == 3.0) before computing rank payouts.

    Parameters
    ----------
    p0, p1, p2, p3 : Sequence[int] | NDArray
        The four players' 5-castle allocations.

    Returns
    -------
    NDArray[np.float64]
        1D array of length 4 containing the rank payouts for [p0, p1, p2, p3].
        Always sums to 15.0.
    """
    pod = np.stack(
        [
            validate_allocation(p0),
            validate_allocation(p1),
            validate_allocation(p2),
            validate_allocation(p3),
        ],
        axis=0,
    )  # shape (4, 5)

    pts_int12 = np.zeros(4, dtype=np.int64)
    for c in range(NUM_CASTLES):
        col = pod[:, c]
        mx = int(np.max(col))
        winners = col == mx
        count = int(np.sum(winners))
        share = int(CASTLE_VALUES_INT12[c]) // count
        pts_int12[winners] += share

    payouts = np.zeros(4, dtype=np.float64)
    for i in range(4):
        greater = int(np.sum(pts_int12 > pts_int12[i]))
        equal = int(np.sum(pts_int12 == pts_int12[i]))
        payouts[i] = float(np.sum(GAME2_RANK_PAYOUTS[greater : greater + equal])) / equal

    return payouts


def score_4way_pods_batch(
    pods: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    High-speed vectorized NumPy scorer for K four-player pods simultaneously.

    Both layers of tie handling (castle max-bid ties and final rank ties) are evaluated
    in exact integer units (12x castle values) across all K pods without Python loops over K.

    Parameters
    ----------
    pods : NDArray[np.integer] of shape (K, 4, 5)
        Batch of K four-player pods, where pods[k, p, c] is player p's bid on castle c in pod k.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        - payouts: shape (K, 4) rank payouts in [0, 9] for each player in each pod (row sums == 15.0).
        - castle_points: shape (K, 4) raw castle points in [0, 15] won by each player in each pod.
    """
    pods_arr = np.asarray(pods, dtype=np.int64)
    if pods_arr.ndim != 3 or pods_arr.shape[1:] != (4, NUM_CASTLES):
        raise ValueError(f"pods must have shape (K, 4, 5), got {pods_arr.shape}.")

    # Layer 1: Castle point resolution in 12x integer units
    # max_bids shape: (K, 1, 5)
    max_bids = np.max(pods_arr, axis=1, keepdims=True)
    is_winner = pods_arr == max_bids  # (K, 4, 5) bool
    tie_counts = np.sum(is_winner, axis=1, keepdims=True)  # (K, 1, 5) in {1, 2, 3, 4}

    # Integer division is exact because 12 is divisible by 1, 2, 3, and 4
    shares_int12 = (CASTLE_VALUES_INT12[None, None, :] // tie_counts) * is_winner  # (K, 4, 5)
    pts_int12 = np.sum(shares_int12, axis=2)  # (K, 4)

    # Layer 2: Rank resolution across the 4 players in each pod
    # Compare every player i against every player j: shapes (K, 4, 1) vs (K, 1, 4)
    pts_i = pts_int12[:, :, None]
    pts_j = pts_int12[:, None, :]

    greater = np.sum(pts_j > pts_i, axis=2)  # (K, 4) in {0, 1, 2, 3}
    equal = np.sum(pts_j == pts_i, axis=2)   # (K, 4) in {1, 2, 3, 4}

    # Lookup table of size (4, 5) for average payout given (greater, equal)
    payout_lut = np.zeros((4, 5), dtype=np.float64)
    for g in range(4):
        for eq in range(1, 5 - g):
            payout_lut[g, eq] = float(np.mean(GAME2_RANK_PAYOUTS[g : g + eq]))

    payouts = payout_lut[greater, equal]
    castle_points = pts_int12.astype(np.float64) / 12.0
    return payouts, castle_points
