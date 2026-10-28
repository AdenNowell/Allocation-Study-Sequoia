"""
Closed-form and empirical order-statistics analysis for the Sequoia Castle games:
1. Exact marginal trade formula in Game 1:
   Delta E = (c_gain / 2) * [P(Y_{c_gain} == b_gain - 1) + P(Y_{c_gain} == b_gain)]
           - (c_donor / 2) * [P(Y_{c_donor} == b_donor) + P(Y_{c_donor} == b_donor + 1)]
2. Algebraic check of the continuous uniform-marginal Nash equilibrium:
   Upper bounds U_j = 200 * v_j / 15 = [13.33, 26.67, 40.00, 53.33, 66.67].
3. Exhaustive 216-ordered-triple verification of Finding 2 (`[12, 13, 50, 25, 0]` vs the 6 AIs).
4. Verification of the Game 2 audit counterexample pod (`[13, 20, 4, 59, 4]` and `[13, 21, 3, 60, 3]`
   against `[1, 1, 34, 63, 1]`, `[2, 2, 2, 43, 51]`, `[1, 1, 1, 43, 54]`).
5. Order statistics of 3-opponent maximums F_{M_c}(x) = (F_c(x))^3 for the 7-point pod lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from blotto.core.rules import (
    CASTLE_VALUES,
    NUM_CASTLES,
    TOTAL_SOLDIERS,
    validate_allocation,
    validate_population_matrix,
)
from blotto.core.scoring import score_1v1_matrix, score_4way_pod, score_4way_pods_batch
from blotto.strategies.archetypes import GAME2_LLM_BASELINES


@dataclass(frozen=True)
class MarginalTradeAudit:
    """Result of shifting 1 soldier between two castles in Game 1."""

    base_vector: list[int]
    modified_vector: list[int]
    donor_castle_1idx: int
    recipient_castle_1idx: int
    prob_recipient_tie_win_window: float
    prob_donor_win_tie_window: float
    formula_expected_delta: float
    empirical_expected_delta: float


@dataclass(frozen=True)
class PodLockAnalysis:
    """Order-statistics breakdown for a candidate allocation in 4-player Game 2 pods."""

    vector: list[int]
    single_opp_win_probs: list[float]
    three_opp_solo_win_probs: list[float]
    expected_independent_points: float


def compute_exact_game1_marginal_trade(
    base_alloc: Sequence[int] | NDArray[np.integer],
    donor_castle_0idx: int,
    recipient_castle_0idx: int,
    field: NDArray[np.integer] | Sequence[Sequence[int]],
    field_weights: NDArray[np.floating] | Sequence[float] | None = None,
) -> MarginalTradeAudit:
    """
    Verify the exact closed-form marginal trade equation for shifting 1 soldier in Game 1.

    For moving 1 troop from castle `d` (from b_d to b_d - 1) to castle `r` (from b_r to b_r + 1):
        Delta E[S] = 0.5 * v_r * (P(Y_r == b_r) + P(Y_r == b_r + 1))
                   - 0.5 * v_d * (P(Y_d == b_d - 1) + P(Y_d == b_d))
    """
    base = validate_allocation(base_alloc)
    if donor_castle_0idx == recipient_castle_0idx:
        raise ValueError("donor and recipient castles must be distinct.")
    if base[donor_castle_0idx] <= 0:
        raise ValueError("Donor castle must have at least 1 soldier.")

    mod = base.copy()
    mod[donor_castle_0idx] -= 1
    mod[recipient_castle_0idx] += 1

    field_mat = validate_population_matrix(field)
    n_opp = field_mat.shape[0]
    if field_weights is None:
        w = np.full(n_opp, 1.0 / n_opp, dtype=np.float64)
    else:
        w_raw = np.asarray(field_weights, dtype=np.float64)
        w = w_raw / np.sum(w_raw)

    b_d = int(base[donor_castle_0idx])
    b_r = int(base[recipient_castle_0idx])

    prob_r_window = float(
        np.sum(w[(field_mat[:, recipient_castle_0idx] == b_r) | (field_mat[:, recipient_castle_0idx] == b_r + 1)])
    )
    prob_d_window = float(
        np.sum(w[(field_mat[:, donor_castle_0idx] == b_d - 1) | (field_mat[:, donor_castle_0idx] == b_d)])
    )

    v_r = float(CASTLE_VALUES[recipient_castle_0idx])
    v_d = float(CASTLE_VALUES[donor_castle_0idx])
    formula_delta = 0.5 * v_r * prob_r_window - 0.5 * v_d * prob_d_window

    scores = score_1v1_matrix(np.stack([base, mod], axis=0), field_mat, field_weights=w)
    empirical_delta = float(scores[1] - scores[0])

    return MarginalTradeAudit(
        base_vector=base.tolist(),
        modified_vector=mod.tolist(),
        donor_castle_1idx=donor_castle_0idx + 1,
        recipient_castle_1idx=recipient_castle_0idx + 1,
        prob_recipient_tie_win_window=prob_r_window,
        prob_donor_win_tie_window=prob_d_window,
        formula_expected_delta=formula_delta,
        empirical_expected_delta=empirical_delta,
    )


def compute_continuous_equilibrium_score(
    allocation: Sequence[int] | NDArray[np.integer],
) -> tuple[float, list[float]]:
    """
    Evaluate a discrete allocation against the continuous Gross-Wagner uniform marginals:
        Y_j ~ Uniform[0, 200 * v_j / 15]
    Upper bounds: [13.3333, 26.6667, 40.0, 53.3333, 66.6667].
    For any allocation where x_j <= U_j for all j, the theoretical expected score is identically 7.50.
    """
    vec = validate_allocation(allocation)
    upper_bounds = (200.0 * CASTLE_VALUES) / 15.0
    win_probs = np.minimum(1.0, vec.astype(np.float64) / upper_bounds)
    expected_score = float(np.dot(CASTLE_VALUES, win_probs))
    return expected_score, upper_bounds.tolist()


def audit_6ai_ordered_triples(
    candidate: Sequence[int] | NDArray[np.integer] = (12, 13, 50, 25, 0),
) -> tuple[float, int, int]:
    """
    Exhaustively evaluate `candidate` across all 6^3 = 216 ordered triples of the 6 collected
    Game 2 AI outputs (Finding 2 check: `[12, 13, 50, 25, 0]` earns payout 9.0 in all 216 triples).

    Returns
    -------
    tuple[float, int, int]
        (mean_payout, num_solo_firsts, total_triples=216)
    """
    hero = validate_allocation(candidate)
    ai_vecs = [s.vector for s in GAME2_LLM_BASELINES.values()]
    pods = np.empty((216, 4, 5), dtype=np.int64)
    idx = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                pods[idx, 0] = hero
                pods[idx, 1] = ai_vecs[i]
                pods[idx, 2] = ai_vecs[j]
                pods[idx, 3] = ai_vecs[k]
                idx += 1

    payouts, _ = score_4way_pods_batch(pods)
    hero_pay = payouts[:, 0]
    solo_firsts = int(np.sum(hero_pay >= 8.999))
    return float(np.mean(hero_pay)), solo_firsts, 216


def verify_audit_counterexample_pod() -> dict[str, tuple[float, float]]:
    """
    Reproduce the exact Game 2 counterexample pod from the audit:
    Opponents:
      - [1, 1, 34, 63, 1] (C3+C4 concentration)
      - [2, 2, 2, 43, 51] (C4+C5 concentration)
      - [1, 1, 1, 43, 54] (C4+C5 concentration)
    Demonstrates that both [13, 20, 4, 59, 4] and [13, 21, 3, 60, 3] score 3 castle points
    (from C1 and C2) against opponent totals (7, 0, 5), finishing 3rd for 2.0 rank payout.
    """
    o1 = [1, 1, 34, 63, 1]
    o2 = [2, 2, 2, 43, 51]
    o3 = [1, 1, 1, 43, 54]

    heroes = {
        "[13,20,4,59,4]": [13, 20, 4, 59, 4],
        "[13,21,3,60,3]": [13, 21, 3, 60, 3],
        "[11,21,4,60,4]": [11, 21, 4, 60, 4],
    }
    out: dict[str, tuple[float, float]] = {}
    for label, h in heroes.items():
        pods = np.array([[h, o1, o2, o3]], dtype=np.int64)
        pay, cpts = score_4way_pods_batch(pods)
        out[label] = (float(cpts[0, 0]), float(pay[0, 0]))
    return out


def analyze_7point_pod_lock(
    candidate: Sequence[int] | NDArray[np.integer],
    field: NDArray[np.integer] | Sequence[Sequence[int]],
) -> PodLockAnalysis:
    """
    Compute first-order (1v1) and third-order (max of 3 i.i.d. opponents) castle win probabilities:
        P(max(Y_{1,c}, Y_{2,c}, Y_{3,c}) < x_c) = [P(Y_c < x_c)]^3
    """
    vec = validate_allocation(candidate)
    field_mat = validate_population_matrix(field)

    p1_list: list[float] = []
    p3_list: list[float] = []
    for c in range(NUM_CASTLES):
        p_strict = float(np.mean(field_mat[:, c] < vec[c]))
        p1_list.append(p_strict)
        p3_list.append(p_strict**3)

    exp_pts = float(np.dot(CASTLE_VALUES, np.array(p3_list)))
    return PodLockAnalysis(
        vector=vec.tolist(),
        single_opp_win_probs=p1_list,
        three_opp_solo_win_probs=p3_list,
        expected_independent_points=exp_pts,
    )
