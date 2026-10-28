"""Unit tests for 1v1 scoring, 4-way pod rank-payout tie-splitting, exact DP, and audit claims."""

from __future__ import annotations

import numpy as np
import pytest

from blotto.analysis.order_stats import (
    audit_6ai_ordered_triples,
    compute_continuous_equilibrium_score,
    compute_exact_game1_marginal_trade,
    verify_audit_counterexample_pod,
)
from blotto.core.scoring import (
    exact_game1_dp_best_response,
    score_1v1_matrix,
    score_1v1_pair,
    score_4way_pod,
    score_4way_pods_batch,
)
from blotto.strategies.archetypes import GAME1_LLM_BASELINES


def test_1v1_pair_and_conservation() -> None:
    # [10, 17, 25, 45, 3] vs [5, 3, 3, 38, 51] -> Hero wins C1..C4 (10 pts), Opp wins C5 (5 pts)
    h_pts, o_pts = score_1v1_pair([10, 17, 25, 45, 3], [5, 3, 3, 38, 51])
    assert h_pts == pytest.approx(10.0)
    assert o_pts == pytest.approx(5.0)
    assert h_pts + o_pts == pytest.approx(15.0)

    # Identical vectors tie all castles for 7.5 each
    t1, t2 = score_1v1_pair([10, 16, 25, 45, 4], [10, 16, 25, 45, 4])
    assert t1 == pytest.approx(7.5)
    assert t2 == pytest.approx(7.5)


def test_game1_scores_against_6_ai_baselines() -> None:
    llm6 = np.stack([s.vector for s in GAME1_LLM_BASELINES.values()], axis=0)
    candidates = np.array(
        [
            [10, 17, 25, 45, 3],  # 10.0000
            [9, 17, 25, 45, 4],   # 9.9167 (ties Gemini's 9 on C1)
            [10, 16, 25, 45, 4],  # 9.8333 (ties Fable's 16 on C2)
            [10, 17, 24, 45, 4],  # 9.7500 (ties Gemini's 24 on C3)
            [10, 17, 25, 44, 4],  # 10.0000
            [5, 3, 3, 38, 51],    # 11.3333
            [5, 3, 3, 42, 47],    # 11.3333
        ],
        dtype=np.int64,
    )
    means = score_1v1_matrix(candidates, llm6, field_weights=np.ones(6))
    assert means[0] == pytest.approx(10.0)
    assert means[1] == pytest.approx(9.9166667, rel=1e-5)
    assert means[2] == pytest.approx(9.8333333, rel=1e-5)
    assert means[3] == pytest.approx(9.75)
    assert means[4] == pytest.approx(10.0)
    assert means[5] == pytest.approx(11.3333333, rel=1e-5)
    assert means[6] == pytest.approx(11.3333333, rel=1e-5)


def test_exact_marginal_trade_formula() -> None:
    llm6 = np.stack([s.vector for s in GAME1_LLM_BASELINES.values()], axis=0)
    # [10, 17, 25, 45, 3] -> [10, 17, 24, 45, 4] costs exactly 0.25 average points vs the 6 AIs
    audit = compute_exact_game1_marginal_trade([10, 17, 25, 45, 3], 2, 4, llm6)
    assert audit.formula_expected_delta == pytest.approx(-0.25)
    assert audit.empirical_expected_delta == pytest.approx(-0.25)


def test_4way_pod_ties_and_payout_conservation() -> None:
    # 1. Four identical allocations must split 9+4+2+0 = 15 equally (3.75 each)
    v = [20, 20, 20, 20, 20]
    pay_all_tie = score_4way_pod(v, v, v, v)
    assert np.allclose(pay_all_tie, [3.75, 3.75, 3.75, 3.75])

    # 2. Finding 4 zero-payout scenario for [0, 30, 0, 70, 0] vs [0, 29, 0, 71, 0] and two [0, 0, 0, 0, 100]
    # Hero wins C2 (2 pts) + 1/4 of C1 (0.25) + 1/4 of C3 (0.75) = 3.0 pts.
    # Opp 1 wins C4 (4 pts) + 1.0 tied = 5.0 pts (1st place -> 9.0 payout).
    # Opp 2 & 3 split C5 (2.5 pts each) + 1.0 tied = 3.5 pts each (tie 2nd/3rd -> 3.0 payout each).
    # Hero finishes 4th with 3.0 pts -> 0.0 payout!
    pay_f4 = score_4way_pod(
        [0, 30, 0, 70, 0],
        [0, 29, 0, 71, 0],
        [0, 0, 0, 0, 100],
        [0, 0, 0, 0, 100],
    )
    assert np.allclose(pay_f4, [0.0, 9.0, 3.0, 3.0])
    assert float(np.sum(pay_f4)) == pytest.approx(15.0)


def test_batch_vs_scalar_4way_equivalence() -> None:
    rng = np.random.default_rng(123)
    # Generate 200 random legal pods including heavy ties
    pool = np.array(
        [
            [13, 21, 3, 60, 3],
            [11, 21, 4, 60, 4],
            [0, 30, 0, 70, 0],
            [0, 0, 0, 0, 100],
            [12, 0, 36, 52, 0],
            [1, 1, 34, 63, 1],
            [2, 2, 2, 43, 51],
        ],
        dtype=np.int64,
    )
    idx = rng.integers(0, len(pool), size=(200, 4))
    pods = pool[idx]

    batch_pay, _ = score_4way_pods_batch(pods)
    for k in range(200):
        scalar_pay = score_4way_pod(pods[k, 0], pods[k, 1], pods[k, 2], pods[k, 3])
        assert np.allclose(batch_pay[k], scalar_pay)
        assert float(np.sum(batch_pay[k])) == pytest.approx(15.0)


def test_finding2_and_audit_counterexample() -> None:
    # Finding 2: [12, 13, 50, 25, 0] earns payout 9.0 in all 216 ordered triples of the 6 AIs
    mean_pay, firsts, total = audit_6ai_ordered_triples([12, 13, 50, 25, 0])
    assert total == 216
    assert firsts == 216
    assert mean_pay == pytest.approx(9.0)

    # Audit counterexample pod: hero earns 3.0 castle points and 2.0 rank payout
    ce = verify_audit_counterexample_pod()
    assert ce["[13,20,4,59,4]"] == (pytest.approx(3.0), pytest.approx(2.0))
    assert ce["[13,21,3,60,3]"] == (pytest.approx(3.0), pytest.approx(2.0))


def test_continuous_equilibrium_and_dp() -> None:
    score, _ = compute_continuous_equilibrium_score([10, 17, 25, 45, 3])
    assert score == pytest.approx(7.5)

    llm6 = np.stack([s.vector for s in GAME1_LLM_BASELINES.values()], axis=0)
    best_vec, best_score = exact_game1_dp_best_response(llm6)
    assert int(np.sum(best_vec)) == 100
    assert best_score >= 11.3333333 - 1e-6
