"""
Metrics, confidence intervals, and order-statistics trackers for Game 1 and Game 2 evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Game1CandidateMetrics:
    """Comprehensive statistical profile of a candidate strategy in Game 1 (1v1)."""

    name: str
    vector: list[int]
    mean_points: float
    std_points: float
    win_rate: float
    tie_rate: float
    loss_rate: float
    p10_points: float
    median_points: float
    p90_points: float
    score_vs_6_ais: float
    percentile_in_field: float


@dataclass(frozen=True)
class Game2CandidateMetrics:
    """Comprehensive statistical profile of a candidate strategy in Game 2 (4-way pods)."""

    name: str
    vector: list[int]
    mean_payout: float
    stderr_payout: float
    ci95_halfwidth: float
    mean_castle_points: float
    solo_first_rate: float
    any_first_rate: float
    top2_rate: float
    zero_payout_rate: float
    prob_ge_7_castle_pts: float
    percentile_in_field: float


def compute_game1_metrics(
    name: str,
    vector: NDArray[np.int64],
    scores_vs_field: NDArray[np.float64],
    score_vs_6_ais: float,
    field_mean_scores: NDArray[np.float64],
) -> Game1CandidateMetrics:
    """
    Compute Game 1 summary statistics for a candidate strategy from its 1D array of match scores.
    """
    mean_pts = float(np.mean(scores_vs_field))
    std_pts = float(np.std(scores_vs_field))
    wins = float(np.mean(scores_vs_field > 7.5))
    ties = float(np.mean(scores_vs_field == 7.5))
    losses = float(np.mean(scores_vs_field < 7.5))
    p10, med, p90 = [float(x) for x in np.percentile(scores_vs_field, [10.0, 50.0, 90.0])]
    pct_rank = 100.0 * float(np.mean(field_mean_scores <= mean_pts))

    return Game1CandidateMetrics(
        name=name,
        vector=vector.tolist(),
        mean_points=mean_pts,
        std_points=std_pts,
        win_rate=wins,
        tie_rate=ties,
        loss_rate=losses,
        p10_points=p10,
        median_points=med,
        p90_points=p90,
        score_vs_6_ais=score_vs_6_ais,
        percentile_in_field=pct_rank,
    )


def compute_game2_metrics(
    name: str,
    vector: NDArray[np.int64],
    hero_payouts: NDArray[np.float64],
    hero_castle_pts: NDArray[np.float64],
    field_mean_payouts: NDArray[np.float64] | None = None,
) -> Game2CandidateMetrics:
    """
    Compute Game 2 summary statistics from K simulated 4-player pods.
    """
    k_pods = hero_payouts.shape[0]
    mean_pay = float(np.mean(hero_payouts))
    std_pay = float(np.std(hero_payouts, ddof=1)) if k_pods > 1 else 0.0
    stderr = std_pay / np.sqrt(k_pods) if k_pods > 0 else 0.0
    ci95 = 1.95996 * stderr

    mean_c_pts = float(np.mean(hero_castle_pts))
    solo_first = float(np.mean(hero_payouts >= 8.999))
    any_first = float(np.mean(hero_payouts >= 3.75))  # 4-way tie for 1st is 3.75, 3-way is 5.0, 2-way is 6.5
    top2 = float(np.mean(hero_payouts >= 4.0))
    zero_rate = float(np.mean(hero_payouts <= 1e-9))
    prob_ge_7 = float(np.mean(hero_castle_pts >= 7.0 - 1e-9))

    if field_mean_payouts is not None and field_mean_payouts.size > 0:
        pct_rank = 100.0 * float(np.mean(field_mean_payouts <= mean_pay))
    else:
        pct_rank = 0.0

    return Game2CandidateMetrics(
        name=name,
        vector=vector.tolist(),
        mean_payout=mean_pay,
        stderr_payout=float(stderr),
        ci95_halfwidth=float(ci95),
        mean_castle_points=mean_c_pts,
        solo_first_rate=solo_first,
        any_first_rate=any_first,
        top2_rate=top2,
        zero_payout_rate=zero_rate,
        prob_ge_7_castle_pts=prob_ge_7,
        percentile_in_field=pct_rank,
    )
