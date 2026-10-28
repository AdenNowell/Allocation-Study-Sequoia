"""
High-performance batch/multiprocessing tournament engine for evaluating candidate strategies
against synthetic and empirical fields over 10,000+ shared-triple pods.

Key Design Features:
- Shared Opponent Triples (Common Random Numbers): Every Game 2 candidate is evaluated against
  the exact same sequence of K random opponent triples (o1, o2, o3), drastically reducing
  paired comparison variance (paired 95% half-widths ~0.006 to 0.010 at K=100,000).
- Chunked Vectorized Execution + ProcessPoolExecutor support for multi-core scaling.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from blotto.core.rules import validate_population_matrix
from blotto.core.scoring import (
    exact_game1_dp_best_response,
    score_1v1_matrix,
    score_4way_pods_batch,
)
from blotto.sim.metrics import (
    Game1CandidateMetrics,
    Game2CandidateMetrics,
    compute_game1_metrics,
    compute_game2_metrics,
)
from blotto.strategies.archetypes import GAME1_LLM_BASELINES, GAME2_LLM_BASELINES
from blotto.strategies.base import FixedStrategy


def _eval_candidate_game2_worker(
    args: tuple[str, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
) -> tuple[str, NDArray[np.float64], NDArray[np.float64]]:
    """Top-level worker function for ProcessPoolExecutor Game 2 evaluation."""
    name, hero_vec, field_mat, triples = args
    k_pods = triples.shape[0]
    chunk_size = 25000
    hero_payouts = np.empty(k_pods, dtype=np.float64)
    hero_cpts = np.empty(k_pods, dtype=np.float64)

    for start in range(0, k_pods, chunk_size):
        end = min(start + chunk_size, k_pods)
        batch_len = end - start
        pods = np.empty((batch_len, 4, 5), dtype=np.int64)
        pods[:, 0, :] = hero_vec
        pods[:, 1, :] = field_mat[triples[start:end, 0]]
        pods[:, 2, :] = field_mat[triples[start:end, 1]]
        pods[:, 3, :] = field_mat[triples[start:end, 2]]
        pay, cpts = score_4way_pods_batch(pods)
        hero_payouts[start:end] = pay[:, 0]
        hero_cpts[start:end] = cpts[:, 0]

    return name, hero_payouts, hero_cpts


@dataclass
class TournamentReport:
    """Container for complete Game 1 and Game 2 benchmark outputs."""

    game1_leaderboard: list[Game1CandidateMetrics]
    game2_leaderboard: list[Game2CandidateMetrics]
    game1_global_dp_best: tuple[list[int], float]
    game1_zero_c5_dp_best: tuple[list[int], float]
    game1_c5_zero_opportunity_cost: float
    num_pods: int
    field_size: int


class TournamentEngine:
    """
    Orchestrates full-scale Game 1 and Game 2 tournaments over shared opponent pools.
    """

    def __init__(
        self,
        game1_field: NDArray[np.int64],
        game2_field: NDArray[np.int64],
        num_pods: int = 25000,
        seed: int = 42,
        max_workers: int | None = None,
    ) -> None:
        self.game1_field = validate_population_matrix(game1_field)
        self.game2_field = validate_population_matrix(game2_field)
        if num_pods < 100:
            raise ValueError("num_pods must be at least 100.")
        self.num_pods = int(num_pods)
        self.seed = int(seed)
        self.max_workers = max_workers

    def run_game1_benchmark(
        self,
        candidates: Sequence[FixedStrategy],
    ) -> tuple[list[Game1CandidateMetrics], tuple[list[int], float], tuple[list[int], float]]:
        """
        Evaluate all Game 1 candidates against `self.game1_field` and against the 6 exact Game 1 LLMs,
        plus compute the exact unrestricted and zero-on-C5 DP optima.
        """
        cand_vecs = np.stack([c.vector for c in candidates], axis=0)
        pairwise_vs_field = score_1v1_matrix(cand_vecs, self.game1_field)  # (M, N)

        llm6_vecs = np.stack([s.vector for s in GAME1_LLM_BASELINES.values()], axis=0)
        means_vs_llm6 = score_1v1_matrix(cand_vecs, llm6_vecs, field_weights=np.ones(6))

        # Compute field self-scores for percentile ranking (sample up to 1000 field members for speed)
        rng = np.random.default_rng(self.seed)
        sample_idx = rng.choice(
            self.game1_field.shape[0],
            size=min(1000, self.game1_field.shape[0]),
            replace=False,
        )
        field_sample = self.game1_field[sample_idx]
        field_means = score_1v1_matrix(
            field_sample,
            self.game1_field,
            field_weights=np.ones(self.game1_field.shape[0]),
        )

        metrics_list: list[Game1CandidateMetrics] = []
        for idx, cand in enumerate(candidates):
            metrics_list.append(
                compute_game1_metrics(
                    name=cand.name,
                    vector=cand.vector,
                    scores_vs_field=pairwise_vs_field[idx],
                    score_vs_6_ais=float(means_vs_llm6[idx]),
                    field_mean_scores=field_means,
                )
            )

        metrics_list.sort(key=lambda m: m.mean_points, reverse=True)

        dp_unrestricted_vec, dp_unrestricted_val = exact_game1_dp_best_response(self.game1_field)
        dp_zero5_vec, dp_zero5_val = exact_game1_dp_best_response(
            self.game1_field, fixed_castle_bids={4: 0}
        )
        return (
            metrics_list,
            (dp_unrestricted_vec.tolist(), dp_unrestricted_val),
            (dp_zero5_vec.tolist(), dp_zero5_val),
        )

    def run_game2_benchmark(
        self,
        candidates: Sequence[FixedStrategy],
    ) -> list[Game2CandidateMetrics]:
        """
        Evaluate all Game 2 candidates across `self.num_pods` shared opponent triples.
        """
        rng = np.random.default_rng(self.seed)
        n_field = self.game2_field.shape[0]
        triples = rng.integers(0, n_field, size=(self.num_pods, 3), dtype=np.int64)

        # Estimate empirical distribution of field member payouts for percentile calculation
        n_field_probe = min(200, n_field)
        probe_indices = rng.choice(n_field, size=n_field_probe, replace=False)
        probe_triples = triples[: min(2500, self.num_pods)]
        field_mean_payouts = np.empty(n_field_probe, dtype=np.float64)
        for i, f_idx in enumerate(probe_indices):
            _, pay, _ = _eval_candidate_game2_worker(
                ("field_probe", self.game2_field[f_idx], self.game2_field, probe_triples)
            )
            field_mean_payouts[i] = float(np.mean(pay))

        tasks = [
            (cand.name, cand.vector, self.game2_field, triples)
            for cand in candidates
        ]

        results_map: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        if self.max_workers is not None and self.max_workers > 1 and len(tasks) > 2:
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                for name, pay, cpts in pool.map(_eval_candidate_game2_worker, tasks):
                    results_map[name] = (pay, cpts)
        else:
            for t in tasks:
                name, pay, cpts = _eval_candidate_game2_worker(t)
                results_map[name] = (pay, cpts)

        metrics_list: list[Game2CandidateMetrics] = []
        for cand in candidates:
            pay, cpts = results_map[cand.name]
            metrics_list.append(
                compute_game2_metrics(
                    name=cand.name,
                    vector=cand.vector,
                    hero_payouts=pay,
                    hero_castle_pts=cpts,
                    field_mean_payouts=field_mean_payouts,
                )
            )

        metrics_list.sort(key=lambda m: m.mean_payout, reverse=True)
        return metrics_list

    def run_full_tournament(
        self,
        game1_candidates: Sequence[FixedStrategy],
        game2_candidates: Sequence[FixedStrategy],
    ) -> TournamentReport:
        """Execute both Game 1 and Game 2 benchmarks and return a unified TournamentReport."""
        g1_board, dp_unrest, dp_zero5 = self.run_game1_benchmark(game1_candidates)
        g2_board = self.run_game2_benchmark(game2_candidates)
        opp_cost = dp_unrest[1] - dp_zero5[1]

        return TournamentReport(
            game1_leaderboard=g1_board,
            game2_leaderboard=g2_board,
            game1_global_dp_best=dp_unrest,
            game1_zero_c5_dp_best=dp_zero5,
            game1_c5_zero_opportunity_cost=opp_cost,
            num_pods=self.num_pods,
            field_size=self.game2_field.shape[0],
        )
