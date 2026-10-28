#!/usr/bin/env python3
"""
CLI entry point to run the full Game 1 and Game 2 benchmark tournament,
exact marginal trade audits, continuous Nash checks, and 3-vs-4 scavenger sensitivity table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Ensure repository root is importable when invoked as `python scripts/run_tournament.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blotto.analysis.order_stats import (
    analyze_7point_pod_lock,
    audit_6ai_ordered_triples,
    compute_continuous_equilibrium_score,
    compute_exact_game1_marginal_trade,
    verify_audit_counterexample_pod,
)
from blotto.sim.engine import TournamentEngine
from blotto.strategies.archetypes import (
    GAME1_CANDIDATES,
    GAME1_LLM_BASELINES,
    GAME1_SHARK_VECTORS,
    GAME2_CANDIDATES,
    GAME2_LLM_BASELINES,
    GAME2_SHARK_VECTORS,
    build_realistic_field_distribution,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 10,000+ match Colonel Blotto Game 1 & Game 2 tournament benchmark."
    )
    parser.add_argument(
        "--pods",
        type=int,
        default=25000,
        help="Number of 4-player Game 2 pods to simulate per candidate (default: 25000).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5000,
        help="Size of the synthetic opponent pool (default: 5000).",
    )
    parser.add_argument(
        "--optimizer-share",
        type=float,
        default=0.50,
        help="Fraction of field modeled as quantitative sharks (default: 0.50).",
    )
    parser.add_argument(
        "--llm-share",
        type=float,
        default=0.38,
        help="Fraction of field modeled as LLM baselines (default: 0.38).",
    )
    parser.add_argument(
        "--all-in-share",
        type=float,
        default=0.00,
        help="Fraction of field submitting [0,0,0,0,100] (default: 0.00 for zero-all-in shark pool).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shared opponent triples (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()

    print("=" * 96)
    print(" BLOTTO SIMULATION ENGINE - PRODUCTION TOURNAMENT & AUDIT BENCHMARK")
    print("=" * 96)
    print(
        f"Configuration: Pods={args.pods:,} | Field Size={args.pool_size:,} | "
        f"Optimizer Share={args.optimizer_share:.0%} | LLM Share={args.llm_share:.0%} | "
        f"All-In Share={args.all_in_share:.0%} | Seed={args.seed}"
    )

    g1_field = build_realistic_field_distribution(
        game=1,
        pool_size=args.pool_size,
        optimizer_share=args.optimizer_share,
        llm_share=args.llm_share,
        all_in_share=args.all_in_share,
        seed=args.seed,
    )
    g2_field = build_realistic_field_distribution(
        game=2,
        pool_size=args.pool_size,
        optimizer_share=args.optimizer_share,
        llm_share=args.llm_share,
        all_in_share=args.all_in_share,
        seed=args.seed + 100,
    )

    g1_candidates = (
        list(GAME1_CANDIDATES.values())
        + list(GAME1_SHARK_VECTORS.values())
        + list(GAME1_LLM_BASELINES.values())
    )
    g2_candidates = (
        list(GAME2_CANDIDATES.values())
        + list(GAME2_SHARK_VECTORS.values())
        + list(GAME2_LLM_BASELINES.values())
    )

    engine = TournamentEngine(
        game1_field=g1_field,
        game2_field=g2_field,
        num_pods=args.pods,
        seed=args.seed,
    )
    report = engine.run_full_tournament(g1_candidates, g2_candidates)

    # 1. Print Game 1 Leaderboard
    print("\n" + "-" * 96)
    print(" GAME 1 LEADERBOARD (1v1 HEAD-TO-HEAD MARGINAL SCORING, MAX = 15.0 PTS)")
    print("-" * 96)
    print(
        f"{'Rank':<5} {'Strategy Name':<44} {'Mean Pts':>9} {'vs 6 AIs':>9} "
        f"{'Win%':>7} {'P10':>6} {'P90':>6} {'Pctle':>7}"
    )
    print("-" * 96)
    for idx, m in enumerate(report.game1_leaderboard, start=1):
        print(
            f"{idx:<5} {m.name:<44} {m.mean_points:>9.4f} {m.score_vs_6_ais:>9.4f} "
            f"{m.win_rate * 100:>6.1f}% {m.p10_points:>6.2f} {m.p90_points:>6.2f} "
            f"{m.percentile_in_field:>6.1f}%"
        )

    print("-" * 96)
    print(
        f"Exact Global DP Best Response vs Field   : {report.game1_global_dp_best[0]} "
        f"-> {report.game1_global_dp_best[1]:.4f} pts"
    )
    print(
        f"Exact Constrained DP (Castle 5 == 0)     : {report.game1_zero_c5_dp_best[0]} "
        f"-> {report.game1_zero_c5_dp_best[1]:.4f} pts "
        f"(Opportunity cost of forcing C5=0: {report.game1_c5_zero_opportunity_cost:.4f} pts)"
    )

    # 2. Print Game 2 Leaderboard
    print("\n" + "-" * 96)
    print(
        f" GAME 2 LEADERBOARD (4-PLAYER RANK PAYOUTS [9,4,2,0], {args.pods:,} SHARED PODS, MEAN = 3.750)"
    )
    print("-" * 96)
    print(
        f"{'Rank':<5} {'Strategy Name':<44} {'Mean Pay':>9} {'95% CI':>8} "
        f"{'Solo 1st':>9} {'C-Pts':>7} {'P(>=7pt)':>9} {'Pctle':>7}"
    )
    print("-" * 96)
    for idx, m in enumerate(report.game2_leaderboard, start=1):
        print(
            f"{idx:<5} {m.name:<44} {m.mean_payout:>9.4f} {m.ci95_halfwidth:>8.4f} "
            f"{m.solo_first_rate * 100:>8.2f}% {m.mean_castle_points:>7.2f} "
            f"{m.prob_ge_7_castle_pts * 100:>8.2f}% {m.percentile_in_field:>6.1f}%"
        )

    # 3. Print Exact Analytical & Audit Verifications
    print("\n" + "-" * 96)
    print(" ANALYTICAL & AUDIT VERIFICATIONS")
    print("-" * 96)
    llm6_mat = np.stack([s.vector for s in GAME1_LLM_BASELINES.values()], axis=0)
    for donor_idx, label in [(0, "C1 (10->9)"), (1, "C2 (17->16)"), (2, "C3 (25->24)"), (3, "C4 (45->44)")]:
        audit = compute_exact_game1_marginal_trade([10, 17, 25, 45, 3], donor_idx, 4, llm6_mat)
        print(
            f"Marginal Trade [10,17,25,45,3] -> {str(audit.modified_vector):<18} "
            f"(Donor {label:<12}): Formula Delta = {audit.formula_expected_delta:+.4f} pts vs 6 AIs "
            f"(Empirical = {audit.empirical_expected_delta:+.4f})"
        )

    cont_score, cont_bounds = compute_continuous_equilibrium_score([10, 17, 25, 45, 3])
    print(
        f"Continuous Uniform-Marginal Nash Check   : [10,17,25,45,3] vs U[0, {[round(x, 2) for x in cont_bounds]}] "
        f"= {cont_score:.4f} pts"
    )

    f2_mean, f2_firsts, f2_total = audit_6ai_ordered_triples([12, 13, 50, 25, 0])
    print(
        f"Finding 2 Check ([12,13,50,25,0] vs 6 AIs): Mean Payout = {f2_mean:.4f} "
        f"({f2_firsts}/{f2_total} solo 1st place finishes)"
    )

    ce_results = verify_audit_counterexample_pod()
    print("Audit Counterexample Pod ([1,1,34,63,1], [2,2,2,43,51], [1,1,1,43,54]):")
    for h_label, (cpts, pay) in ce_results.items():
        print(f"  Hero {h_label:<16} -> Castle Points: {cpts:.1f}, Rank Payout: {pay:.1f}")

    lock_info = analyze_7point_pod_lock([13, 21, 3, 60, 3], g2_field)
    print(
        f"Order-Stats 3-Opp Solo Win Probs [13,21,3,60,3]: "
        f"{[round(p, 3) for p in lock_info.three_opp_solo_win_probs]} "
        f"(Independent E[Castle Pts] = {lock_info.expected_independent_points:.2f})"
    )

    elapsed = time.perf_counter() - t0
    print("=" * 96)
    print(f"Completed full tournament and audit in {elapsed:.3f} seconds.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
