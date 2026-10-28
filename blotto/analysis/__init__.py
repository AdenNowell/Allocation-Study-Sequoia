"""Closed-form order statistics, marginal trade equations, and pod counterexample audits."""

from blotto.analysis.order_stats import (
    MarginalTradeAudit,
    PodLockAnalysis,
    analyze_7point_pod_lock,
    audit_6ai_ordered_triples,
    compute_continuous_equilibrium_score,
    compute_exact_game1_marginal_trade,
    verify_audit_counterexample_pod,
)

__all__ = [
    "MarginalTradeAudit",
    "PodLockAnalysis",
    "compute_exact_game1_marginal_trade",
    "compute_continuous_equilibrium_score",
    "audit_6ai_ordered_triples",
    "verify_audit_counterexample_pod",
    "analyze_7point_pod_lock",
]
