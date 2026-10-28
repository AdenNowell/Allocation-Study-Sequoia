"""
Exact hardcoded strategy vectors formulated and audited during the Sequoia Castle review.

Includes:
1. Game 1 & Game 2 Final Recommended Submissions (3-bid and 4-bid scavenger variants).
2. Collected LLM Baselines (Gemini, 6.1 Sol, 6 Astera Max, GPT-6 Luna, GPT-5.6 Luna, Fable 5.1 Max).
3. Quantitative "Shark" Archetypes (C4+C5 heavy snipers, C3+C5 snipers, C3+C4 concentrators,
   and heavy agent benchmarks).
4. Boundary & Heuristic Benchmarks (Proportional, Equal, All-In C5, AI-Table Exploiter).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from blotto.strategies.base import FixedStrategy
from blotto.strategies.generative import IntegerTransferPerturber, UniformSimplexGenerator


# ============================================================================
# 1. COLLECTED LLM BASELINES (EXACT TABLE FROM AUDIT)
# ============================================================================

GAME1_LLM_BASELINES: dict[str, FixedStrategy] = {
    "LLM_G1_Gemini": FixedStrategy(
        "LLM_G1_Gemini", [9, 0, 24, 31, 36], category="llm_g1", description="Gemini Game 1 baseline"
    ),
    "LLM_G1_6.1_Sol": FixedStrategy(
        "LLM_G1_6.1_Sol", [4, 14, 0, 36, 46], category="llm_g1", description="6.1 Sol Game 1 baseline"
    ),
    "LLM_G1_6_Astera_Max": FixedStrategy(
        "LLM_G1_6_Astera_Max", [2, 0, 22, 33, 43], category="llm_g1", description="6 Astera Max Game 1 baseline"
    ),
    "LLM_G1_GPT6_Luna": FixedStrategy(
        "LLM_G1_GPT6_Luna", [4, 12, 20, 28, 36], category="llm_g1", description="GPT-6 Luna Game 1 baseline"
    ),
    "LLM_G1_GPT5.6_Luna": FixedStrategy(
        "LLM_G1_GPT5.6_Luna", [7, 13, 18, 27, 35], category="llm_g1", description="GPT-5.6 Luna Game 1 baseline"
    ),
    "LLM_G1_Fable_5.1_Max": FixedStrategy(
        "LLM_G1_Fable_5.1_Max", [2, 16, 1, 37, 44], category="llm_g1", description="Fable 5.1 Max Game 1 baseline"
    ),
}

GAME2_LLM_BASELINES: dict[str, FixedStrategy] = {
    "LLM_G2_Gemini": FixedStrategy(
        "LLM_G2_Gemini", [12, 0, 36, 52, 0], category="llm_g2", description="Gemini Game 2 baseline (C1+C3+C4)"
    ),
    "LLM_G2_6.1_Sol": FixedStrategy(
        "LLM_G2_6.1_Sol", [0, 0, 39, 0, 61], category="llm_g2", description="6.1 Sol Game 2 baseline (C3+C5)"
    ),
    "LLM_G2_6_Astera_Max": FixedStrategy(
        "LLM_G2_6_Astera_Max", [1, 12, 0, 0, 87], category="llm_g2", description="6 Astera Max Game 2 baseline (C2+C5)"
    ),
    "LLM_G2_GPT6_Luna": FixedStrategy(
        "LLM_G2_GPT6_Luna", [3, 8, 14, 25, 50], category="llm_g2", description="GPT-6 Luna Game 2 baseline"
    ),
    "LLM_G2_GPT5.6_Luna": FixedStrategy(
        "LLM_G2_GPT5.6_Luna", [5, 10, 15, 25, 45], category="llm_g2", description="GPT-5.6 Luna Game 2 baseline"
    ),
    "LLM_G2_Fable_5.1_Max": FixedStrategy(
        "LLM_G2_Fable_5.1_Max", [0, 0, 49, 51, 0], category="llm_g2", description="Fable 5.1 Max Game 2 baseline (C3+C4)"
    ),
}


# ============================================================================
# 2. OUR AUDITED CANDIDATES & SUBMISSIONS (GAME 1 & GAME 2)
# ============================================================================

GAME1_CANDIDATES: dict[str, FixedStrategy] = {
    "Hero_G1_C5Sac_3Bid [10,17,25,45,3]": FixedStrategy(
        "Hero_G1_C5Sac_3Bid [10,17,25,45,3]",
        [10, 17, 25, 45, 3],
        category="hero_g1",
        description="Primary Game 1 C5-sacrifice: sweeps C1..C4 (10 pts) vs all 6 LLMs, 3 on C5 beats 0/1/2 scavengers",
    ),
    "Hero_G1_C5Sac_4Bid_C2Donor [10,16,25,45,4]": FixedStrategy(
        "Hero_G1_C5Sac_4Bid_C2Donor [10,16,25,45,4]",
        [10, 16, 25, 45, 4],
        category="hero_g1",
        description="Submitted Game 1 variant: funds 4 on C5 from C2 (17->16), keeps 25 on C3 and 45 on C4",
    ),
    "Hero_G1_C5Sac_4Bid_C4Donor [10,17,25,44,4]": FixedStrategy(
        "Hero_G1_C5Sac_4Bid_C4Donor [10,17,25,44,4]",
        [10, 17, 25, 44, 4],
        category="hero_g1",
        description="Zero-LLM-loss 4-bid variant: pulls donor troop from C4 (45->44), scores 10.0 vs all 6 LLMs",
    ),
    "Hero_G1_C5Sac_4Bid_C1Donor [9,17,25,45,4]": FixedStrategy(
        "Hero_G1_C5Sac_4Bid_C1Donor [9,17,25,45,4]",
        [9, 17, 25, 45, 4],
        category="hero_g1",
        description="Screenshot 4-bid variant: pulls donor troop from C1 (10->9), scores 9.9167 vs 6 LLMs",
    ),
    "Hero_G1_C5Sac_4Bid_C3Donor [10,17,24,45,4]": FixedStrategy(
        "Hero_G1_C5Sac_4Bid_C3Donor [10,17,24,45,4]",
        [10, 17, 24, 45, 4],
        category="hero_g1",
        description="Suboptimal C3-donor variant: ties Gemini's 24 on C3 (9.75 vs 6 LLMs)",
    ),
    "Hero_G1_C4C5_Refined [11,1,1,39,48]": FixedStrategy(
        "Hero_G1_C4C5_Refined [11,1,1,39,48]",
        [11, 1, 1, 39, 48],
        category="hero_g1",
        description="Refined C1+C4+C5 heavy candidate (10 pt core + 1-soldier scavengers on C2/C3)",
    ),
    "Hero_G1_Early_Probe [10,1,4,38,47]": FixedStrategy(
        "Hero_G1_Early_Probe [10,1,4,38,47]",
        [10, 1, 4, 38, 47],
        category="hero_g1",
        description="Earlier scenario-dependent Game 1 candidate",
    ),
}

GAME2_CANDIDATES: dict[str, FixedStrategy] = {
    "Hero_G2_Fortress_3Bid [13,21,3,60,3]": FixedStrategy(
        "Hero_G2_Fortress_3Bid [13,21,3,60,3]",
        [13, 21, 3, 60, 3],
        category="hero_g2",
        description="Primary Game 2 7-point fortress (C1+C2+C4 = 7 pts) with 3-soldier scavengers on C3 and C5",
    ),
    "Hero_G2_Fortress_4Bid_C1Donor [11,21,4,60,4]": FixedStrategy(
        "Hero_G2_Fortress_4Bid_C1Donor [11,21,4,60,4]",
        [11, 21, 4, 60, 4],
        category="hero_g2",
        description="UI-screen 4-bid Game 2 allocation: funds 4s on C3/C5 from C1 (13->11) while holding 60 on C4",
    ),
    "Hero_G2_Fortress_61Peak [13,20,3,61,3]": FixedStrategy(
        "Hero_G2_Fortress_61Peak [13,20,3,61,3]",
        [13, 20, 3, 61, 3],
        category="hero_g2",
        description="Asymmetric peak-shifted fortress: steps to 61 on C4 to beat modal 60 bidders",
    ),
    "Hero_G2_Fortress_2Bid [13,22,2,61,2]": FixedStrategy(
        "Hero_G2_Fortress_2Bid [13,22,2,61,2]",
        [13, 22, 2, 61, 2],
        category="hero_g2",
        description="Initial 2-soldier scavenger 7-point fortress",
    ),
    "Hero_G2_Fortress_4Bid_C2C4Donor [13,20,4,59,4]": FixedStrategy(
        "Hero_G2_Fortress_4Bid_C2C4Donor [13,20,4,59,4]",
        [13, 20, 4, 59, 4],
        category="hero_g2",
        description="Audited 59-on-C4 4-bid variant",
    ),
    "Hero_G2_Early_30_70 [0,30,0,70,0]": FixedStrategy(
        "Hero_G2_Early_30_70 [0,30,0,70,0]",
        [0, 30, 0, 70, 0],
        category="hero_g2",
        description="Earlier 2-castle Game 2 candidate (capped at 6 pts, vulnerable to 0-bid ties)",
    ),
}


# ============================================================================
# 3. QUANTITATIVE "SHARK" ARCHETYPES & COUNTEREXAMPLES
# ============================================================================

GAME1_SHARK_VECTORS: dict[str, FixedStrategy] = {
    "Shark_G1_RegretMin_51 [5,3,3,38,51]": FixedStrategy(
        "Shark_G1_RegretMin_51 [5,3,3,38,51]",
        [5, 3, 3, 38, 51],
        category="shark_g1",
        description="Conditional challenger: scores 11.33 vs 6 AIs via C4+C5 + low scavengers, loses 5-10 to C5 sacrifice",
    ),
    "Shark_G1_RegretMin_47 [5,3,3,42,47]": FixedStrategy(
        "Shark_G1_RegretMin_47 [5,3,3,42,47]",
        [5, 3, 3, 42, 47],
        category="shark_g1",
        description="Equivalent 6-AI scorer (11.33) shifting 4 troops from C5 to C4",
    ),
    "Shark_G1_HeavyAgent [7,3,3,43,44]": FixedStrategy(
        "Shark_G1_HeavyAgent [7,3,3,43,44]",
        [7, 3, 3, 43, 44],
        category="shark_g1",
        description="Balanced C4/C5 quant agent with 3-soldier floors on C2/C3",
    ),
    "Shark_G1_C5Abandoner_Low [9,15,25,49,2]": FixedStrategy(
        "Shark_G1_C5Abandoner_Low [9,15,25,49,2]",
        [9, 15, 25, 49, 2],
        category="shark_g1",
        description="Rival C5-abandoner bidding 2 on C5 and over-stacking C4 at 49",
    ),
    "Shark_G1_C3C5_Sniper [2,2,38,2,56]": FixedStrategy(
        "Shark_G1_C3C5_Sniper [2,2,38,2,56]",
        [2, 2, 38, 2, 56],
        category="shark_g1",
        description="8-point C3+C5 minimal winning coalition sniper",
    ),
}

GAME2_SHARK_VECTORS: dict[str, FixedStrategy] = {
    "Shark_G2_AuditCounter_C3C4 [1,1,34,63,1]": FixedStrategy(
        "Shark_G2_AuditCounter_C3C4 [1,1,34,63,1]",
        [1, 1, 34, 63, 1],
        category="shark_g2",
        description="Direct audit counterexample: C3+C4 concentrator bidding 63 on C4",
    ),
    "Shark_G2_AuditCounter_C4C5_A [2,2,2,43,51]": FixedStrategy(
        "Shark_G2_AuditCounter_C4C5_A [2,2,2,43,51]",
        [2, 2, 2, 43, 51],
        category="shark_g2",
        description="Direct audit counterexample: C4+C5 concentrator (43/51)",
    ),
    "Shark_G2_AuditCounter_C4C5_B [1,1,1,43,54]": FixedStrategy(
        "Shark_G2_AuditCounter_C4C5_B [1,1,1,43,54]",
        [1, 1, 1, 43, 54],
        category="shark_g2",
        description="Direct audit counterexample: C4+C5 concentrator (43/54)",
    ),
    "Shark_G2_C2C5_HeavyAgent [6,24,0,0,70]": FixedStrategy(
        "Shark_G2_C2C5_HeavyAgent [6,24,0,0,70]",
        [6, 24, 0, 0, 70],
        category="shark_g2",
        description="Heavy C1+C2+C5 quant agent targeting 8 points via C5 lock (70)",
    ),
    "Shark_G2_C3C5_Sniper [2,1,38,1,58]": FixedStrategy(
        "Shark_G2_C3C5_Sniper [2,1,38,1,58]",
        [2, 1, 38, 1, 58],
        category="shark_g2",
        description="8-point C3+C5 pod sniper with non-zero scavengers",
    ),
    "Shark_G2_C3C4_Splitter [3,1,41,54,1]": FixedStrategy(
        "Shark_G2_C3C4_Splitter [3,1,41,54,1]",
        [3, 1, 41, 54, 1],
        category="shark_g2",
        description="7-point C3+C4 pod splitter",
    ),
    "Probe_G2_6AI_Perfect [12,13,50,25,0]": FixedStrategy(
        "Probe_G2_6AI_Perfect [12,13,50,25,0]",
        [12, 13, 50, 25, 0],
        category="shark_g2",
        description="Finding 2 probe: earns payout 9 in all 216 ordered triples of the 6 exact Game 2 AI outputs",
    ),
    "Heuristic_Proportional [7,13,20,27,33]": FixedStrategy(
        "Heuristic_Proportional [7,13,20,27,33]",
        [7, 13, 20, 27, 33],
        category="heuristic",
        description="Proportional allocation matching castle weights [1,2,3,4,5]/15",
    ),
    "Heuristic_Equal [20,20,20,20,20]": FixedStrategy(
        "Heuristic_Equal [20,20,20,20,20]",
        [20, 20, 20, 20, 20],
        category="heuristic",
        description="Uniform 20 soldiers per castle",
    ),
    "All_In_C5 [0,0,0,0,100]": FixedStrategy(
        "All_In_C5 [0,0,0,0,100]",
        [0, 0, 0, 0, 100],
        category="all_in",
        description="Pure Castle 5 all-in stress test vector",
    ),
}

ALL_ARCHETYPES: dict[str, FixedStrategy] = {
    **GAME1_LLM_BASELINES,
    **GAME2_LLM_BASELINES,
    **GAME1_CANDIDATES,
    **GAME2_CANDIDATES,
    **GAME1_SHARK_VECTORS,
    **GAME2_SHARK_VECTORS,
}


def build_realistic_field_distribution(
    game: int,
    pool_size: int = 5000,
    optimizer_share: float = 0.50,
    llm_share: float = 0.38,
    all_in_share: float = 0.05,
    seed: int = 42,
) -> NDArray[np.int64]:
    """
    Construct a synthetic applicant field of size `pool_size` combining:
    - Perturbed LLM baselines (from the exact Game 1 or Game 2 table)
    - Perturbed quantitative shark strategies
    - Optional Castle-5 all-in stress vectors (`all_in_share`, can be set to 0.0)
    - Proportional/Dirichlet human noise for the remaining weight.

    Parameters
    ----------
    game : int
        1 for Game 1 field, 2 for Game 2 field.
    pool_size : int
        Number of opponent vectors to generate.
    optimizer_share : float
        Fraction of field drawn from perturbed shark/quant strategies.
    llm_share : float
        Fraction of field drawn from perturbed LLM baseline strategies.
    all_in_share : float
        Fraction of field submitting [0, 0, 0, 0, 100].
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    NDArray[np.int64]
        2D array of shape (pool_size, 5).
    """
    if game not in (1, 2):
        raise ValueError(f"game must be 1 or 2, got {game}.")
    if optimizer_share < 0 or llm_share < 0 or all_in_share < 0 or (optimizer_share + llm_share + all_in_share) > 1.0:
        raise ValueError("Shares must be non-negative and sum to at most 1.0.")

    rng = np.random.default_rng(seed)
    llm_dict = GAME1_LLM_BASELINES if game == 1 else GAME2_LLM_BASELINES
    shark_dict = GAME1_SHARK_VECTORS if game == 1 else GAME2_SHARK_VECTORS

    llm_list = [s.vector for s in llm_dict.values()]
    shark_list = [
        s.vector for k, s in shark_dict.items() if s.category in ("shark_g1", "shark_g2")
    ]
    if game == 1:
        shark_list.append(GAME1_CANDIDATES["Hero_G1_Early_Probe [10,1,4,38,47]"].vector)
        shark_list.append(GAME1_CANDIDATES["Hero_G1_C5Sac_3Bid [10,17,25,45,3]"].vector)
    else:
        shark_list.append(GAME2_CANDIDATES["Hero_G2_Early_30_70 [0,30,0,70,0]"].vector)
        shark_list.append(GAME2_CANDIDATES["Hero_G2_Fortress_3Bid [13,21,3,60,3]"].vector)

    n_all_in = int(round(pool_size * all_in_share))
    n_llm = int(round(pool_size * llm_share))
    n_opt = int(round(pool_size * optimizer_share))
    n_ordinary = max(0, pool_size - n_all_in - n_llm - n_opt)

    rows: list[NDArray[np.int64]] = []

    # 1. All-in players
    if n_all_in > 0:
        rows.append(np.tile(np.array([0, 0, 0, 0, 100], dtype=np.int64), (n_all_in, 1)))

    # 2. LLM-assisted cluster (perturbed by 0..5 soldiers to reflect prompt variations)
    if n_llm > 0:
        llm_samples = np.zeros((n_llm, 5), dtype=np.int64)
        for i in range(n_llm):
            base = llm_list[i % len(llm_list)]
            steps = 0 if (i % 3 == 0) else int(rng.integers(1, 6))
            llm_samples[i] = IntegerTransferPerturber.perturb_vector(base, steps=steps, rng=rng)
        rows.append(llm_samples)

    # 3. Quantitative optimizer cluster
    if n_opt > 0:
        opt_samples = np.zeros((n_opt, 5), dtype=np.int64)
        for i in range(n_opt):
            base = shark_list[i % len(shark_list)]
            steps = 0 if (i % 4 == 0) else int(rng.integers(1, 7))
            opt_samples[i] = IntegerTransferPerturber.perturb_vector(base, steps=steps, rng=rng)
        rows.append(opt_samples)

    # 4. Ordinary human / proportional + random integer compositions
    if n_ordinary > 0:
        n_prop = n_ordinary // 2
        n_unif = n_ordinary - n_prop
        if n_prop > 0:
            prop_pert = IntegerTransferPerturber(
                name="PerturbedProportional", base_allocation=[7, 13, 20, 27, 33], steps=12
            )
            rows.append(prop_pert.sample_batch(n_prop, rng=rng))
        if n_unif > 0:
            unif_gen = UniformSimplexGenerator()
            rows.append(unif_gen.sample_batch(n_unif, rng=rng))

    field = np.vstack(rows)
    rng.shuffle(field, axis=0)
    return field
