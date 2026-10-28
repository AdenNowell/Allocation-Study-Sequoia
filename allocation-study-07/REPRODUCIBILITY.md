# Reproducing the calculations

These are experiments under stated assumptions, not results from the actual event.
The source was supplied as pasted code. This package assembles that code and adds
documentation and captured execution results. It does not establish the chronology
of the original decisions or prove that the simulations preceded a submission.

## Run

Use Python 3.11 or newer in a fresh virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python scripts/run_tournament.py --pods 25000 --pool-size 5000 --optimizer-share 0.50 --llm-share 0.38 --all-in-share 0.00 --seed 42
```

`results/seed-42.txt` captures the full seeded benchmark. `results/tests.txt`
captures the supplied test suite. `results/environment.txt` records versions.
For closer numerical reproduction, install the versions recorded there. Timing
and minor floating-point rounding can vary across machines and dependency versions.

## What this run actually shows

- Game 1: among the named candidates, `[10,16,25,45,4]` scores 9.3658
  average points against the generated field. The unrestricted DP response is
  `[10,4,26,3,57]`, scoring 9.7056. That response is optimal against this field
  for the additive scoring objective, up to floating-point arithmetic.
- Game 2: `[0,30,0,70,0]` leads the tested candidates at 7.0055 mean rank payout.
  `[13,21,3,60,3]` scores 5.0919 and `[11,21,4,60,4]` scores 5.0056.
  The supplied narrative does not establish that the 7-point family is best.
- `[12,13,50,25,0]` earns first-place payout in all 216 ordered triples
  from the six supplied Game 2 baseline vectors. This is a narrow exact audit,
  not a result against every possible opponent.
- The counterexample pod gives the proposed fortress vectors only 3 castle
  points and a rank payout of 2. This rules out an unconditional pod lock.

## Assumptions that matter

Both fields contain 5,000 generated allocations: 50% perturbed optimizer examples,
38% perturbed baseline examples, and 12% ordinary/random examples, with no explicit
all-in component. Game 1 uses field seed 42; Game 2 uses field seed 142.
Opponent triples use seed 42 and are sampled independently with replacement.
Every candidate faces the same 25,000 triples. This approximates an independent
opponent population; it is not a partition of a finite entrant list into pods.

The Game 2 95% column is an approximate individual mean confidence half-width.
It is not a paired confidence interval for the difference between two candidates.
Percentile rankings use sampled field probes and are estimates. Increasing the
number of pods reduces simulation noise but cannot fix an incorrect population
model. The population-wide mean of 3.75 follows payout conservation; it is not
the expected payout of every possible fixed candidate.

The supplied tests cover scoring, conservation, validation examples, and several
analytical checks. Passing them is not a complete audit of every input or helper.
Before treating the choices as final, verify the event rules, test multiple field
mixtures and held-out seeds, and keep the original decision notes separately.

## Packaging changes

The folder, display title, package metadata, and installed command use the neutral
name Allocation Study 07. Python imports retain `blotto` so existing code works.
The original mathematical rationale is retained as hypotheses, with limitations
and measured results made explicit. No gameplay logic was changed.
