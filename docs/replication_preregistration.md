# Replication pre-registration (written Nov 13, 2026, before any run on these seeds)

Committed to the repository before the runs; the commit timestamp is the record.

## Why

Phase 6 (seeds 101 + 103, pre-registered) found smoothed GD K16 × 62 ahead of single-start tuned GD, 19 vs 8 discordant episodes (0.26 vs 0.15), p = 0.052: not significant. K16 also had fewer fooled failures (36 vs 64). This replication tests those two effects on data that played no part in any decision.

## Runs

- **Seeds 107, 109, 113** (fresh; primes above 50, so no episode overlaps with seeds 99/101/103 or each other except episode 0, which uses simulator seed 1 in every seed). 50 episodes each, **n = 150 paired episodes**.
- Planners, with exactly the phase-6 settings (`scripts/replication_run.sh`; configs identical to phase 6 apart from `seed`): **CEM@10**, **tuned GD** (Adam 0.1, randn 1.0, R1 × 1000), **smoothed K16 × 62** (σ₀ 1, geom → 0.1, antithetic). Optional: GD + restarts R8 × 125 (`WITH_R8=1`), descriptive only.
- Open-loop Wall, `oracle_stop=false`, `episode_chunk=25`, one run at a time on the lease GPU. A crashed run is rerun with the same settings; no run is dropped.

## Tests (analysis of record: `scripts/replication_report.py`)

- **Primary:** smoothed K16 vs tuned GD, final success, exact two-sided McNemar over the 150 paired episodes, α = 0.05.
- **Secondary:** smoothed K16 vs tuned GD on the per-episode indicator "fooled failure" (episode failed and its final `model_cost` is below CEM@10's on the same episode and seed; the phase-6 definition), exact two-sided McNemar, α = 0.05.
- Two tests, no multiplicity correction; both reported regardless of outcome.
- Power: if the true effect equals phase 6's (27% discordant, 19:8), the primary test has ≈ 0.70 power; the phase-6 estimate is probably optimistic, so less.

## Descriptive only

Per-seed and pooled success with 95% normal intervals, fooled/stuck counts, mean/median final distance, solo `plan_s`, K16 vs CEM@10 and vs GD + restarts, and phase 6 + replication pooled (250 episodes). The pooled number is not a test, because the phase-6 seeds motivated this replication.

## Validity checks before reading results

All required runs present; every run `git_dirty: false` on the same code; `plan_equiv` 150,000 (K16 148,800, R8 150,400); identical starts/goals across planners within each seed.

## Interpretation fixed in advance

- Primary significant with K16 ahead: "smoothing makes single-start gradient planning more reliable at equal compute and wall-clock", reported alongside phase 6 (restarts at least as good, CEM far ahead).
- Not significant: reported as a failed replication; the phase-6 conclusion stands unchanged.
