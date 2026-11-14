# Replication report (fresh seeds 107, 109, 113)

## 1. Validity checks

- git_dirty false for all runs: yes
- commits: 8a57349; same code: yes
- plan_equiv as expected: yes
- same starts/goals across planners, seed 107: yes
- same starts/goals across planners, seed 109: yes
- same starts/goals across planners, seed 113: yes

**All checks pass.**

## 2. Final success

| Planner | Seed 107 | Seed 109 | Seed 113 | Pooled (n = 150, 95% CI) |
|---|---|---|---|---|
| CEM@10 | 0.54 | 0.58 | 0.64 | 0.59 [0.51, 0.67] |
| Tuned GD | 0.08 | 0.10 | 0.14 | 0.11 [0.06, 0.16] |
| Smoothed K16 | 0.28 | 0.20 | 0.32 | 0.27 [0.20, 0.34] |

## 3. Primary test: smoothed K16 vs tuned GD (final success)

Only K16 solves: **32**; only tuned GD solves: **8**; exact two-sided McNemar p = **0.000182** → **significant** at α = 0.05 (K16 better).

## 4. Secondary test: fooled failures (failed, final model cost < CEM@10's)

| Planner | Failures | Fooled | Stuck |
|---|---|---|---|
| Tuned GD | 134 | 98 | 36 |
| Smoothed K16 | 110 | 52 | 58 |
| CEM@10 | 62 | (reference) | |

Episodes fooled only under tuned GD: **57**; only under K16: **11**; exact two-sided McNemar p = **1.28e-08** → **significant** (K16 fooled less).

## 5. Descriptive (no tests)

- Smoothed K16 vs CEM@10: only first 10, only second 58 (p = 2.36e-09, descriptive)
- CEM@10: mean / median final distance 6.0 / 4.1; solo plan_s per seed 17 min, 17 min, 17 min
- Tuned GD: mean / median final distance 18.2 / 17.5; solo plan_s per seed 12 min, 12 min, 12 min
- Smoothed K16: mean / median final distance 12.5 / 11.3; solo plan_s per seed 12 min, 12 min, 12 min
- Phase 6 + replication pooled (seeds 101, 103, 107, 109, 113; n = 250): K16 0.26 vs tuned GD 0.12; only K16 51, only GD 16, p = 2.17e-05. Descriptive only: the phase-6 seeds motivated this replication.
