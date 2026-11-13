# Phase 6 report (seeds 101, 103)

## 1. Validity checks

- git_dirty false for all runs: yes
- commits: 9b665aa; same code: yes
- plan_equiv as expected: yes
- same starts/goals across planners, seed 101: yes
- same starts/goals across planners, seed 103: yes

**All checks pass.**

## 2. Final success

| Planner | Seed 101 | Seed 103 | Pooled (95% CI) |
|---|---|---|---|
| CEM@10 | 0.52 | 0.52 | 0.52 [0.42, 0.62] |
| Tuned GD | 0.16 | 0.14 | 0.15 [0.08, 0.22] |
| GD + restarts | 0.32 | 0.32 | 0.32 [0.23, 0.41] |
| Smoothed K16 | 0.22 | 0.30 | 0.26 [0.17, 0.35] |
| Smoothed K8 | 0.22 | 0.28 | 0.25 [0.17, 0.33] |

n = 100 paired episodes.

## 3a. Primary comparisons

| Comparison | Only first solves | Only second solves | Exact McNemar p | Significant (α 0.05) |
|---|---|---|---|---|
| Smoothed K16 vs Tuned GD | 19 | 8 | 0.0522 | no |
| Smoothed K16 vs GD + restarts | 15 | 21 | 0.405 | no |
| Smoothed K16 vs CEM@10 | 7 | 33 | 4.23e-05 | yes |

## 3b. Secondary comparison (identical compute: averaging vs independent starts)

| Comparison | Only first solves | Only second solves | Exact McNemar p | Significant (α 0.05) |
|---|---|---|---|---|
| Smoothed K8 vs GD + restarts | 14 | 21 | 0.311 | no |

## 4. Failure types (fooled = final model cost below CEM@10's on the same episode)

| Planner | Failures | Fooled | Stuck |
|---|---|---|---|
| CEM@10 | 48 | (not classified) | |
| Tuned GD | 85 | 64 | 21 |
| GD + restarts | 68 | 62 | 6 |
| Smoothed K16 | 74 | 36 | 38 |
| Smoothed K8 | 75 | 54 | 21 |

On the 21 episodes where tuned GD failed **stuck** and the 64 where it failed **fooled**, successes by planner:

| Planner | Solves of tuned-GD stuck failures | Solves of tuned-GD fooled failures |
|---|---|---|
| CEM@10 | 14 / 21 | 25 / 64 |
| GD + restarts | 9 / 21 | 19 / 64 |
| Smoothed K16 | 4 / 21 | 15 / 64 |
| Smoothed K8 | 3 / 21 | 16 / 64 |

## 5. Secondary measures (descriptive; best/any are oracle diagnostics)

| Planner | Mean / median dist | Mean model cost | Mean gap | Best (at), per seed | Any | plan_equiv / run | Solo plan_s per seed | GPU |
|---|---|---|---|---|---|---|---|---|
| CEM@10 | 7.0 / 4.4 | 0.367 | 84 | 0.54 (9), 0.52 (10) | 0.74 | 150000, 150000 | 17 min, 17 min | Quadro RTX 6000 |
| Tuned GD | 16.5 / 14.5 | 0.173 | 169 | 0.16 (101), 0.28 (371) | 0.41 | 150000, 150000 | 12 min, 12 min | Quadro RTX 6000 |
| GD + restarts | 12.5 / 8.4 | 0.141 | 148 | 0.44 (51), 0.36 (41) | 0.69 | 150400, 150400 | 12 min, 12 min | Quadro RTX 6000 |
| Smoothed K16 | 12.5 / 12.0 | 0.396 | 98 | 0.22 (61), 0.30 (51) | 0.41 | 148800, 148800 | 12 min, 12 min | Quadro RTX 6000 |
| Smoothed K8 | 12.4 / 10.5 | 0.26 | 121 | 0.28 (71), 0.32 (111) | 0.50 | 150000, 150000 | 12 min, 12 min | Quadro RTX 6000 |

Footnotes: episode 0 has the same start/goal states in every seed (simulator seed `seed × n + 1`); all tuning was on seed 99, with K8/K16 and σ₀ ties broken by mean distance.
