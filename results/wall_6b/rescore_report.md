# E3 rescore report (50 episodes, 12 candidates, M = 64)

## 1. Per candidate: success, mean J_σ, AUC(lower J_σ → success)

| Candidate | Success | mean J σ=0 | mean J σ=0.1 | mean J σ=0.33 | mean J σ=1 | AUC σ=0 | AUC σ=0.1 | AUC σ=0.33 | AUC σ=1 |
|---|---|---|---|---|---|---|---|---|---|
| sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99 | 0.38 | 0.293 | 0.3 | 0.389 | 0.771 | 0.92 | 0.91 | 0.88 | 0.76 |
| sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99 | 0.38 | 0.184 | 0.191 | 0.267 | 0.748 | 0.65 | 0.66 | 0.65 | 0.64 |
| endpoint_gd_s99 | 0.26 | 0.229 | 0.251 | 0.365 | 1.05 | 0.79 | 0.77 | 0.73 | 0.70 |
| endpoint_cem10_s99 | 0.54 | 0.36 | 0.365 | 0.436 | 0.744 | 0.81 | 0.81 | 0.75 | 0.59 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r0 | 0.04 | 0.531 | 0.556 | 0.637 | 0.911 | 0.95 | 0.97 | 0.98 | 0.94 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r1 | 0.08 | 0.482 | 0.506 | 0.591 | 0.914 | 0.76 | 0.78 | 0.69 | 0.66 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r2 | 0.12 | 0.448 | 0.467 | 0.551 | 0.935 | 0.88 | 0.86 | 0.87 | 0.79 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r3 | 0.16 | 0.437 | 0.461 | 0.557 | 0.878 | 0.86 | 0.83 | 0.84 | 0.76 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r4 | 0.20 | 0.499 | 0.516 | 0.582 | 0.899 | 0.86 | 0.85 | 0.85 | 0.74 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r5 | 0.16 | 0.54 | 0.567 | 0.657 | 0.942 | 0.95 | 0.94 | 0.93 | 0.70 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r6 | 0.06 | 0.508 | 0.531 | 0.617 | 0.895 | 0.94 | 0.91 | 0.94 | 0.94 |
| gd_adam_lr0.1_randn_R8_S125_save_s99#r7 | 0.24 | 0.434 | 0.459 | 0.553 | 0.847 | 0.79 | 0.76 | 0.70 | 0.57 |
| **all restarts of gd_adam_lr0.1_randn_R8_S125_save_s99 pooled** | 0.13 | 0.485 | 0.508 | 0.593 | 0.903 | 0.85 | 0.84 | 0.82 | 0.71 |

## 2. Pick per episode the candidate with the lowest J_σ

| Candidate set | Oracle (any) | pick σ=0 | pick σ=0.1 | pick σ=0.33 | pick σ=1 | σ=0.1 vs σ=0 (only / only, p) | σ=0.33 vs σ=0 (only / only, p) | σ=1 vs σ=0 (only / only, p) |
|---|---|---|---|---|---|---|---|---|
| gd_adam_lr0.1_randn_R8_S125_save_s99 (its 8 restarts) | 0.52 | 0.24 | 0.30 | 0.30 | 0.32 | 4 / 1, 0.38 | 6 / 3, 0.51 | 6 / 2, 0.29 |
| all candidates | 0.78 | 0.40 | 0.34 | 0.38 | 0.42 | 1 / 4, 0.38 | 4 / 5, 1 | 7 / 6, 1 |
| sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99 + sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99 | 0.52 | 0.42 | 0.42 | 0.42 | 0.46 | 0 / 0, 1 | 0 / 0, 1 | 2 / 0, 0.5 |
| sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99 + endpoint_gd_s99 | 0.48 | 0.40 | 0.40 | 0.40 | 0.40 | 0 / 0, 1 | 1 / 1, 1 | 1 / 1, 1 |
| sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99 + endpoint_cem10_s99 | 0.58 | 0.50 | 0.50 | 0.46 | 0.46 | 0 / 0, 1 | 0 / 2, 0.5 | 1 / 3, 0.62 |
| sgd_adam_lr0.1_randn_K16_S62_sig1_geom_s99 + gd_adam_lr0.1_randn_R8_S125_save_s99 | 0.60 | 0.30 | 0.38 | 0.38 | 0.40 | 5 / 1, 0.22 | 6 / 2, 0.29 | 7 / 2, 0.18 |
| sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99 + endpoint_gd_s99 | 0.54 | 0.44 | 0.40 | 0.40 | 0.44 | 1 / 3, 0.62 | 2 / 4, 0.69 | 4 / 4, 1 |
| sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99 + endpoint_cem10_s99 | 0.66 | 0.46 | 0.46 | 0.44 | 0.46 | 0 / 0, 1 | 0 / 1, 1 | 2 / 2, 1 |
| sgd_adam_lr0.1_randn_K8_S125_sig1_geom_s99 + gd_adam_lr0.1_randn_R8_S125_save_s99 | 0.60 | 0.30 | 0.32 | 0.40 | 0.38 | 2 / 1, 1 | 7 / 2, 0.18 | 6 / 2, 0.29 |
| endpoint_gd_s99 + endpoint_cem10_s99 | 0.62 | 0.46 | 0.46 | 0.40 | 0.44 | 0 / 0, 1 | 0 / 3, 0.25 | 2 / 3, 1 |
| endpoint_gd_s99 + gd_adam_lr0.1_randn_R8_S125_save_s99 | 0.56 | 0.32 | 0.26 | 0.26 | 0.28 | 0 / 3, 0.25 | 3 / 6, 0.51 | 3 / 5, 0.73 |
| endpoint_cem10_s99 + gd_adam_lr0.1_randn_R8_S125_save_s99 | 0.68 | 0.32 | 0.38 | 0.38 | 0.40 | 4 / 1, 0.38 | 6 / 3, 0.51 | 7 / 3, 0.34 |

## 3. E4 decision rule (pre-declared)

- gd_adam_lr0.1_randn_R8_S125_save_s99: no pass (σ=0 pick solves 12/50, AUC 0.85)
