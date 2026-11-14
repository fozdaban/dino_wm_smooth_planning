| run | settings | seed | role | final | best (at) | any | dist | gap | cost | plan_eq (k) | enc img (M) | plan time | peak GB | git |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cem40_s99 | cem | 99 | tuning | 0.34 | 0.58 (9) | 0.80 | 9.5 | 145.5 | 0.0843 | 600.0 | 0.60 | 2.19 h | 2.7 | 449972f |
| e3_rescore_s99 | rescore | 99 | tuning | 0.38 | 0.38 (final) | 0.38 | 8.8 | 91.0 | 0.293 | 115.8 | 0.12 | 25 min | 2.5 | 449972f |
| gd_adam_lr0.1_randn_R1_S4000_s99 | gd steps=4000 restarts=1 save_restarts=- | 99 | tuning | 0.16 | 0.26 (551) | 0.60 | 18.1 | 175.5 | 0.109 | 600.0 | 0.20 | 1.76 h | 5.6 | 449972f |
| gd_adam_lr0.1_randn_R8_S125_save_s99 | gd steps=125 restarts=8 save_restarts=True | 99 | tuning | 0.24 | 0.44 (41) | 0.70 | 10.8 | 150.0 | 0.144 | 150.4 | 0.05 | 26 min | 5.9 | 449972f |
| gd_adam_lr0.1_randn_R8_S500_s99 | gd steps=500 restarts=8 save_restarts=- | 99 | tuning | 0.22 | 0.44 (41) | 0.76 | 13.1 | 173.7 | 0.0362 | 600.4 | 0.20 | 1.68 h | 5.9 | 449972f |
| sgd_adam_lr0.1_randn_K16_S125_sig1_geom_s99 | smooth_gd K=16 steps=125 sigma_min=0.1 | 99 | tuning | 0.22 | 0.30 (91) | 0.58 | 11.0 | 133.0 | 0.167 | 300.0 | 0.10 | 53 min | 6.2 | 449972f |
| sgd_adam_lr0.1_randn_K16_S250_sig1_geom_s99 | smooth_gd K=16 steps=250 sigma_min=0.1 | 99 | tuning | 0.22 | 0.30 (121) | 0.54 | 11.4 | 154.7 | 0.0952 | 600.0 | 0.20 | 1.75 h | 6.2 | 449972f |
| sgd_adam_lr0.1_randn_K16_S62_sig1_geom0.333_s99 | smooth_gd K=16 steps=62 sigma_min=0.333 | 99 | tuning | 0.32 | 0.34 (51) | 0.48 | 9.6 | 103.5 | 0.305 | 148.8 | 0.05 | 25 min | 6.2 | 449972f |
| sgd_adam_lr0.1_randn_K8_S125_sig1_geom0.333_s99 | smooth_gd K=8 steps=125 sigma_min=0.333 | 99 | tuning | 0.30 | 0.30 (final) | 0.54 | 10.0 | 133.5 | 0.188 | 150.0 | 0.05 | 26 min | 5.9 | 449972f |

Pooled over held-out seeds [101, 103] (95% normal interval), one row per configuration:
  (no runs with held-out seeds)

Tuning runs (seeds outside [101, 103]), not pooled:
  cem                          seed 99  cem40_s99  final 0.34  any 0.80  best 0.58 (9)
  rescore                      seed 99  e3_rescore_s99  final 0.38  any 0.38  best 0.38 (final)
  gd steps=4000 restarts=1 save_restarts=- seed 99  gd_adam_lr0.1_randn_R1_S4000_s99  final 0.16  any 0.60  best 0.26 (551)
  gd steps=125 restarts=8 save_restarts=True seed 99  gd_adam_lr0.1_randn_R8_S125_save_s99  final 0.24  any 0.70  best 0.44 (41)
  gd steps=500 restarts=8 save_restarts=- seed 99  gd_adam_lr0.1_randn_R8_S500_s99  final 0.22  any 0.76  best 0.44 (41)
  smooth_gd K=16 steps=125 sigma_min=0.1 seed 99  sgd_adam_lr0.1_randn_K16_S125_sig1_geom_s99  final 0.22  any 0.58  best 0.30 (91)
  smooth_gd K=16 steps=250 sigma_min=0.1 seed 99  sgd_adam_lr0.1_randn_K16_S250_sig1_geom_s99  final 0.22  any 0.54  best 0.30 (121)
  smooth_gd K=16 steps=62 sigma_min=0.333 seed 99  sgd_adam_lr0.1_randn_K16_S62_sig1_geom0.333_s99  final 0.32  any 0.48  best 0.34 (51)
  smooth_gd K=8 steps=125 sigma_min=0.333 seed 99  sgd_adam_lr0.1_randn_K8_S125_sig1_geom0.333_s99  final 0.30  any 0.54  best 0.30 (final)

Configurations are pooled separately; runs with different settings are never pooled.
  planners: cem, rescore, gd, smooth_gd (their planner settings differ by design)
  gd: settings that differ between its 3 configurations:
    planner.opt_steps:  gd steps=4000 restarts=1 save_restarts=-: 4000  |  gd steps=125 restarts=8 save_restarts=True: 125  |  gd steps=500 restarts=8 save_restarts=-: 500
    planner.restarts:  gd steps=4000 restarts=1 save_restarts=-: 1  |  gd steps=125 restarts=8 save_restarts=True: 8  |  gd steps=500 restarts=8 save_restarts=-: 8
    planner.save_restarts:  gd steps=4000 restarts=1 save_restarts=-: -  |  gd steps=125 restarts=8 save_restarts=True: True  |  gd steps=500 restarts=8 save_restarts=-: -
  smooth_gd: settings that differ between its 4 configurations:
    planner.K:  smooth_gd K=16 steps=125 sigma_min=0.1: 16  |  smooth_gd K=16 steps=250 sigma_min=0.1: 16  |  smooth_gd K=16 steps=62 sigma_min=0.333: 16  |  smooth_gd K=8 steps=125 sigma_min=0.333: 8
    planner.opt_steps:  smooth_gd K=16 steps=125 sigma_min=0.1: 125  |  smooth_gd K=16 steps=250 sigma_min=0.1: 250  |  smooth_gd K=16 steps=62 sigma_min=0.333: 62  |  smooth_gd K=8 steps=125 sigma_min=0.333: 125
    planner.sigma_min:  smooth_gd K=16 steps=125 sigma_min=0.1: 0.1  |  smooth_gd K=16 steps=250 sigma_min=0.1: 0.1  |  smooth_gd K=16 steps=62 sigma_min=0.333: 0.333  |  smooth_gd K=8 steps=125 sigma_min=0.333: 0.333

wrote results/wall_6b/summary.csv
