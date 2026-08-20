# Research Protocol

1. Fix study region/time period before looking at final-test performance.
2. Download EMIT V002 and record provenance.
3. Parse metadata; inspect emission units/keys.
4. Align ERA5 event meteorology.
5. Construct real samples.
6. Assign geographic group splits.
7. Fit normalization using training data only.
8. Generate independent physics pretraining data.
9. Train classical inverse/ML baselines.
10. Tune deep hyperparameters using train + validation only.
11. Pretrain on physics data.
12. Evaluate frozen/pretrained model on real validation for transfer baseline.
13. Fine-tune on real training regions.
14. Select checkpoint from validation only.
15. Calibrate on calibration split only.
16. Run final test once for the main result.
17. Run geographic/seed/ablation experiments.
18. Report effect sizes and uncertainty, not only single metrics.
19. Separate exact-source results from proxy-source results.
20. Preserve failed/negative experiments.
