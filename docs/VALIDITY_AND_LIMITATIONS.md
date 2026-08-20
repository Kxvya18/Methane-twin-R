# Validity and Limitations

## Claims allowed after successful real-data evaluation

- performance on the configured held-out EMIT geographic test regions;
- emission-estimation agreement with EMIT V002 reference estimates;
- segmentation agreement with EMIT plume masks;
- source localization on synthetic exact-ground-truth data;
- source localization on the subset of real scenes whose metadata contains exact source coordinates;
- calibrated interval coverage on the held-out test set.

## Claims not allowed without additional evidence

- global generalization to all methane sources;
- regulatory-grade quantification;
- proof that a ranked facility caused a plume;
- exact source accuracy when using a physics proxy;
- causal responsibility of an operator;
- production deployment readiness merely from Dockerization.

## Domain shift

Synthetic plume pretraining and real imaging-spectrometer products differ. Fine-tuning is therefore evaluated empirically. If it does not improve real held-out metrics, do not advertise it as a successful technique.
