# Data Card

## NASA EMIT

Default product:
- `EMITL2BCH4PLM`, Version 002
- optional background scenes: `EMITL2BCH4ENH`, Version 002

The V002 plume-complex product provides methane plume-complex COGs and JSON/GeoJSON metadata containing emissions/uncertainty information. The enhancement collection contains enhancement, uncertainty and sensitivity products. EMIT products are approximately 60 m at the equator and are WGS-84 / EPSG:4326 COG products.

NASA retired the Version 001 methane/carbon-dioxide products in March 2026. The repository therefore defaults to Version 002.

## ERA5

ERA5 single-level reanalysis is used for event meteorology:
- U10
- V10
- T2M
- surface pressure
- boundary-layer height

## Data provenance

Every downloaded/processed dataset should retain:
- source collection,
- product version,
- acquisition time,
- file hash,
- processing configuration,
- geographic split assignment.

## Negative examples

Negative patches are not arbitrary black images. They are sampled from real EMIT methane-enhancement scenes only when the local enhancement remains below a configured threshold. This reduces obvious label leakage.

## Label quality

`source_label_quality`:
- `metadata_exact`
- `physics_proxy`
- `exact_synthetic`
- `none`

Only `metadata_exact` and `exact_synthetic` may be used for claims of source-location accuracy.
