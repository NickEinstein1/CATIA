# Geographic regions

CATIA uses **stable string identifiers** for regions (for example `US_Gulf_Coast`). There is no external gazetteer; identifiers are project-defined and must stay consistent across config, APIs, and dashboards.

## Canonical supported regions

The authoritative list of region IDs that participate in **maps, proximity scoring, live intelligence, and dashboard focal-region selection** is the set of keys in `REGION_CENTROIDS` in `catia/geo_regions.py` (repository package root).

Keep documentation aligned with that module: when adding a region, update `REGION_CENTROIDS` first, then refresh the table below if you maintain it here.

| Region ID | Approx. centroid (lat, lon) | Notes |
|-----------|-------------------------------|--------|
| `US_Gulf_Coast` | 28.5, -90.0 | Default in CLI/API examples |
| `US_East_Coast` | 38.0, -75.0 | |
| `US_West_Coast` | 37.5, -121.0 | |
| `US_Midwest` | 41.5, -93.0 | |
| `US_Southwest` | 34.0, -112.0 | |
| `Caribbean` | 18.0, -66.0 | |
| `Southeast_Asia` | 5.0, 110.0 | |
| `South_Asia` | 22.0, 79.0 | |
| `Europe` | 48.0, 10.0 | |
| `Mediterranean` | 38.0, 18.0 | |
| `Australia` | -25.0, 134.0 | |
| `South_America` | -15.0, -60.0 | |
| `Africa` | 0.0, 20.0 | |
| `Japan` | 36.0, 138.0 | |
| `Turkey` | 39.0, 35.0 | |
| `Chile` | -30.0, -71.0 | |
| `Indonesia` | -2.0, 118.0 | |

Centroids are indicative only (documentation and visualization), **not** for underwriting boundaries.

## Per-peril applicability

Which regions are associated with each modeled peril is defined per peril under `PERIL_CONFIG` in `catia/config.py` (`"regions"` on each peril entry). A region may appear in multiple perils; pipeline behavior for a given `region` + `perils` combination depends on that config and on downstream modules.

## Mock data vs live connectors

- **`use_mock=True` (default in many examples)** — Data acquisition can produce plausible synthetic series for **arbitrary** region strings. That does not imply full platform support (see below).
- **`REGION_CENTROIDS`** — Required for focal-region visualization and distance-based live scoring when those features need a known point.
- **External APIs** — Connectors may map a subset of names (for example to ISO country codes). Unsupported names typically fall back to documented defaults in connector code.

## APIs and CLI

Request/CLI fields named `region` expect these identifiers when you rely on geo-aligned behavior. Default in schemas and examples is `US_Gulf_Coast`. See `catia/api/schemas.py` and `catia/cli.py`.

## Adding or renaming a region

1. Add or adjust the key in `REGION_CENTROIDS` in `catia/geo_regions.py`.
2. Update `PERIL_CONFIG` region lists in `catia/config.py` where the region should apply.
3. Update connector mappings in `catia/data/connectors.py` if real-data paths need ISO or vendor-specific IDs.
4. Refresh this page’s table and any notebooks/examples that reference region names.

See also: [Perils](perils.md).
