# Perils

CATIA identifies modeled catastrophe types by **stable string slugs** (for example `hurricane`, `flood`). The authoritative catalog lives in **`PERIL_CONFIG`** in `catia/config.py`.

## Supported modeled perils

These keys are the supported peril identifiers for simulation, multi-peril analysis, APIs that accept a `perils` list, and live-feed mapping (`infer_catia_peril` in `catia/live_intel.py`) where applicable.

| Slug | Display name | Summary |
|------|----------------|---------|
| `hurricane` | Hurricane | Tropical cyclones (wind-driven catastrophe modeling assumptions). |
| `flood` | Flood | River and flash flooding (frequency–severity parameters in config). |
| `wildfire` | Wildfire | Wildland fire peril with seasonal assumptions by region set. |
| `earthquake` | Earthquake | Seismic peril (year-round seasonality in config). |
| `drought` | Drought | Agricultural / hydrological drought-style peril with severity scale in config. |

Each entry also defines **`regions`** (where that peril is considered applicable in config—see [Regions](regions.md)), **`seasonality`** (calendar months), **`frequency_base`**, **`severity_params`**, **`climate_drivers`**, and **`magnitude_scale`**. Tune values there rather than hard-coding in call sites.

## Default peril bundle

**`DEFAULT_PERILS`** in `catia/config.py` lists the perils used when the pipeline does not specify a subset:

`hurricane`, `flood`, `wildfire`, `earthquake`

**`drought`** is fully configured in `PERIL_CONFIG` and **`INTENSITY_DISTRIBUTION`** but is **not** included in `DEFAULT_PERILS`; pass `perils=[..., "drought"]` explicitly when you want it in a run.

## Intensity and loss modeling

**`INTENSITY_DISTRIBUTION`** in `catia/config.py` defines a sampling distribution per peril for exposure-based intensity (units are documented inline there—wind mph, flood depth ft, wildfire index, earthquake MMI, drought severity scale). Keys align with `PERIL_CONFIG` peril slugs.

## Related sources

- **Correlation matrix path**: `SIMULATION_CONFIG["correlation_matrix_path"]` (`catia/config.py`) for multi-peril dependency assumptions.
- **Climate scenarios**: `CLIMATE_SCENARIOS` in `catia/config.py` applies optional frequency/severity multipliers **by peril slug**.
- **Regions**: [Regions](regions.md) — how geographic IDs relate to each peril’s `"regions"` list.

## Adding a peril

1. Add a new slug entry under **`PERIL_CONFIG`** with the same structural keys as existing perils.
2. Add a matching entry under **`INTENSITY_DISTRIBUTION`** if exposure-based simulation should support it.
3. Extend **`infer_catia_peril`** (and any feed parsers) if live events should map to the new slug.
4. Update correlation/scenario CSV or **`CLIMATE_SCENARIOS`** if multi-peril or stress logic must include it.
5. Add tests covering API/pipeline acceptance of the new slug if it becomes user-facing.

See also: [Regions](regions.md).
