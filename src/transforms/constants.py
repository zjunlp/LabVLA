"""Legacy dispatch dicts kept intentionally empty after Phase 4.

Before the DatasetSchema refactor, this module owned `FEATURE_MAPPING`,
`MASK_MAPPING`, and `IMAGE_MAPPING` — three parallel dicts keyed by
`robot_type` that every transform hydration helper consulted. That
mechanism is now replaced by:

  - per-dataset `meta/labvla_manifest.json` (preferred)
  - auto-inference from `info.json` `features[*].names`
  - `src/robots/*.py` `RobotConfig` registry (last-resort fallback)

all unified under `src/schema/` (`DatasetSchema` + `discover_schema`).

This file is deliberately minimal so static grep shows zero references
to the old dicts anywhere outside the schema package + migration tests.
See `doc/schema_manifest.md` for the migration guide.
"""

from .utils import make_bool_mask  # re-exported for backward compat

__all__ = ["make_bool_mask"]
