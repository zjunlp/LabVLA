"""In-repo schema registry — lazy auto-import.

`schema.registry._ensure_autoload()` calls `importlib.import_module("schemas")`
once per process. Importing this package imports every sibling `*.py` file
(not `_`-prefixed), and each of those files is expected to either:

  (a) define a module-level `SCHEMA: DatasetSchema`, or
  (b) call `schema.register(schema, name=..., aliases=[...])` at import time.

Out-of-repo schemas can live anywhere — pass them via
`--dataset_schema /abs/path/file.py` (or set LABVLA_SCHEMA_PATH) instead.
"""
from __future__ import annotations

from pathlib import Path
import importlib
import logging

_logger = logging.getLogger(__name__)

# C11/C20: sibling modules that failed to import during autoload, as
# ``[(module_name, exception), ...]``. Previously a failed import was only
# warning-logged and the broken schema then looked like an "unknown schema"
# downstream, masking the real ImportError root cause. The registry reads this
# list after autoload and fails loud (splicing the real exception) on the first
# ``resolve()`` / ``registered_names()`` call.
IMPORT_FAILURES: list[tuple[str, Exception]] = []


def _autoload_siblings() -> None:
    """Import every non-`_`-prefixed `.py` sibling and register any top-level
    `SCHEMA` found. Wrapped in a function so we don't leak loop-local names
    into the package namespace (and avoid a NameError `del` when the dir has
    no matching files).
    """
    this_dir = Path(__file__).parent
    from schema import register, DatasetSchema  # local import to avoid cycles at package-init time
    IMPORT_FAILURES.clear()
    for py in sorted(this_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        modname = f"schemas.{py.stem}"
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            # Record (don't swallow) — the registry surfaces this loudly.
            _logger.warning("schemas/%s failed to import: %r", py.name, e)
            IMPORT_FAILURES.append((modname, e))
            continue
        s = getattr(mod, "SCHEMA", None)
        if isinstance(s, DatasetSchema):
            aliases = [s.schema_id] if s.schema_id and s.schema_id != py.stem else []
            register(s, name=py.stem, aliases=aliases)


_autoload_siblings()
