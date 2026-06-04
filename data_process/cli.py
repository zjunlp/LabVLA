"""Unified entry: python -m data_process <subcommand>

Subcommands:
  scan      — run detectors, produce JSON report
  clean     — apply cleanup (scan report or manual drop-list)
  vqa-clean — remove bad RoboInter-VQA records and write a clean manifest
  stats     — compute norm stats (openpi-style, pipeline-driven)
  migrate   — convert legacy JSON manifest → Python schema module
  validate  — run dataset-level validators across one or more repos
  preflight — gate training launch on data-integrity invariants
"""
import logging
import sys

_USAGE = "usage: python -m data_process <scan|clean|vqa-clean|stats|migrate|validate|preflight> [...args]"


def main():
    # Install DedupeFilter early so subcommands that emit warn_once (preflight,
    # stats) dedupe properly. Don't swallow failures silently: log debug for the
    # expected ImportError (optional ``utils`` not on path), warning otherwise,
    # so a genuine breakage is visible.
    try:
        from utils.logging_utils import install_dedupe_filter
        install_dedupe_filter()
    except ImportError as e:
        logging.getLogger(__name__).debug(
            "install_dedupe_filter unavailable (utils.logging_utils not "
            "importable): %r — warn_once dedupe disabled.", e
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            "install_dedupe_filter failed: %r — warn_once dedupe disabled.", e
        )
    # Don't mutate sys.argv globally: anything that re-reads it later (plugin
    # loader, wandb hook, etc.) would see corrupted state. Save the original,
    # rebuild for the subcommand, and restore in try/finally below.
    if len(sys.argv) < 2:
        print(_USAGE, file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    original_argv = list(sys.argv)
    sub_argv = [original_argv[0]] + original_argv[2:]

    dispatch = {
        "scan":      "data_process.cleanup.scan:main",
        "clean":     "data_process.cleanup.apply:main",
        "vqa-clean": "data_process.vqa_clean:main",
        "stats":     "data_process.stats.compute:main",
        "migrate":   "data_process.migrate.manifest_to_py:main",
        "validate":  "data_process.validators.run:main",
        "preflight": "data_process.preflight:main",
    }
    # Top-level -h/--help prints the router help and exits 0 BEFORE dispatch;
    # otherwise it falls through to "unknown subcommand: '-h'" and exits 2.
    if cmd in ("-h", "--help"):
        print(__doc__.strip() if __doc__ else _USAGE)
        print()
        print(_USAGE)
        sys.exit(0)
    if cmd not in dispatch:
        print(f"unknown subcommand: {cmd!r}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        sys.exit(2)

    module_path, fn_name = dispatch[cmd].split(":")
    import importlib
    module = importlib.import_module(module_path)
    entry = getattr(module, fn_name)

    try:
        sys.argv = sub_argv
        entry()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
