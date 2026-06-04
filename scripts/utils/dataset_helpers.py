from __future__ import annotations

import os

import torch


_LIBC_HANDLE = None


def _try_malloc_trim() -> bool:
    """Best-effort glibc malloc_trim(0) to return free heap pages to the OS."""

    global _LIBC_HANDLE
    if _LIBC_HANDLE is False:
        return False
    if _LIBC_HANDLE is None:
        try:
            import ctypes

            _LIBC_HANDLE = ctypes.CDLL("libc.so.6")
        except Exception:
            _LIBC_HANDLE = False
            return False
    try:
        _LIBC_HANDLE.malloc_trim(0)
        return True
    except Exception:
        return False


_WORKER_TRIM_COUNTER = 0
_WORKER_TRIM_EVERY = max(
    1,
    int(os.environ.get("LABVLA_WORKER_TRIM_EVERY", "5000")),
)


def _maybe_worker_malloc_trim() -> None:
    global _WORKER_TRIM_COUNTER
    _WORKER_TRIM_COUNTER += 1
    if _WORKER_TRIM_COUNTER >= _WORKER_TRIM_EVERY:
        _WORKER_TRIM_COUNTER = 0
        _try_malloc_trim()


class TransformedAdapterDataset(torch.utils.data.Dataset):
    """Applies pre-hydrated transforms to any adapter output."""

    def __init__(self, adapter, transform_fn):
        self.adapter = adapter
        self.meta = adapter.meta
        self._transform = transform_fn

    def __getitem__(self, idx):
        sample = self.adapter[idx]
        out = self._transform(sample)
        _maybe_worker_malloc_trim()
        return out

    def __len__(self):
        return len(self.adapter)

    @property
    def num_episodes(self):
        return self.adapter.num_episodes

    @property
    def num_frames(self):
        return self.adapter.num_frames


_INCOMPATIBLE_GRIPPER_SEMANTIC_PAIRS = frozenset({
    ("velocity", "width"), ("width", "velocity"),
    ("velocity", "position"), ("position", "velocity"),
    ("velocity", "open_fraction"), ("open_fraction", "velocity"),
    ("width", "position"), ("position", "width"),
    ("position", "open_fraction"), ("open_fraction", "position"),
})


def _parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


class GripperSemanticResolveError(RuntimeError):
    """Raised when the registry fallback for gripper_semantic FAILS to run.

    Distinct from "semantic genuinely unknown" (a clean None): a failure here
    means the lookup could not even be attempted (registry import/iteration
    error), so the repo cannot be asserted conflict-free. Multi-repo training
    must fail-loud rather than silently skip the cross-dataset check.
    """


def _resolve_gripper_semantic(schema):
    """Return the schema's gripper_semantic, falling back to the registered
    in-repo schema by schema_id when the manifest-loaded one is None.

    Manifests written before the `gripper_semantic` field existed return None at
    runtime even though `schemas/*.py` declares it; without the fallback the
    cross-repo compatibility guard misses real conflicts (position vs
    open_fraction etc.). The fallback iterates the registry and matches by
    `schema_id` (registry names often differ from schema_id, so match-by-name
    is unreliable).

    Returns None ONLY when the semantic is genuinely unknown (no schema, no
    schema_id, or no registry match). A *failure* of the fallback itself
    (registry import/iteration error) raises GripperSemanticResolveError so the
    caller can fail-loud rather than mistaking it for "unknown".
    """
    if schema is None:
        return None
    val = getattr(schema, "gripper_semantic", None)
    if val is not None:
        return val
    sid = getattr(schema, "schema_id", None)
    if not sid:
        return None
    try:
        from schema.registry import registered_names, resolve as _resolve_schema
        names = list(registered_names())
    except Exception as exc:  # registry unavailable: cannot assert safety
        raise GripperSemanticResolveError(
            f"gripper_semantic registry fallback failed for schema_id={sid!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for name in names:
        try:
            candidate = _resolve_schema(name)
        except Exception:
            # One bad sibling schema must not mask the rest of the registry.
            continue
        if getattr(candidate, "schema_id", None) == sid:
            return getattr(candidate, "gripper_semantic", None)
    return None


def _check_gripper_semantic_compat(repo_ids, schemas, *, ignored_repo_ids=None) -> None:
    """Reject known-incompatible gripper semantics across multi-repo training."""

    ignored = set(ignored_repo_ids or ())
    annotated = []
    for rid, sch in zip(repo_ids, schemas):
        if rid in ignored or sch is None:
            continue
        try:
            sem = _resolve_gripper_semantic(sch)
        except GripperSemanticResolveError as exc:
            # A fallback parse FAILURE is not "semantic unknown": we cannot
            # prove this repo's gripper targets are compatible, so refuse to
            # silently skip its conflict check.
            raise ValueError(
                f"Cannot resolve gripper_semantic for repo {rid!r} "
                f"(schema_id={getattr(sch, 'schema_id', None)!r}) during "
                "multi-repo gripper-semantic compatibility check: "
                f"{exc}\n"
                "This is a resolution FAILURE, distinct from a genuinely "
                "unknown semantic. Fix the schema registry / manifest, or "
                "drop this repo from --repo_ids, before mixing it."
            ) from exc
        if sem is None:
            continue
        annotated.append((rid, sch, sem))
    if len(annotated) < 2:
        return

    for i in range(len(annotated)):
        for j in range(i + 1, len(annotated)):
            ri, si, sem_i = annotated[i]
            rj, sj, sem_j = annotated[j]
            pair = (sem_i, sem_j)
            if pair in _INCOMPATIBLE_GRIPPER_SEMANTIC_PAIRS:
                # Operator override: LABVLA_ALLOW_HETEROGENEOUS_GRIPPER_SEMANTIC=1
                # demotes the hard error to a warning (a prior pretrain mixed
                # position + open_fraction without a canonicalizer and still
                # produced a viable KI ckpt). Default stays fail-loud.
                if os.environ.get(
                    "LABVLA_ALLOW_HETEROGENEOUS_GRIPPER_SEMANTIC", "0"
                ) == "1":
                    import logging
                    logging.getLogger(__name__).warning(
                        "[gripper-semantic] heterogeneous mix accepted via "
                        "LABVLA_ALLOW_HETEROGENEOUS_GRIPPER_SEMANTIC=1: "
                        "%r(%s) + %r(%s) — q01/q99 aligns scale not semantics.",
                        ri, sem_i, rj, sem_j,
                    )
                    continue
                raise ValueError(
                    "Cross-dataset gripper-semantic mismatch detected; "
                    "refusing to train on contradictory gripper targets.\n"
                    f"  {ri!r} (schema={si.schema_id!r}): "
                    f"gripper_semantic={sem_i!r}\n"
                    f"  {rj!r} (schema={sj.schema_id!r}): "
                    f"gripper_semantic={sem_j!r}\n"
                    "These represent different physical quantities. "
                    "Drop one incompatible repo from --repo_ids or add a "
                    "gripper canonicalizer before mixing them.\n"
                    "Operator override (use with care): "
                    "export LABVLA_ALLOW_HETEROGENEOUS_GRIPPER_SEMANTIC=1"
                )
