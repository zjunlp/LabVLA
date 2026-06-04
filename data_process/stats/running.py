"""openpi-port RunningStats with Welford + histogram quantiles.

Same math as openpi's shared/normalize.py::RunningStats, rewritten to use
plain numpy dataclasses (no pydantic dependency). Update formula:

  mean           += (batch_mean - mean)           * (n / total)
  mean_of_squares += (batch_mean_sq - mean_of_squares) * (n / total)
  variance        = mean_of_squares - mean**2
  std             = sqrt(max(variance, 0))

q01/q99 computed via 5000-bin adaptive histograms (bin edges grow to
cover new min/max as more data arrives).

Terminology
-----------
Two distinct concepts use the strings "q01" / "q99":

1. **JSON field names** (this module) — the 1st and 99th percentile values
   stored as numpy arrays in ``stats.json`` under keys ``"q01"`` / ``"q99"``.
   These field names are STABLE across code versions; renaming them breaks
   stats.json compatibility with every existing checkpoint.

2. **Normalization MODE label** (``transforms/core.py``) — historically
   ``"q99"`` and now ``"q01_q99"``. This is the string users pass to
   ``NormalizeTransformFn.mode=`` selecting the [q01, q99] → [-1, 1] rescale.
   The label was renamed for clarity; the implementation still reads the
   ``q01`` and ``q99`` JSON fields.

Do not conflate the two. The JSON fields stay ``q01`` / ``q99``.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _as_1d_numeric_vector(value, field_name: str) -> np.ndarray:
    """Coerce a JSON stats field to a 1-D float64 numpy vector, fail-loud on
    wrong rank / non-numeric / bool payloads.

    A bare ``np.asarray(..., float64)`` would silently absorb a scalar, nested
    list, bool array, or string list into the wrong rank/shape and only surface
    as a cryptic broadcast error during normalization. Non-finite values are
    already caught at normalize time; the gap was *shape*. ``bool`` is rejected
    explicitly because numpy upcasts ``[True, False]`` to ``[1.0, 0.0]`` — a
    boolean stats vector is always corrupt.
    """
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or arr.dtype.kind == "b":
        raise ValueError(
            f"stats field {field_name!r} is a boolean vector ({value!r}); "
            f"expected a 1-D numeric vector."
        )
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(
            f"stats field {field_name!r} is non-numeric (dtype={arr.dtype}, "
            f"value={value!r}); expected a 1-D numeric vector."
        )
    if arr.ndim != 1:
        raise ValueError(
            f"stats field {field_name!r} must be a 1-D vector, got "
            f"ndim={arr.ndim} shape={arr.shape} (value={value!r})."
        )
    return arr.astype(np.float64)


class StatsBundle(dict):
    """``dict[str, NormStats]`` that also preserves top-level underscore
    metadata (e.g. ``_chunk_size``) read from ``stats.json``.

    This subclass keeps the ``dict[str, NormStats]`` contract — iteration,
    indexing, and ``.items()`` yield only feature entries, with metadata held
    out-of-band on ``.meta`` — so ``load()`` no longer silently discards
    ``_chunk_size`` (and future metadata) the way a plain dict did.
    """

    meta: dict


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None
    # min/max persisted so deploy gripper clip can actually engage.
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    # Persisted so partial-coverage stats runs are detectable at load time.
    count: Optional[int] = None

    def to_json(self) -> dict:
        out = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        if self.q01 is not None:
            out["q01"] = self.q01.tolist()
        if self.q99 is not None:
            out["q99"] = self.q99.tolist()
        if self.min is not None:
            out["min"] = self.min.tolist()
        if self.max is not None:
            out["max"] = self.max.tolist()
        if self.count is not None:
            out["count"] = [int(self.count)]
        return out

    @classmethod
    def from_json(cls, d: dict) -> "NormStats":
        # ``count`` is a partial-coverage signal, so validate strictly: a loose
        # ``int(cnt)`` would swallow 3.9→3, True→1, "5"→5. Accept only a plain
        # non-bool int or a 1-element list/tuple of one (legacy stats_delta.json
        # shape); anything else fails loud rather than being silently corrected.
        cnt = d.get("count")
        if cnt is not None:
            cnt = cls._coerce_count(cnt)
        return cls(
            mean=_as_1d_numeric_vector(d["mean"], "mean"),
            std=_as_1d_numeric_vector(d["std"], "std"),
            q01=_as_1d_numeric_vector(d["q01"], "q01") if "q01" in d else None,
            q99=_as_1d_numeric_vector(d["q99"], "q99") if "q99" in d else None,
            min=_as_1d_numeric_vector(d["min"], "min") if "min" in d else None,
            max=_as_1d_numeric_vector(d["max"], "max") if "max" in d else None,
            count=cnt,
        )

    @staticmethod
    def _coerce_count(cnt) -> int:
        """Validate ``count`` as a plain non-bool int, or a 1-element list/tuple
        of one such int. Reject float / string / bool."""
        if isinstance(cnt, (list, tuple)):
            if len(cnt) != 1:
                raise ValueError(
                    f"stats 'count' list must have exactly 1 element, got "
                    f"{len(cnt)} ({cnt!r})."
                )
            inner = cnt[0]
        else:
            inner = cnt
        # bool is a subclass of int — reject it explicitly first.
        if isinstance(inner, bool) or not isinstance(inner, int):
            raise ValueError(
                f"stats 'count' must be a plain int (or 1-element int list), "
                f"got {type(inner).__name__} value {inner!r}."
            )
        return inner


class RunningStats:
    """Streaming Welford + adaptive-histogram quantiles."""

    NUM_BINS = 5000

    def __init__(self):
        self._n = 0
        self._mean = None
        self._mean_sq = None
        self._min = None
        self._max = None
        self._hist = None
        self._bin_edges = None

    def update(self, batch: np.ndarray) -> None:
        """Update stats with a new batch. `batch` may have arbitrary leading
        dimensions; last dim is the feature dim. Flattened to (N, D)."""
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, np.shape(batch)[-1])
        n, D = batch.shape
        if n == 0:
            return
        if not np.isfinite(batch).all():
            bad = np.argwhere(~np.isfinite(batch))
            raise ValueError(
                f"batch contains NaN/Inf at rows {bad[:5].tolist()}; "
                f"total non-finite elements: {int((~np.isfinite(batch)).sum())}"
            )

        if self._n == 0:
            self._mean = batch.mean(axis=0)
            self._mean_sq = (batch ** 2).mean(axis=0)
            self._min = batch.min(axis=0)
            self._max = batch.max(axis=0)
            self._hist = [np.zeros(self.NUM_BINS) for _ in range(D)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10,
                            self.NUM_BINS + 1)
                for i in range(D)
            ]
        else:
            if self._mean.size != D:
                raise ValueError(
                    f"feature dim mismatch: existing {self._mean.size}, batch {D}"
                )
            new_min = batch.min(axis=0)
            new_max = batch.max(axis=0)
            changed = np.any(new_min < self._min) or np.any(new_max > self._max)
            self._min = np.minimum(self._min, new_min)
            self._max = np.maximum(self._max, new_max)
            if changed:
                # Rebin existing histograms to the wider range, placing each old
                # bin's mass at its CENTER (not left edge): left-edge placement
                # shifts the recovered distribution one half-bin left, biasing
                # q01/q99 used for normalization. A streaming sketch (t-digest /
                # KLL) would be more accurate, but bin-center is a free win.
                for i in range(D):
                    new_edges = np.linspace(self._min[i], self._max[i],
                                            self.NUM_BINS + 1)
                    old_centers = 0.5 * (
                        self._bin_edges[i][:-1] + self._bin_edges[i][1:]
                    )
                    new_h, _ = np.histogram(
                        old_centers,
                        bins=new_edges,
                        weights=self._hist[i],
                    )
                    self._hist[i] = new_h
                    self._bin_edges[i] = new_edges

        self._n += n
        b_mean = batch.mean(axis=0)
        b_sq = (batch ** 2).mean(axis=0)
        self._mean += (b_mean - self._mean) * (n / self._n)
        self._mean_sq += (b_sq - self._mean_sq) * (n / self._n)
        for i in range(D):
            h, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._hist[i] += h

    def get_statistics(self) -> NormStats:
        if self._n < 2:
            raise ValueError("need >= 2 samples to compute statistics")
        var = self._mean_sq - self._mean ** 2
        std = np.sqrt(np.maximum(var, 0))
        q01 = self._quantile(0.01)
        q99 = self._quantile(0.99)
        return NormStats(
            mean=self._mean.copy(),
            std=std,
            q01=q01,
            q99=q99,
            min=self._min.copy() if self._min is not None else None,
            max=self._max.copy() if self._max is not None else None,
            count=int(self._n),
        )

    def _quantile(self, q: float) -> np.ndarray:
        out = np.zeros_like(self._mean)
        target = q * self._n
        for i, h in enumerate(self._hist):
            cum = np.cumsum(h)
            idx = int(np.searchsorted(cum, target))
            idx = min(idx, len(h) - 1)
            lo = self._bin_edges[i][idx]
            hi = self._bin_edges[i][idx + 1]
            prev_cum = cum[idx - 1] if idx > 0 else 0.0
            denom = h[idx] if h[idx] > 0 else 1.0
            frac = (target - prev_cum) / denom
            out[i] = lo + frac * (hi - lo)
        return out


def save(path: Path, stats: dict[str, NormStats], chunk_size: int | None = None) -> None:
    """Atomic write via tmp + os.replace so crashed processes don't leave
    partial JSON that poisons downstream loaders.

    When ``chunk_size`` is provided, embeds it as a top-level ``_chunk_size``
    key so downstream training can verify the stats were computed against the
    same chunk_size the model uses. The leading underscore distinguishes it
    from feature-key entries (which must be NormStats payloads).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {k: v.to_json() for k, v in stats.items()}
    if chunk_size is not None:
        payload["_chunk_size"] = int(chunk_size)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load(path: Path) -> "StatsBundle":
    """Load ``stats.json`` into feature entries while PRESERVING top-level
    underscore metadata (e.g. ``_chunk_size``).

    Returns a :class:`StatsBundle` (``dict[str, NormStats]`` subclass) whose
    feature entries are the non-underscore keys and whose ``.meta`` carries the
    underscore metadata verbatim, so callers treating it as ``dict[str,
    NormStats]`` are unaffected while the metadata is no longer dropped.
    """
    with open(path) as f:
        raw = json.load(f)
    bundle = StatsBundle(
        (k, NormStats.from_json(v))
        for k, v in raw.items()
        if not k.startswith("_")
    )
    bundle.meta = {k: v for k, v in raw.items() if k.startswith("_")}
    return bundle
