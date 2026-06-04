"""Swap an EMA state dict into a checkpoint's model.safetensors in place.

Background: training writes the *live* (gradient-stepped) weights to
``pretrained_model/model.safetensors`` and the EMA buffer to
``training_state/ema_state.pt``. Deployment loads model.safetensors, so without
this swap evaluation uses unsmoothed weights and the EMA effort is wasted. This
script overlays the EMA buffer onto the saved model so deploy reads the smoothed
weights.

The swap creates ``model.safetensors`` with EMA values, and backs up the
original live weights to ``model_live.safetensors`` (only if a backup does
not already exist).

Usage:
    python scripts/swap_ema_into_model.py <ckpt_dir>

<ckpt_dir> must contain:
  - pretrained_model/model.safetensors  (live weights, will be backed up)
  - training_state/ema_state.pt          (EMA buffer)
"""
from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", type=Path, help="path/to/checkpoint-XXXX")
    p.add_argument("--inplace", action="store_true",
                   help="overwrite model.safetensors in place (default keeps "
                        "live weights at model_live.safetensors)")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="minimum fraction of the EMA buffer that must be "
                        "applied to model.safetensors before writing "
                        "(default 0.95). Below this the swap fails loud "
                        "instead of producing a part-EMA/part-live model.")
    p.add_argument("--force", action="store_true",
                   help="write even if coverage is below --min-coverage or "
                        "shapes mismatch (records a part-EMA model on purpose).")
    args = p.parse_args()

    ckpt = args.ckpt_dir
    model_path = ckpt / "pretrained_model" / "model.safetensors"
    ema_path = ckpt / "training_state" / "ema_state.pt"
    live_backup = ckpt / "pretrained_model" / "model_live.safetensors"

    if not model_path.exists():
        sys.exit(f"missing: {model_path}")
    if not ema_path.exists():
        sys.exit(f"missing: {ema_path}")

    print(f"loading model.safetensors from {model_path}")
    model_state = load_file(str(model_path))
    print(f"loaded {len(model_state)} tensors")

    print(f"loading ema_state.pt from {ema_path}")
    ema_state = torch.load(str(ema_path), map_location="cpu", weights_only=False)
    if isinstance(ema_state, dict) and "decay" in ema_state:
        decay = ema_state.pop("decay")
        if hasattr(decay, "item"):
            decay = decay.item()
        print(f"  decay={decay}")
    ema_state = {k: v for k, v in ema_state.items() if isinstance(v, torch.Tensor)}
    print(f"loaded EMA buffer with {len(ema_state)} tensors")

    matched = 0
    missing = []
    skipped_shape = []
    for name, live_tensor in model_state.items():
        # EMA tracked named_parameters() on the unwrapped module; the safetensors
        # file may carry the same names, possibly prefixed/un-prefixed. Try
        # exact match first, then strip a leading 'model.' prefix.
        ema_tensor = ema_state.get(name)
        if ema_tensor is None and name.startswith("model."):
            ema_tensor = ema_state.get(name[len("model."):])
        if ema_tensor is None:
            missing.append(name)
            continue
        if tuple(ema_tensor.shape) != tuple(live_tensor.shape):
            skipped_shape.append((name, tuple(live_tensor.shape), tuple(ema_tensor.shape)))
            continue
        # Cast EMA fp32 back to the live dtype (bf16/fp16/fp32) so save_file
        # preserves the original dtype layout.
        model_state[name] = ema_tensor.to(live_tensor.dtype)
        matched += 1

    print(f"matched {matched}/{len(model_state)} parameters")
    if missing:
        print(f"  {len(missing)} tensors not in EMA (e.g. buffers / non-trainable):")
        for n in missing[:5]:
            print(f"    {n}")
        if len(missing) > 5:
            print(f"    ... +{len(missing) - 5} more")
    if skipped_shape:
        print(f"  {len(skipped_shape)} shape mismatches (kept live):")
        for n, live_sh, ema_sh in skipped_shape[:5]:
            print(f"    {n}: live={live_sh} ema={ema_sh}")

    # Refuse to write a part-EMA/part-live model. Coverage is the fraction of
    # the EMA buffer that actually landed in model.safetensors; a shape mismatch
    # on any matching key is never benign. Without this gate a name/shape drift
    # silently produces a mixed-weights checkpoint that deploy loads as "the EMA
    # model".
    ema_total = len(ema_state)
    coverage = (matched / ema_total) if ema_total > 0 else 0.0
    print(f"EMA coverage: {coverage:.4f} ({matched}/{ema_total} EMA tensors applied)")
    problems = []
    if ema_total == 0:
        problems.append("EMA buffer is empty (0 tensors)")
    if skipped_shape:
        problems.append(f"{len(skipped_shape)} shape mismatch(es) on matching keys")
    if coverage < args.min_coverage:
        problems.append(
            f"coverage {coverage:.4f} < --min-coverage {args.min_coverage}"
        )
    if problems:
        msg = (
            "refusing to write a part-EMA/part-live model.safetensors:\n  - "
            + "\n  - ".join(problems)
            + "\nThe EMA buffer and model.safetensors are not aligned; deploy "
            "would silently load a mixed-weights checkpoint. Re-export the EMA "
            "state, or pass --force to write anyway."
        )
        if args.force:
            print(f"[force] {msg}")
        else:
            sys.exit(msg)

    if not args.inplace and not live_backup.exists():
        print(f"backing up live weights to {live_backup}")
        shutil.copyfile(model_path, live_backup)

    print(f"writing EMA-overlaid weights to {model_path}")
    save_file(model_state, str(model_path))
    print("done")


if __name__ == "__main__":
    main()
