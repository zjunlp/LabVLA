from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from transformers.models.qwen3_vl import Qwen3VLProcessor

import torch

from utils.constants import (
    ACTION,
    OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR, NUM_IMAGE_SLOTS,
)
from transforms.core import DataTransformFn, DataDict


@DataTransformFn.register_subclass("labvla_processor")
@dataclass
class Qwen3_VLProcessorTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = 'Qwen/Qwen3-VL-4B-Instruct'
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    spatial_merge_size: int = 2

    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    process: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.processor = Qwen3VLProcessor.from_pretrained(self.pretrained_model_name_or_path)
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

        # Validate spatial_merge_size against the actual processor: a silent
        # mismatch between the dataclass default and the HF checkpoint scales
        # image-token counts by (actual/expected)^2. `merge_size` is the standard
        # name on Qwen3-VL's image_processor; warn_once on older HF that lacks it.
        image_proc = getattr(self.processor, "image_processor", None)
        actual_merge_size = getattr(image_proc, "merge_size", None)
        if actual_merge_size is None:
            from utils.logging_utils import warn_once
            import logging as _logging
            warn_once(
                _logging.getLogger(__name__),
                ("qwen3_vl_merge_size_unavailable",
                 self.pretrained_model_name_or_path),
                "[Qwen3_VLProcessorTransformFn] processor.image_processor has "
                "no `merge_size` attribute (old HF transformers?); skipping "
                "spatial_merge_size validation. Configured value=%d.",
                self.spatial_merge_size,
            )
        elif int(actual_merge_size) != int(self.spatial_merge_size):
            raise ValueError(
                f"[Qwen3_VLProcessorTransformFn] spatial_merge_size mismatch: "
                f"configured={self.spatial_merge_size} but processor reports "
                f"merge_size={int(actual_merge_size)}. Image-token counts would "
                f"be wrong by a factor of "
                f"({int(actual_merge_size)}/{self.spatial_merge_size})^2. "
                f"Align the config or use a different checkpoint."
            )

    def _get_num_img_tokens(self, grid_thw: torch.Tensor) -> int:
        """Compute the number of image tokens per camera = prod(grid_thw) / spatial_merge_size^2."""
        return int(torch.prod(grid_thw) // self.spatial_merge_size ** 2)

    def __call__(self, data: DataDict) -> DataDict:
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        first_valid_img_inputs = None
        first_valid_num_img_token = None
        for i in range(NUM_IMAGE_SLOTS):
            k = f"{OBS_IMAGES}.image{i}"
            if data[f"{k}_mask"]:
                first_valid_img_inputs = self.processor.image_processor(
                    data[k],
                    do_rescale=False,
                )
                first_valid_num_img_token = self._get_num_img_tokens(
                    first_valid_img_inputs.image_grid_thw[-1]
                )
                break

        if first_valid_img_inputs is None:
            # Every schema-mapped camera fell back to an invalid zero-frame
            # (missing/corrupt video or transient read), so this sample has no
            # visual evidence. Fail loud rather than feed a zero-frame with all
            # vision attention zeroed, which would silently train on a vision-less
            # sample. SkipBadSamplesDataset (the default wrapper) skips it;
            # without that wrapper the error surfaces the data problem at source.
            raise ValueError(
                "[Qwen3_VLProcessorTransformFn] all schema-mapped cameras are "
                f"invalid for this sample (checked {NUM_IMAGE_SLOTS} image "
                "slots; none had a True *_mask). This sample carries no visual "
                "evidence — treating it as a bad sample. Fix the source data "
                "(missing/corrupt video or read failure) or rely on "
                "SkipBadSamplesDataset to drop it."
            )

        for i in range(NUM_IMAGE_SLOTS):
            k = f"{OBS_IMAGES}.image{i}"
            if data[f"{k}_mask"]:
                img_inputs = self.processor.image_processor(
                    data[k],
                    do_rescale=False,
                )
                pixel_values.append(img_inputs.pixel_values)
                image_grid_thw.append(img_inputs.image_grid_thw)
                num_img_token = self._get_num_img_tokens(image_grid_thw[-1][-1])
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
                attention_mask += [1] * (num_img_token + 2)
            else:
                pixel_values.append(first_valid_img_inputs.pixel_values)
                image_grid_thw.append(first_valid_img_inputs.image_grid_thw)
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * first_valid_num_img_token + [self.vision_end_token_id]
                attention_mask += [0] * (first_valid_num_img_token + 2)

        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixel_values)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(image_grid_thw)

        lang_inputs = self.processor.tokenizer(
            data[self.task_key],
            max_length=self.max_length,
            padding_side=self.padding_side,
            padding=self.padding,
            truncation=self.truncation,
        )

        input_ids += lang_inputs.input_ids
        attention_mask += lang_inputs.attention_mask
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask)

        return data


@DataTransformFn.register_subclass("unify_labvla_inputs")
@dataclass
class UnifyLabVLAInputsTransformFn(DataTransformFn):
    # Populated by hydrate_all() from schema.action_keys. Default ("action",)
    # matches the canonical single-action-key path (legacy single-key datasets);
    # multi-key schemas like robointer_droid need this overridden so per-sub-key
    # `_is_pad` tensors can be OR-aggregated into the unified `action_is_pad`.
    action_keys: tuple[str, ...] = ("action",)

    def __call__(self, data: DataDict) -> DataDict:
        # Aggregate per-action-key `_is_pad` → unified `action_is_pad`. The
        # adapter writes `{key}_is_pad` per action sub-key; ComposeFieldsTransform
        # merges the per-key *values* into `action` but leaves the `_is_pad`
        # tensors alone (not in its src_keys). OR across sub-keys: any sub-key
        # padded → whole frame padded. Skip when canonical `action_is_pad` is
        # already present (single-key path where the adapter wrote it directly).
        if "action_is_pad" not in data:
            pad_keys = [f"{k}_is_pad" for k in self.action_keys if f"{k}_is_pad" in data]
            if pad_keys:
                merged = data[pad_keys[0]].clone()
                for pk in pad_keys[1:]:
                    merged = merged | data[pk]
                data["action_is_pad"] = merged

        out = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"],
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
        }
        for i in range(NUM_IMAGE_SLOTS):
            out[f"{OBS_IMAGES}.image{i}"] = data[f"{OBS_IMAGES}.image{i}"]
            out[f"{OBS_IMAGES}.image{i}_mask"] = data[f"{OBS_IMAGES}.image{i}_mask"]
        # Preserve optional keys when present (FAST tokens produced pre-Pad; pad mask; task).
        for k in ("fast_action_tokens", "fast_action_mask", "action_is_pad", "task"):
            if k in data:
                out[k] = data[k]
        # Preserve per-annotation token/mask/weight tensors (prefix-matched).
        # Empty in OXE-style datasets → no keys forwarded → model sees no
        # annotations and takes the pure-MSE fast path.
        for k, v in data.items():
            if (k.startswith("annotation_tokens__")
                    or k.startswith("annotation_mask__")
                    or k.startswith("annotation_weight__")):
                out[k] = v
        return out
