#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import warnings
from pathlib import Path
from typing import TypeVar

import imageio

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def write_video(video_path, stacked_frames, fps):
    # Filter out DeprecationWarnings raised from pkg_resources
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning
        )
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """
    Loads the JSON data from `fpath` and recursively fills `obj` with the
    corresponding values (strictly matching structure and types).
    Tuples in `obj` are expected to be lists in the JSON data, which will be
    converted back into tuples.

    Atomic ("all or nothing"): the recursion builds brand-new containers
    instead of mutating ``obj`` in place, so a mismatch raised deep in the
    recursion can't leave ``obj`` half-updated.
    """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):
        """Recursively rebuild a copy of `target` from `source` with strict
        structure/type checks; `target` is never mutated."""

        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")
            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )
            # Build a new dict so a mid-iteration failure can't partially mutate.
            return {k: _deserialize(target[k], source[k]) for k in target}

        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")
            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")
            return [_deserialize(target[i], source[i]) for i in range(len(target))]

        # Tuples are JSON lists; convert back to tuple.
        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")
            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")
            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))
            return tuple(converted_items)

        else:  # primitive (int, float, str, bool, None) — exact type match
            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source

    updated_obj = _deserialize(obj, data)
    return updated_obj
