"""Preset DatasetSchema factories for common robot layouts.

Importing these factories keeps schema files short for the common case:

    from schema.presets import franka8
    SCHEMA = franka8("labutopia_level3_press_v1",
                     cams=("camera_1_rgb", "camera_2_rgb", "camera_3_rgb"))

Add new presets here only when ≥3 in-tree datasets share the pattern;
otherwise authors should instantiate `DatasetSchema(...)` directly.
"""
from .franka import franka8, franka_split
from .aloha import aloha14
from .agibot import agibot_g1

__all__ = ["franka8", "franka_split", "aloha14", "agibot_g1"]
