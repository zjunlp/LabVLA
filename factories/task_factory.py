# ============================================================================
# task_factory.py: task factory.
# Register and instantiate task types through a factory.
# Tasks provide scene state, camera data, object placement, and task status.
# ============================================================================

from typing import Dict, Type  # Type annotations.
from tasks.base_task import BaseTask  # Base task.
from tasks.open_task import OpenTask  # Door and drawer opening task.
from tasks.pick_task import PickTask  # Pick task.
from tasks.place_task import PlaceTask  # Place task.
from tasks.press_task import PressTask  # Press task.
from tasks.shake_task import ShakeTask  # Shake task.
from tasks.stir_task import StirTask  # Stir task.
from tasks.pickpour_task import PickPourTask  # Pick-and-pour task.
from tasks.pickplace_task import PickPlaceTask  # Pick-and-place task.
from tasks.placepress_task import PlacePressTask  # Place-and-press task.
from tasks.cleanbeaker_task import CleanBeakerTask  # Beaker cleaning task.
from tasks.device_operate_task import DeviceOperateTask  # Device operation task.
from tasks.opentransportpour_task import OpenTransportPourTask  # Open, transport, and pour task.
from tasks.LiquidMixing_task import LiquidMixing  # Liquid mixing task.
from tasks.navigation_task import NavigationTask  # Navigation task.
from tasks.mobile_pick_task import MobilePickTask  # Mobile picking task.

# Map task names to task classes.
_task_registry: Dict[str, Type[BaseTask]] = {}


def register_task(name: str, task_class: Type[BaseTask]):
    """Register a BaseTask subclass under a unique name such as "pick", "place", or "pour"."""
    _task_registry[name] = task_class  # Store the name-to-class mapping.


def create_task(task_name: str, *args, **kwargs) -> BaseTask:
    """Instantiate task_name from the registry, forwarding args and kwargs such as cfg, world, stage, and robot to its constructor. Return a BaseTask instance, or raise ValueError for an unregistered name."""
    if task_name not in _task_registry:  # Check that the task is registered.
        raise ValueError(f": {task_name}")  # Raise a value error.
    return _task_registry[task_name](*args, **kwargs)  # Instantiate and return the registered class.


# ============================================================================
# Task registrations.
# Register all supported task types here.
# Level 1: atomic tasks.
# ============================================================================

register_task("pick", PickTask)  # Register the pick task.
register_task("place", PlaceTask)  # Register the place task.
register_task("press", PressTask)  # Register the press task.
register_task("shake", ShakeTask)  # Register the shake task.
register_task("stir", StirTask)  # Register the stir task.
register_task("openclose", OpenTask)  # Register the open-and-close task.

# ============================================================================
# Levels 2-4: combined and complex tasks.
# ============================================================================

register_task("device_operate", DeviceOperateTask)  # Register the device operation task.
register_task("pickpour", PickPourTask)  # Register the pick-and-pour task.
register_task("pickplace", PickPlaceTask)  # Register the pick-and-place task.
register_task("placepress", PlacePressTask)  # Register the place-and-press task.
register_task("cleanbeaker", CleanBeakerTask)  # Register the beaker cleaning task.
register_task("OpenTransportPour", OpenTransportPourTask)  # Register the open, transport, and pour task.
register_task("LiquidMixing", LiquidMixing)  # Register the liquid mixing task.
register_task("navigation", NavigationTask)  # Register the navigation task.
register_task("mobile_pick", MobilePickTask)  # Register the mobile picking task.

# ============================================================================
# Additional tasks using previously unused assets.
# ============================================================================

# Level 1: additional atomic tasks.
from tasks.pipette_pick_task import PipettePickTask, TestTubePickTask
register_task("pipette_pick", PipettePickTask)  # Register the pipette picking task.
register_task("testtube_pick", TestTubePickTask)  # Register the test tube picking task.

# Level 2: additional combined tasks.
from tasks.centrifuge_task import CentrifugeLoadTask
from tasks.scale_task import ScaleMeasureTask
register_task("centrifuge_load", CentrifugeLoadTask)  # Register the centrifuge loading task.
register_task("scale_measure", ScaleMeasureTask)  # Register the weighing task.

# Level 3: additional tasks of moderate complexity.
from tasks.incubator_task import IncubatorLoadTask
from tasks.rotavap_task import RotavapSetupTask
register_task("incubator_load", IncubatorLoadTask)  # Register the incubator loading task.
register_task("rotavap_setup", RotavapSetupTask)  # Register the rotary evaporator setup task.

# Level 4: additional long-sequence tasks.
from tasks.fumehood_task import FumehoodOperationTask
from tasks.microscope_task import MicroscopeSampleTask
register_task("fumehood_operation", FumehoodOperationTask)  # Register the fume hood task.
register_task("microscope_sample", MicroscopeSampleTask)  # Register the microscope sample task.

# Level 5: additional complex-sequence tasks.
from tasks.centrifuge_task import FullCentrifugeTask
from tasks.synthesis_task import ChemicalSynthesisTask
register_task("full_centrifuge", FullCentrifugeTask)  # Register the full centrifuge task.
register_task("chemical_synthesis", ChemicalSynthesisTask)  # Register the chemical synthesis task.

# Additional scenarios.
from tasks.new_scenario_tasks import (
    SpectrophotometerTask,
    PCRSetupTask,
    GelElectrophoresisTask,
    pHMeasurementTask,
    AutoclaveSterilizationTask
)
register_task("spectrophotometer", SpectrophotometerTask)  # Register the spectrophotometer task.
register_task("pcr_setup", PCRSetupTask)  # Register the PCR preparation task.
register_task("gel_electrophoresis", GelElectrophoresisTask)  # Register the gel electrophoresis task.
register_task("ph_measurement", pHMeasurementTask)  # Register the pH measurement task.
register_task("autoclave_sterilization", AutoclaveSterilizationTask)  # Register the autoclave task.
