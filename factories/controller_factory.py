# ============================================================================
# controller_factory.py: controller factory.
# Register and instantiate controller types through a factory.
# Controllers handle robot control, collection or inference, and success checks.
# ============================================================================

from typing import Dict, Type  # Type annotations.
from controllers.base_controller import BaseController  # Base controller.
from controllers.open_controller import OpenTaskController  # Open controller.
from controllers.pickpour_controller import PickPourTaskController  # Pick-and-pour controller.
from controllers.placepress_controller import PlacePressTaskController  # Place-and-press controller.
from controllers.pick_controller import PickTaskController  # Pick controller.
from controllers.pour_controller import PourTaskController  # Pour controller.
from controllers.place_controller import PlaceTaskController  # Place controller.
from controllers.press_controller import PressTaskController  # Press controller.
from controllers.shake_controller import ShakeTaskController  # Shake controller.
from controllers.stir_controller import StirTaskController  # Stir controller.
from controllers.stirglassrod_controller import StirGlassrodTaskController  # Glass rod stirring controller.
from controllers.pickplace_controller import PickPlaceTaskController  # Pick-and-place controller.
from controllers.shakebeaker_controller import ShakeBeakerTaskController  # Beaker shaking controller.
from controllers.cleanbeaker_controller import CleanBeakerTaskController  # Beaker cleaning controller.
from controllers.cleanbeaker7policy_controller import CleanBeaker7PolicyTaskController  # Beaker cleaning controller with seven policies.
from controllers.device_operate_controller import DeviceOperateController  # Device operation controller.
from controllers.opentransportpour_controller import OpenTransportPourController  # Open, transport, and pour controller.
from controllers.LiquidMixing_controller import LiquidMixingController  # Liquid mixing controller.
from controllers.close_controller import CloseTaskController  # Close controller.
from controllers.openclose_controller import OpenCloseTaskController  # Open-and-close controller.
from controllers.navigation_controller import NavigationController  # Navigation controller.
from controllers.mobile_pick_controller import MobilePickController  # Mobile picking controller.

# Map controller names to controller classes.
_controller_registry: Dict[str, Type[BaseController]] = {}


def register_controller(name: str, controller_class: Type[BaseController]):
    """Register a BaseController subclass under a unique name such as "pick" or "place"."""
    _controller_registry[name] = controller_class  # Store the name-to-class mapping.


def create_controller(controller_name: str, *args, **kwargs) -> BaseController:
    """Instantiate controller_name from the registry, forwarding args and kwargs to the constructor. Return a BaseController instance, or raise ValueError for an unregistered name."""
    if controller_name not in _controller_registry:  # Check that the controller is registered.
        raise ValueError(f": {controller_name}")  # Raise a value error.
    return _controller_registry[controller_name](*args, **kwargs)  # Instantiate and return the registered class.


# ============================================================================
# Controller registrations.
# Register all supported controller types here.
# Controller types usually correspond directly to task types.
# ============================================================================

register_controller("pickpour", PickPourTaskController)  # Register the pick-and-pour controller.
register_controller("open", OpenTaskController)  # Register the open controller.
register_controller("close", CloseTaskController)  # Register the close controller.
register_controller("openclose", OpenCloseTaskController)  # Register the open-and-close controller.
register_controller("pick", PickTaskController)  # Register the pick controller.
register_controller("pour", PourTaskController)  # Register the pour controller.
register_controller("place", PlaceTaskController)  # Register the place controller.
register_controller("pickplace", PickPlaceTaskController)  # Register the pick-and-place controller.
register_controller("placepress", PlacePressTaskController)  # Register the place-and-press controller.
register_controller("press", PressTaskController)  # Register the press controller.
register_controller("shake", ShakeTaskController)  # Register the shake controller.
register_controller("stir", StirTaskController)  # Register the stir controller.
register_controller("stirglassrod", StirGlassrodTaskController)  # Register the glass rod stirring controller.
register_controller("shakebeaker", ShakeBeakerTaskController)  # Register the beaker shaking controller.
register_controller("cleanbeaker", CleanBeakerTaskController)  # Register the beaker cleaning controller.
register_controller("cleanbeaker7policy", CleanBeaker7PolicyTaskController)  # Register the beaker cleaning controller with seven policies.
register_controller("device_operate", DeviceOperateController)  # Register the device operation controller.
register_controller("OpenTransportPour", OpenTransportPourController)  # Register the open, transport, and pour controller.
register_controller("LiquidMixing", LiquidMixingController)  # Register the liquid mixing controller.
register_controller("navigation", NavigationController)  # Register the navigation controller.
register_controller("mobile_pick", MobilePickController)  # Register the mobile picking controller.

# ============================================================================
# Additional controllers for tasks using previously unused assets.
# ============================================================================

# Level 1: additional atomic task controllers.
from controllers.pipette_controller import PipettePickController, TestTubePickController
register_controller("pipette_pick", PipettePickController)  # Register the pipette picking controller.
register_controller("testtube_pick", TestTubePickController)  # Register the test tube picking controller.

# Level 2: additional combined task controllers.
from controllers.centrifuge_controller import CentrifugeLoadController
from controllers.scale_controller import ScaleMeasureController
register_controller("centrifuge_load", CentrifugeLoadController)  # Register the centrifuge loading controller.
register_controller("scale_measure", ScaleMeasureController)  # Register the weighing controller.

# Level 3: additional controllers for tasks of moderate complexity.
from controllers.incubator_controller import IncubatorLoadController
from controllers.rotavap_controller import RotavapSetupController
register_controller("incubator_load", IncubatorLoadController)  # Register the incubator loading controller.
register_controller("rotavap_setup", RotavapSetupController)  # Register the rotary evaporator setup controller.

# Level 4: additional long-sequence task controllers.
from controllers.fumehood_controller import FumehoodOperationController
from controllers.microscope_controller import MicroscopeSampleController
register_controller("fumehood_operation", FumehoodOperationController)  # Register the fume hood controller.
register_controller("microscope_sample", MicroscopeSampleController)  # Register the microscope sample controller.

# Level 5: additional complex-sequence task controllers.
from controllers.centrifuge_controller import FullCentrifugeController
from controllers.synthesis_controller import ChemicalSynthesisController
register_controller("full_centrifuge", FullCentrifugeController)  # Register the full centrifuge controller.
register_controller("chemical_synthesis", ChemicalSynthesisController)  # Register the chemical synthesis controller.

# Controllers for additional scenarios.
from controllers.new_scenario_controllers import (
    SpectrophotometerController,
    PCRSetupController,
    GelElectrophoresisController,
    pHMeasurementController,
    AutoclaveSterilizationController
)
register_controller("spectrophotometer", SpectrophotometerController)  # Register the spectrophotometer controller.
register_controller("pcr_setup", PCRSetupController)  # Register the PCR preparation controller.
register_controller("gel_electrophoresis", GelElectrophoresisController)  # Register the gel electrophoresis controller.
register_controller("ph_measurement", pHMeasurementController)  # Register the pH measurement controller.
register_controller("autoclave_sterilization", AutoclaveSterilizationController)  # Register the autoclave controller.
