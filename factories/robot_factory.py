# ============================================================================
# robot_factory.py: robot factory.
# Register and instantiate robot types through a factory.
# ============================================================================

from typing import Dict, Type  # Type annotations.
from isaacsim.core.api.robots.robot import Robot  # Isaac Sim Robot base class.
from robots.franka.franka import Franka  # Franka Panda robot implementation.
from robots.ridgebase_franka.ridgebase import Ridgebase  # Ridgebase mobile robot implementation.

# Map robot type names to robot classes.
_robot_registry: Dict[str, Type[Robot]] = {}


def register_robot(name: str, robot_class: Type[Robot]):
    """Register a Robot subclass under a unique type name such as "franka" or "ridgebase"."""
    _robot_registry[name] = robot_class  # Store the name-to-class mapping.


def create_robot(robot_type: str, *args, **kwargs) -> Robot:
    """Instantiate robot_type from the registry, forwarding args and kwargs such as position and orientation to its constructor. Return a Robot instance, or raise ValueError for an unregistered type."""
    if robot_type not in _robot_registry:  # Check that the robot type is registered.
        raise ValueError(f": {robot_type}")  # Raise a value error.
    return _robot_registry[robot_type](*args, **kwargs)  # Instantiate and return the registered class.


# ============================================================================
# Robot registrations.
# Register all supported robot types here.
# ============================================================================

register_robot("franka", Franka)  # Register the Franka Panda arm.
register_robot("ridgebase", Ridgebase)  # Register the Ridgebase mobile robot.
