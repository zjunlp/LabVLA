# ============================================================================
# collector_factory.py: data collector factory.
# Register and instantiate data collector types.
# Collectors save camera images, robot states, and actions to HDF5.
# ============================================================================

from typing import Dict, Type  # Type annotations.
from data_collectors.data_collector import DataCollector  # Base data collector.
from data_collectors.mock_collector import MockCollector  # Mock collector for testing.

# Map collector names to collector classes.
_collector_registry: Dict[str, Type[DataCollector]] = {}


def register_collector(name: str, collector_class: Type[DataCollector]):
    """Register a collector class under a unique name such as "default" or "mock". collector_class must inherit from DataCollector."""
    _collector_registry[name] = collector_class  # Store the name-to-class mapping.


def create_collector(collector_type: str, *args, **kwargs) -> DataCollector:
    """Instantiate collector_type from the registry, forwarding args and kwargs to its constructor. Return the DataCollector instance, or raise ValueError for an unregistered type."""
    if collector_type not in _collector_registry:  # Check that the collector type is registered.
        raise ValueError(f": {collector_type}")  # Raise a value error.
    return _collector_registry[collector_type](*args, **kwargs)  # Instantiate and return the registered class.


# ============================================================================
# Collector registrations.
# ============================================================================

register_collector("default", DataCollector)  # Register the default collector for demonstration collection.
register_collector("mock", MockCollector)  # Register the mock collector for testing and debugging.
