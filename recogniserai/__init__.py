__version__ = "0.1.0"
from recogniserai.core.controller import AdaptiveController
from recogniserai.core.efficiency import EfficiencyMeter
from recogniserai.core.energy import EnergyTracker

__all__ = ["AdaptiveController", "EfficiencyMeter", "EnergyTracker"]


def about():
    """Print basic package information."""
    info = f"""
    RecogniserAI v{__version__}
    Adaptive control framework for self-regulating deep learning systems.
    Licensed under Apache 2.0.
    """
    print(info)
