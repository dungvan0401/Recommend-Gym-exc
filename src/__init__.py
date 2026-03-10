# src/__init__.py
"""
Gym Recommendation System Package
"""

__version__ = "1.0.0"
__author__ = "folontilo"

from . import rules
from . import models
from . import filters
from . import recommend

__all__ = ['rules', 'models', 'filters', 'recommend']