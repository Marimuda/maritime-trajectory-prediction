"""
Maritime Domain Validation Module

This module provides maritime-specific validation and analysis components including:
- CPA/TCPA (Closest Point of Approach / Time to Closest Point of Approach) calculations
- COLREGS (International Regulations for Preventing Collisions at Sea) compliance
- Maritime domain metrics and validators

Key Components:
- CPACalculator: Vectorized CPA/TCPA computation
- CPAValidator: Prediction accuracy validation for CPA scenarios
- COLREGSValidator: Maritime rules compliance checking
- EncounterClassifier: Vessel interaction classification
"""

from .colregs import COLREGSAction, COLREGSValidator, EncounterClassifier, EncounterType
from .cpa_tcpa import CPACalculator, CPAValidator

__all__ = [
    "CPACalculator",
    "CPAValidator",
    "EncounterClassifier",
    "COLREGSValidator",
    "EncounterType",
    "COLREGSAction",
]

__version__ = "1.0.0"
