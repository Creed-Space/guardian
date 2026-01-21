"""
Creed Guardian - Local AI Safety for Any Device.

Free forever. No API costs. No data leaves your device.

Example:
    from creed_guardian import Guardian

    guardian = Guardian()  # Auto-selects best model for your hardware

    result = guardian.check(
        action="Delete all files in /home",
        context="User said: clean up my downloads"
    )

    if result.allowed:
        execute_action()
    else:
        print(f"Blocked: {result.reason}")
"""

from creed_guardian.exceptions import (
    EvaluationTimeoutError,
    GuardianError,
    ModelUnavailableError,
    OllamaConnectionError,
)
from creed_guardian.guardian import Guardian
from creed_guardian.types import GuardianResult, Tier

__version__ = "0.1.0"
__all__ = [
    "Guardian",
    "GuardianResult",
    "Tier",
    "GuardianError",
    "ModelUnavailableError",
    "EvaluationTimeoutError",
    "OllamaConnectionError",
]
