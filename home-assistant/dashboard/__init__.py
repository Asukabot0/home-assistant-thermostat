"""Read-only web dashboard for the Home Assistant three-room thermostat controller.

Strict isolation guarantees:
- Read-only access to controller state, policy, shadow outputs, and audit logs.
- Zero Home Assistant API calls and zero device actuation paths.
- Zero third-party runtime dependencies (pure Python 3.11+ standard library).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
