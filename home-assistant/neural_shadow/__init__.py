"""Independent neural-network thermostat shadow system (observation and scoring only).

This package has no device actuation path and never communicates with Home Assistant
services. It reads read-only snapshots/databases and writes to an independent directory.
"""
from enum import Enum

SCHEMA_VERSION = 1
FEATURE_VERSION = "ns-feat-v1"
MODEL_ARCH_VERSION = "ns-gru-v1"
CONSTRAINT_VERSION = "ns-v1-hold-pm1c"
HORIZONS_MINUTES = (15, 30, 60)
STEP_MINUTES = 5
DEFAULT_HISTORY_MINUTES = 180


class ModelState(str, Enum):
    """Explicit lifecycle state of a shadow neural model artifact."""

    UNTRAINED = "untrained"
    INSUFFICIENT_DATA = "insufficient_data"
    UNCALIBRATED = "uncalibrated"
    READY_TO_USE = "ready_to_use"


class DataOrigin(str, Enum):
    """Distinguishes genuine site history from synthetic verification data."""

    REAL = "real"
    REAL_SHORT_PACK = "real_short_evidence_pack"
    SYNTHETIC = "synthetic_verification_only"
