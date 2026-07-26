"""
Pydantic v2 schemas for patient vitals and adherence.

The series response keys deliberately match the `dataKey` props the existing
Recharts components already use (`day`, `systolic`, `diastolic`, `adherence`),
so the charts bind to live data without any change to the frontend.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class VitalType(StrEnum):
    """
    Recognised reading types.

    A closed vocabulary keeps chart series predictable and stops arbitrary
    strings entering the clinical record.
    """

    BLOOD_PRESSURE_SYSTOLIC = "blood_pressure_systolic"
    BLOOD_PRESSURE_DIASTOLIC = "blood_pressure_diastolic"
    HEART_RATE = "heart_rate"
    TEMPERATURE = "temperature"
    OXYGEN_SATURATION = "oxygen_saturation"
    WEIGHT = "weight"
    HEIGHT = "height"
    BLOOD_GLUCOSE = "blood_glucose"
    RESPIRATORY_RATE = "respiratory_rate"


class VitalStatus(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


# Canonical unit per reading type. Used to validate that a submitted reading
# carries the unit its type is actually measured in.
EXPECTED_UNITS: dict[VitalType, tuple[str, ...]] = {
    VitalType.BLOOD_PRESSURE_SYSTOLIC: ("mmHg",),
    VitalType.BLOOD_PRESSURE_DIASTOLIC: ("mmHg",),
    VitalType.HEART_RATE: ("bpm",),
    VitalType.TEMPERATURE: ("°C", "C", "°F", "F"),
    VitalType.OXYGEN_SATURATION: ("%",),
    VitalType.WEIGHT: ("kg", "lb"),
    VitalType.HEIGHT: ("cm", "m", "in"),
    VitalType.BLOOD_GLUCOSE: ("mg/dL", "mmol/L"),
    VitalType.RESPIRATORY_RATE: ("breaths/min",),
}

# Physiologically plausible bounds. A reading outside these is a data-entry
# error, not a clinical finding, and is rejected rather than charted.
PLAUSIBLE_RANGES: dict[VitalType, tuple[float, float]] = {
    VitalType.BLOOD_PRESSURE_SYSTOLIC: (40.0, 300.0),
    VitalType.BLOOD_PRESSURE_DIASTOLIC: (20.0, 200.0),
    VitalType.HEART_RATE: (20.0, 250.0),
    VitalType.TEMPERATURE: (25.0, 115.0),
    VitalType.OXYGEN_SATURATION: (50.0, 100.0),
    VitalType.WEIGHT: (0.5, 500.0),
    VitalType.HEIGHT: (20.0, 260.0),
    VitalType.BLOOD_GLUCOSE: (10.0, 900.0),
    VitalType.RESPIRATORY_RATE: (4.0, 80.0),
}


class VitalReadingCreate(BaseModel):
    """A new reading submitted by the patient or a device integration."""

    type: VitalType
    value: float
    unit: str = Field(min_length=1, max_length=20)
    timestamp: Optional[str] = Field(
        default=None,
        max_length=50,
        description="ISO-8601 instant. Defaults to now when omitted.",
    )


class VitalReadingResponse(BaseModel):
    id: uuid.UUID
    type: str
    value: float
    unit: str
    timestamp: str
    status: str

    class Config:
        from_attributes = True


class VitalSeriesPoint(BaseModel):
    """
    One point on the dashboard vitals chart.

    Every measure is optional: a day with only a weight reading returns weight
    and leaves the rest null rather than inventing a value.
    """

    day: str = Field(description="Short weekday label, e.g. 'Mon'.")
    date: str = Field(description="ISO date (YYYY-MM-DD) for the bucket.")
    systolic: Optional[float] = None
    diastolic: Optional[float] = None
    heartRate: Optional[float] = None
    temperature: Optional[float] = None
    oxygenSaturation: Optional[float] = None
    weight: Optional[float] = None
    bmi: Optional[float] = None
    glucose: Optional[float] = None


class AdherencePoint(BaseModel):
    """One day of medication adherence, as a percentage of doses taken."""

    day: str
    date: str
    adherence: float = Field(ge=0.0, le=100.0)
    doses_taken: int = 0
    doses_expected: int = 0


class VitalsDashboardResponse(BaseModel):
    """
    Everything the patient dashboard charts need, in one round trip.

    `has_data` lets the UI distinguish "no readings recorded yet" from a
    request that failed — the two must never look the same to a clinician.
    """

    series: list[VitalSeriesPoint] = Field(default_factory=list)
    adherence: list[AdherencePoint] = Field(default_factory=list)
    has_vitals_data: bool = False
    has_adherence_data: bool = False
    latest: dict[str, float] = Field(
        default_factory=dict,
        description="Most recent value per reading type. Empty when none exist.",
    )
    days: int = 7
