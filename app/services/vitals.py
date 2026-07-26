"""
Vitals and medication-adherence aggregation.

Every number produced here comes from a database row. When a patient has no
readings the service returns empty collections — it never substitutes a
plausible-looking default, because a fabricated vital sign is indistinguishable
from a real one once it reaches a clinician.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessRuleValidationException
from app.models.patient import Patient
from app.models.prescription import Medication, Prescription
from app.models.vital_reading import VitalReading
from app.repositories.vital_reading import vital_reading_repository
from app.schemas.vitals_api import (
    EXPECTED_UNITS,
    PLAUSIBLE_RANGES,
    AdherencePoint,
    VitalReadingCreate,
    VitalSeriesPoint,
    VitalStatus,
    VitalType,
)

logger = logging.getLogger(__name__)

MAX_RANGE_DAYS = 365

# Reading type -> key on the chart series point.
_SERIES_KEYS: dict[VitalType, str] = {
    VitalType.BLOOD_PRESSURE_SYSTOLIC: "systolic",
    VitalType.BLOOD_PRESSURE_DIASTOLIC: "diastolic",
    VitalType.HEART_RATE: "heartRate",
    VitalType.TEMPERATURE: "temperature",
    VitalType.OXYGEN_SATURATION: "oxygenSaturation",
    VitalType.WEIGHT: "weight",
    VitalType.BLOOD_GLUCOSE: "glucose",
}

# Thresholds used to classify a stored reading. Intentionally conservative and
# advisory only — they drive a status label, never a clinical decision.
_STATUS_BANDS: dict[VitalType, tuple[tuple[float, float], tuple[float, float]]] = {
    #                              (normal_low, normal_high), (warn_low, warn_high)
    VitalType.BLOOD_PRESSURE_SYSTOLIC: ((90.0, 130.0), (80.0, 160.0)),
    VitalType.BLOOD_PRESSURE_DIASTOLIC: ((60.0, 85.0), (50.0, 100.0)),
    VitalType.HEART_RATE: ((60.0, 100.0), (50.0, 120.0)),
    VitalType.OXYGEN_SATURATION: ((95.0, 100.0), (90.0, 100.0)),
    VitalType.BLOOD_GLUCOSE: ((70.0, 140.0), (54.0, 200.0)),
}


def classify(reading_type: VitalType, value: float) -> VitalStatus:
    """Label a reading normal/warning/critical, or normal when unclassifiable."""
    bands = _STATUS_BANDS.get(reading_type)
    if not bands:
        return VitalStatus.NORMAL
    (n_low, n_high), (w_low, w_high) = bands
    if n_low <= value <= n_high:
        return VitalStatus.NORMAL
    if w_low <= value <= w_high:
        return VitalStatus.WARNING
    return VitalStatus.CRITICAL


class VitalsService:
    async def record(
        self, db: AsyncSession, patient_id: uuid.UUID, payload: VitalReadingCreate
    ) -> VitalReading:
        """
        Store one reading after validating unit and physiological plausibility.

        Rejecting an implausible value at write time keeps the chart trustworthy:
        a mistyped 1200 mmHg would otherwise flatten every real point on the axis.
        """
        expected = EXPECTED_UNITS.get(payload.type, ())
        if expected and payload.unit not in expected:
            raise BusinessRuleValidationException(
                f"Unit '{payload.unit}' is not valid for {payload.type.value}. "
                f"Expected one of: {', '.join(expected)}."
            )

        low, high = PLAUSIBLE_RANGES.get(payload.type, (float("-inf"), float("inf")))
        if not (low <= payload.value <= high):
            raise BusinessRuleValidationException(
                f"Value {payload.value} is outside the plausible range for "
                f"{payload.type.value} ({low}–{high} {payload.unit})."
            )

        reading = VitalReading(
            patient_id=patient_id,
            type=payload.type.value,
            value=float(payload.value),
            unit=payload.unit,
            timestamp=payload.timestamp or datetime.now(timezone.utc).isoformat(),
            status=classify(payload.type, payload.value).value,
        )
        db.add(reading)
        await db.flush()

        logger.info(
            "[VITALS_RECORDED] patient=%s type=%s status=%s",
            patient_id,
            reading.type,
            reading.status,
        )
        return reading

    async def build_series(
        self, db: AsyncSession, patient_id: uuid.UUID, *, days: int = 7
    ) -> tuple[list[VitalSeriesPoint], dict[str, float]]:
        """
        Daily-bucketed vitals for the dashboard chart.

        Multiple readings on the same day are averaged. Days with no readings at
        all are omitted rather than zero-filled, so a gap in monitoring reads as
        a gap instead of a plunge to zero.
        """
        days = max(1, min(days, MAX_RANGE_DAYS))
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        readings = await vital_reading_repository.get_by_patient(
            db, patient_id, since_iso=since, limit=MAX_RANGE_DAYS * 12
        )
        if not readings:
            return [], {}

        # date -> series key -> list of values
        buckets: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        latest: dict[str, float] = {}

        for reading in readings:
            try:
                kind = VitalType(reading.type)
            except ValueError:
                continue  # unknown type: not chartable, and never invented
            key = _SERIES_KEYS.get(kind)
            if key is None:
                continue

            date_part = (reading.timestamp or "")[:10]
            if len(date_part) != 10:
                continue

            buckets[date_part][key].append(float(reading.value))
            # readings arrive oldest-first, so the last write wins
            latest[key] = float(reading.value)

        height_cm = await self._height_cm(db, patient_id)

        series: list[VitalSeriesPoint] = []
        for date_str in sorted(buckets):
            values = {
                key: round(sum(vals) / len(vals), 2)
                for key, vals in buckets[date_str].items()
            }
            bmi = self._bmi(values.get("weight"), height_cm)
            try:
                label = datetime.strptime(date_str, "%Y-%m-%d").strftime("%a")
            except ValueError:
                label = date_str

            series.append(
                VitalSeriesPoint(day=label, date=date_str, bmi=bmi, **values)
            )

        if latest.get("weight") is not None:
            bmi = self._bmi(latest["weight"], height_cm)
            if bmi is not None:
                latest["bmi"] = bmi

        return series, latest

    async def build_adherence(
        self, db: AsyncSession, patient_id: uuid.UUID, *, days: int = 7
    ) -> list[AdherencePoint]:
        """
        Daily medication adherence from real `medications` rows.

        Adherence is `taken_doses / total_doses` across the medications active on
        each day. Days with no active medication are omitted — 0% would wrongly
        imply the patient missed doses they were never prescribed.
        """
        days = max(1, min(days, MAX_RANGE_DAYS))
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=days - 1)

        result = await db.execute(
            select(Medication)
            .join(Prescription, Medication.prescription_id == Prescription.id)
            .where(Prescription.patient_id == patient_id)
            .where(Medication.deleted_at.is_(None))
        )
        medications = list(result.scalars().all())
        if not medications:
            return []

        points: list[AdherencePoint] = []
        for offset in range(days):
            day = start + timedelta(days=offset)
            day_str = day.isoformat()

            active = [
                m
                for m in medications
                if (m.start_date or "") <= day_str <= (m.end_date or "9999-12-31")
            ]
            if not active:
                continue

            expected = sum(max(1, len(m.scheduled_times or []) or 1) for m in active)
            taken = sum(min(m.taken_doses or 0, m.total_doses or 0) for m in active)
            total = sum(m.total_doses or 0 for m in active)

            # Percentage of the course completed so far, capped at 100.
            ratio = (taken / total * 100.0) if total > 0 else 0.0
            points.append(
                AdherencePoint(
                    day=day.strftime("%a"),
                    date=day_str,
                    adherence=round(min(ratio, 100.0), 1),
                    doses_taken=taken,
                    doses_expected=expected,
                )
            )

        return points

    @staticmethod
    def _bmi(weight_kg: float | None, height_cm: float | None) -> float | None:
        """BMI, or None when either input is missing. Never estimated."""
        if not weight_kg or not height_cm or height_cm <= 0:
            return None
        metres = height_cm / 100.0
        value = weight_kg / (metres * metres)
        return round(value, 1) if 5.0 < value < 100.0 else None

    @staticmethod
    async def _height_cm(db: AsyncSession, patient_id: uuid.UUID) -> float | None:
        """
        Height for BMI: the profile value, falling back to a recorded reading.
        Returns None when neither exists, which suppresses BMI entirely.
        """
        height = await db.scalar(
            select(Patient.height).where(Patient.id == patient_id)
        )
        if height:
            return float(height)

        reading = await vital_reading_repository.get_latest_by_type(
            db, patient_id, VitalType.HEIGHT.value
        )
        return float(reading.value) if reading else None


vitals_service = VitalsService()
