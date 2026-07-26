import sys
import os
import asyncio
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.db.base import Base
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.case import Case
from app.models.appointment import Appointment
from app.models.prescription import Prescription, Medication
from app.models.report import Report
from app.models.notification import NotificationItem
from app.core.security import get_password_hash
from sqlalchemy import select

async def seed():
    db_url = settings.DATABASE_URL
    print(f"Connecting to database: {db_url}")
    
    try:
        engine = create_async_engine(db_url, echo=False)
        async with engine.begin() as conn:
            # NOTE: schema is owned by Alembic (`alembic upgrade head`), not by this script.
            # `create_all` here is a convenience for throwaway local/demo databases only.
            # It does NOT write an alembic_version row, so a database built this way will
            # fail a later `alembic upgrade head` with "relation already exists". If you use
            # it, follow up with `alembic stamp head`.
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"PostgreSQL connection fallback to local SQLite: {e}")
        db_url = "sqlite+aiosqlite:///./aronofy_dev.db"
        settings.DATABASE_URL = db_url
        engine = create_async_engine(db_url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_maker() as db:
        # 1. Primary Demo Patient User
        res = await db.execute(select(User).where(User.email == "patient@aronofy.com"))
        patient_user = res.scalars().first()
        if not patient_user:
            patient_user = User(
                email="patient@aronofy.com",
                hashed_password=get_password_hash("password123"),
                role="patient",
                is_verified=True,
                is_active=True
            )
            db.add(patient_user)
            await db.flush()
            
            patient_profile = Patient(
                id=patient_user.id,
                first_name="Alex",
                last_name="Johnson",
                phone="+1555123456",
                date_of_birth="1995-04-12",
                gender="male",
                consent_flags={"dataSharing": True, "aiProcessing": True, "emergencyAccess": True}
            )
            db.add(patient_profile)
            await db.flush()
            print("[SEED] Created Patient user: patient@aronofy.com / password123")
        else:
            res_p = await db.execute(select(Patient).where(Patient.id == patient_user.id))
            patient_profile = res_p.scalars().first()

        # 2. Primary Demo Doctor User
        res = await db.execute(select(User).where(User.email == "doctor@aronofy.com"))
        doctor_user = res.scalars().first()
        if not doctor_user:
            doctor_user = User(
                email="doctor@aronofy.com",
                hashed_password=get_password_hash("password123"),
                role="doctor",
                is_verified=True,
                is_active=True
            )
            db.add(doctor_user)
            await db.flush()

            doctor_profile = Doctor(
                id=doctor_user.id,
                first_name="Dr. Sarah",
                last_name="Smith",
                phone="+1555987654",
                specialty="General Medicine & Cardiology",
                license_number="MD-889900",
                availability="available",
                verification_status="verified"
            )
            db.add(doctor_profile)
            await db.flush()
            print("[SEED] Created Doctor user: doctor@aronofy.com / password123")
        else:
            res_d = await db.execute(select(Doctor).where(Doctor.id == doctor_user.id))
            doctor_profile = res_d.scalars().first()

        # 3. Primary Demo Admin User
        res = await db.execute(select(User).where(User.email == "admin@aronofy.com"))
        admin_user = res.scalars().first()
        if not admin_user:
            admin_user = User(
                email="admin@aronofy.com",
                hashed_password=get_password_hash("password123"),
                role="admin",
                is_verified=True,
                is_active=True
            )
            db.add(admin_user)
            await db.flush()
            print("[SEED] Created Admin user: admin@aronofy.com / password123")

        # 4. Seed Demo Consultation Cases for Doctor Queue
        res_cases = await db.execute(select(Case).where(Case.doctor_id == doctor_user.id))
        existing_cases = res_cases.scalars().all()
        if not existing_cases:
            c1 = Case(
                patient_id=patient_user.id,
                patient_name="Alex Johnson",
                patient_age=30,
                patient_gender="male",
                doctor_id=doctor_user.id,
                doctor_name="Dr. Sarah Smith",
                specialty="Cardiology",
                symptom_summary="Tightness in chest after climbing stairs, occasional mild palpitations.",
                urgency_level="high",
                status="routed",
                ai_extracted_symptoms=["chest tightness", "exertional dyspnea", "palpitations"],
                ai_specialty_recommendation="Cardiology Triage",
                ai_confidence_score=0.94,
                notes="Patient requested priority evaluation."
            )
            c2 = Case(
                patient_id=patient_user.id,
                patient_name="Alex Johnson",
                patient_age=30,
                patient_gender="male",
                doctor_id=doctor_user.id,
                doctor_name="Dr. Sarah Smith",
                specialty="General Medicine",
                symptom_summary="Persistent throbbing headaches with photophobia for 3 days.",
                urgency_level="medium",
                status="in_consultation",
                ai_extracted_symptoms=["migraine", "photophobia", "nausea"],
                ai_specialty_recommendation="Neurology Evaluation",
                ai_confidence_score=0.89,
                notes="Consultation notes updated."
            )
            db.add_all([c1, c2])
            await db.flush()
            print("[SEED] Seeded 2 realistic consultation cases for Doctor Queue.")

        # 5. Seed Demo Appointments for Doctor Schedule
        res_appts = await db.execute(select(Appointment).where(Appointment.doctor_id == doctor_user.id))
        existing_appts = res_appts.scalars().all()
        if not existing_appts:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            a1 = Appointment(
                patient_id=patient_user.id,
                patient_name="Alex Johnson",
                doctor_id=doctor_user.id,
                doctor_name="Dr. Sarah Smith",
                specialty="Cardiology",
                hospital_name="Aronofy Boston General Hospital",
                date=today_str,
                time="10:00",
                duration=30,
                type="video",
                status="scheduled",
                reason="Cardiac follow-up & blood pressure evaluation",
                notes="Routine follow-up"
            )
            a2 = Appointment(
                patient_id=patient_user.id,
                patient_name="Alex Johnson",
                doctor_id=doctor_user.id,
                doctor_name="Dr. Sarah Smith",
                specialty="General Medicine",
                hospital_name="Aronofy Boston General Hospital",
                date=today_str,
                time="14:30",
                duration=30,
                type="in_person",
                status="confirmed",
                reason="Routine general health checkup",
                notes="Patient requested in-person clinic visit"
            )
            db.add_all([a1, a2])
            await db.flush()
            print("[SEED] Seeded Today's appointments for Doctor Schedule.")

        # 6. Seed Prescriptions and Medications
        res_rx = await db.execute(select(Prescription).where(Prescription.patient_id == patient_user.id))
        existing_rx = res_rx.scalars().first()
        if not existing_rx:
            # Fetch a case for Rx foreign key
            res_c = await db.execute(select(Case).where(Case.patient_id == patient_user.id).limit(1))
            case_obj = res_c.scalars().first()
            if case_obj:
                rx = Prescription(
                    case_id=case_obj.id,
                    patient_id=patient_user.id,
                    patient_name="Alex Johnson",
                    doctor_id=doctor_user.id,
                    doctor_name="Dr. Sarah Smith",
                    diagnosis="Essential Hypertension & Mild Angina",
                    notes="Take medications strictly as prescribed after meals.",
                    status="active",
                    ai_parsed=True,
                    ai_parse_confidence=0.96,
                    follow_up_date="2026-08-15"
                )
                db.add(rx)
                await db.flush()

                m1 = Medication(
                    prescription_id=rx.id,
                    name="Lisinopril",
                    generic_name="Lisinopril ACE Inhibitor",
                    dosage="10mg",
                    frequency="Once daily in the morning",
                    duration="30 days",
                    special_instructions="Take with water after breakfast.",
                    status="active",
                    scheduled_times=["08:00"],
                    taken_doses=14,
                    total_doses=30,
                    start_date="2026-07-01",
                    end_date="2026-07-31",
                    side_effects=["Mild dry cough", "Dizziness"],
                    interactions=["Potassium supplements"]
                )
                m2 = Medication(
                    prescription_id=rx.id,
                    name="Metformin",
                    generic_name="Metformin Hydrochloride",
                    dosage="500mg",
                    frequency="Twice daily with meals",
                    duration="30 days",
                    special_instructions="Take during breakfast and dinner.",
                    status="active",
                    scheduled_times=["08:00", "20:00"],
                    taken_doses=28,
                    total_doses=60,
                    start_date="2026-07-01",
                    end_date="2026-07-31",
                    side_effects=["Mild stomach upset"],
                    interactions=["Alcohol"]
                )
                db.add_all([m1, m2])
                await db.flush()
                print("[SEED] Seeded Prescription & Active Medications.")

        # 7. Seed AI Reports
        res_rep = await db.execute(select(Report).where(Report.patient_id == patient_user.id))
        existing_rep = res_rep.scalars().first()
        if not existing_rep:
            rep = Report(
                patient_id=patient_user.id,
                patient_name="Alex Johnson",
                type="ai_symptom_intake",
                title="AI Symptom Triage & Risk Report",
                summary="High priority symptom analysis indicating potential exertional angina. Cardiology consultation advised.",
                content="Clinical Summary:\n- Reported Symptoms: Chest tightness, exertional dyspnea, mild dizziness.\n- Urgency: High (Score 0.94)\n- Recommended Action: Scheduled ECG & Cardiac Panel.",
                doctor_name="Dr. Sarah Smith",
                hospital_name="Aronofy Boston General Hospital",
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                status="ready",
                ai_generated=True,
                ai_confidence_score=94.0,
                tags=["Cardiology", "Chest Pain", "AI Intake"],
                vitals={"urgency": "high", "recommended_specialty": "Cardiology", "confidence": 94.0}
            )
            db.add(rep)
            await db.flush()
            print("[SEED] Seeded AI Symptom Intake Report.")

        # 8. Seed Notifications
        res_notif = await db.execute(select(NotificationItem).where(NotificationItem.user_id == doctor_user.id))
        existing_notif = res_notif.scalars().first()
        if not existing_notif:
            now_iso = datetime.now(timezone.utc).isoformat()
            n1 = NotificationItem(
                user_id=doctor_user.id,
                type="appointment",
                title="New Consultation Scheduled",
                message="Patient Alex Johnson booked a video appointment for today at 10:00 AM.",
                timestamp=now_iso,
                read=False,
                priority="high",
                action_url="/doctor/appointments",
                action_label="View Appointment"
            )
            n2 = NotificationItem(
                user_id=patient_user.id,
                type="medication",
                title="Medicine Reminder",
                message="Time to take Lisinopril 10mg after breakfast.",
                timestamp=now_iso,
                read=False,
                priority="medium",
                action_url="/patient/medications",
                action_label="Track Medication"
            )
            n3 = NotificationItem(
                user_id=admin_user.id,
                type="system",
                title="System Activity Alert",
                message="New Doctor profile verified for Dr. Sarah Smith (General Medicine & Cardiology).",
                timestamp=now_iso,
                read=False,
                priority="low",
                action_url="/admin/doctors",
                action_label="View Doctors"
            )
            db.add_all([n1, n2, n3])
            await db.flush()
            print("[SEED] Seeded Doctor, Patient, and Admin Notifications.")

        await db.commit()
        print("\n[SEED SUCCESS] All demo data populated successfully for doctor@aronofy.com!")

if __name__ == "__main__":
    asyncio.run(seed())

