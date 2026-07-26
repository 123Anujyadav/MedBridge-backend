import asyncio
import logging
from datetime import datetime, timedelta
import uuid
from sqlalchemy import select, delete
from app.core.database import AsyncSessionLocal, engine
from app.db.base import Base
from app.core.security import get_password_hash
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor
from app.models.hospital import Hospital
from app.models.case import Case
from app.models.appointment import Appointment
from app.models.prescription import Prescription, Medication
from app.models.report import Report
from app.models.vital_reading import VitalReading
from app.models.emergency import EmergencyRequest
from app.models.notification import NotificationItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed_full_data():
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        # NOTE: schema is owned by Alembic (`alembic upgrade head`), not by this script.
        # `create_all` here is a convenience for throwaway local/demo databases only.
        # It does NOT write an alembic_version row, so a database built this way will
        # fail a later `alembic upgrade head` with "relation already exists". If you use
        # it, follow up with `alembic stamp head`.
        await conn.run_sync(Base.metadata.create_all)
        
    async with AsyncSessionLocal() as db:
        logger.info("Seeding Users and Profiles...")
        
        # 1. Hospital
        hosp_result = await db.execute(select(Hospital).filter(Hospital.name == "St. Jude General Hospital"))
        hospital = hosp_result.scalars().first()
        if not hospital:
            hospital = Hospital(
                name="St. Jude General Hospital",
                address="500 Medical Center Way",
                city="Chicago",
                state="IL",
                phone="+1 (555) 000-1122",
                email="contact@stjudehospital.org",
                services=["Cardiology", "Neurology", "Emergency Triage", "General Medicine"],
                ambulance_linked=True,
                ambulance_count=8,
                emergency_capacity="available",
                total_doctors=45,
                total_beds=250,
                available_beds=42,
                rating=4.8,
                coordinates={"lat": 41.8781, "lng": -87.6298},
                verification_status="verified"
            )
            db.add(hospital)
            await db.flush()

        # 2. Patient Primary: Jane Doe (patient@example.com / patient123)
        patient_email = "patient@example.com"
        res = await db.execute(select(User).filter(User.email == patient_email))
        p_user = res.scalars().first()

        if not p_user:
            p_user = User(
                email=patient_email,
                hashed_password=get_password_hash("patient123"),
                role="patient",
                is_active=True,
                is_verified=True,
            )
            db.add(p_user)
            await db.flush()

            p_profile = Patient(
                id=p_user.id,
                first_name="Jane",
                last_name="Doe",
                phone="+1 (555) 123-4567",
                date_of_birth="1992-05-15",
                gender="female",
                blood_type="O+",
                weight=64.5,
                height=168.0,
                allergies=["Penicillin"],
                chronic_conditions=["Asthma"],
                emergency_contact={"name": "John Doe", "relationship": "Spouse", "phone": "+1 (555) 987-6543"},
                address="123 Health Ave",
                city="Chicago",
                state="IL",
                health_score=88,
            )
            db.add(p_profile)
        else:
            p_user.hashed_password = get_password_hash("patient123")
            p_user.is_active = True
            p_user.is_verified = True

        # Secondary Patients (for Doctor and Admin portals)
        secondary_patients_data = [
            {"email": "robert.smith@example.com", "first": "Robert", "last": "Smith", "dob": "1985-03-22", "gender": "male", "blood": "A+", "weight": 82.0, "height": 178.0, "score": 92},
            {"email": "emily.davis@example.com", "first": "Emily", "last": "Davis", "dob": "1998-11-04", "gender": "female", "blood": "B-", "weight": 58.0, "height": 162.0, "score": 79},
            {"email": "michael.brown@example.com", "first": "Michael", "last": "Brown", "dob": "1976-08-19", "gender": "male", "blood": "AB+", "weight": 91.5, "height": 183.0, "score": 68},
        ]
        
        patient_objs = [p_user]
        for sp in secondary_patients_data:
            res = await db.execute(select(User).filter(User.email == sp["email"]))
            usr = res.scalars().first()
            if not usr:
                usr = User(
                    email=sp["email"],
                    hashed_password=get_password_hash("patient123"),
                    role="patient",
                    is_active=True,
                    is_verified=True,
                )
                db.add(usr)
                await db.flush()
                p_prof = Patient(
                    id=usr.id,
                    first_name=sp["first"],
                    last_name=sp["last"],
                    phone="+1 (555) 444-5566",
                    date_of_birth=sp["dob"],
                    gender=sp["gender"],
                    blood_type=sp["blood"],
                    weight=sp["weight"],
                    height=sp["height"],
                    allergies=["Dust"],
                    chronic_conditions=["Hypertension"],
                    emergency_contact={"name": "Emergency Contact", "relationship": "Family", "phone": "+1 (555) 000-0000"},
                    address="456 Oak Street",
                    city="Chicago",
                    state="IL",
                    health_score=sp["score"],
                )
                db.add(p_prof)
            patient_objs.append(usr)

        # 3. Doctor Primary: Dr. Rachel Goldberg (doctor@example.com / doctor123)
        doctor_email = "doctor@example.com"
        res = await db.execute(select(User).filter(User.email == doctor_email))
        d_user = res.scalars().first()

        if not d_user:
            d_user = User(
                email=doctor_email,
                hashed_password=get_password_hash("doctor123"),
                role="doctor",
                is_active=True,
                is_verified=True,
            )
            db.add(d_user)
            await db.flush()

            d_profile = Doctor(
                id=d_user.id,
                hospital_id=hospital.id,
                first_name="Rachel",
                last_name="Goldberg",
                phone="+1 (555) 321-7654",
                specialty="Cardiology",
                license_number="MD-987654-IL",
                hospital_name="St. Jude General Hospital",
                years_of_experience=12,
                consultation_fee=150.0,
                bio="Senior Cardiologist specializing in preventive heart health.",
                rating=4.9,
                total_patients=124,
                total_cases=310,
                availability="available",
                verification_status="verified",
            )
            db.add(d_profile)
        else:
            d_user.hashed_password = get_password_hash("doctor123")
            d_user.is_active = True
            d_user.is_verified = True

        # Secondary Doctors
        sec_docs = [
            {"email": "dr.marcus@example.com", "first": "Marcus", "last": "Vance", "specialty": "Neurology", "exp": 15, "fee": 180.0, "rating": 4.8},
            {"email": "dr.sarah@example.com", "first": "Sarah", "last": "Jenkins", "specialty": "Pediatrics", "exp": 9, "fee": 120.0, "rating": 5.0},
        ]
        doc_objs = [d_user]
        for sd in sec_docs:
            res = await db.execute(select(User).filter(User.email == sd["email"]))
            usr = res.scalars().first()
            if not usr:
                usr = User(
                    email=sd["email"],
                    hashed_password=get_password_hash("doctor123"),
                    role="doctor",
                    is_active=True,
                    is_verified=True,
                )
                db.add(usr)
                await db.flush()
                d_prof = Doctor(
                    id=usr.id,
                    hospital_id=hospital.id,
                    first_name=sd["first"],
                    last_name=sd["last"],
                    phone="+1 (555) 888-9900",
                    specialty=sd["specialty"],
                    license_number=f"MD-IL-{sd['exp']}99",
                    hospital_name="St. Jude General Hospital",
                    years_of_experience=sd["exp"],
                    consultation_fee=sd["fee"],
                    bio=f"Expert in {sd['specialty']} with extensive clinical practice.",
                    rating=sd["rating"],
                    total_patients=85,
                    total_cases=190,
                    availability="available",
                    verification_status="verified",
                )
                db.add(d_prof)
            doc_objs.append(usr)

        # 4. Admin Primary (admin@example.com / admin123)
        admin_email = "admin@example.com"
        res = await db.execute(select(User).filter(User.email == admin_email))
        a_user = res.scalars().first()

        if not a_user:
            a_user = User(
                email=admin_email,
                hashed_password=get_password_hash("admin123"),
                role="admin",
                is_active=True,
                is_verified=True,
            )
            db.add(a_user)
        else:
            a_user.hashed_password = get_password_hash("admin123")
            a_user.is_active = True
            a_user.is_verified = True

        await db.flush()
        logger.info("Base Users & Profiles confirmed.")

        # 5. Cases (Triage & Consultations)
        c_res = await db.execute(select(Case).filter(Case.patient_id == p_user.id))
        existing_cases = c_res.scalars().all()
        if not existing_cases:
            case1 = Case(
                patient_id=p_user.id,
                patient_name="Jane Doe",
                patient_age=32,
                patient_gender="female",
                doctor_id=d_user.id,
                doctor_name="Dr. Rachel Goldberg",
                specialty="Cardiology",
                symptom_summary="Occasional chest tightness and shortness of breath after mild aerobic exercise.",
                urgency_level="medium",
                status="in_consultation",
                ai_extracted_symptoms=["chest tightness", "shortness of breath", "mild fatigue"],
                ai_specialty_recommendation="Cardiology",
                ai_confidence_score=0.94,
                attachments=[{"name": "ECG_Scan.pdf", "type": "pdf", "url": "https://example.com/ecg.pdf"}],
                patient_history="Asthma diagnosed 2018.",
                notes="Patient reports symptoms worsen in cold weather.",
                assigned_at=datetime.utcnow() - timedelta(days=2)
            )
            case2 = Case(
                patient_id=p_user.id,
                patient_name="Jane Doe",
                patient_age=32,
                patient_gender="female",
                doctor_id=d_user.id,
                doctor_name="Dr. Rachel Goldberg",
                specialty="General Medicine",
                symptom_summary="Routine annual health evaluation and preventative lipid profile check.",
                urgency_level="low",
                status="completed",
                ai_extracted_symptoms=["annual routine check"],
                ai_specialty_recommendation="General Medicine",
                ai_confidence_score=0.98,
                completed_at=datetime.utcnow() - timedelta(days=10)
            )
            db.add(case1)
            db.add(case2)
            await db.flush()
            logger.info("Seeded Patient Cases.")
        else:
            case1 = existing_cases[0]

        # 6. Appointments
        app_res = await db.execute(select(Appointment).filter(Appointment.patient_id == p_user.id))
        if not app_res.scalars().first():
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            tomorrow_str = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
            next_week_str = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            apt1 = Appointment(
                patient_id=p_user.id,
                patient_name="Jane Doe",
                doctor_id=d_user.id,
                doctor_name="Dr. Rachel Goldberg",
                specialty="Cardiology",
                hospital_name="St. Jude General Hospital",
                date=today_str,
                time="14:30",
                duration=30,
                type="in_person",
                status="confirmed",
                reason="Cardiovascular Follow-up & ECG Review",
                notes="Bring latest medication log.",
                room_number="Room 302B",
                case_id=case1.id
            )
            apt2 = Appointment(
                patient_id=p_user.id,
                patient_name="Jane Doe",
                doctor_id=d_user.id,
                doctor_name="Dr. Rachel Goldberg",
                specialty="Cardiology",
                hospital_name="St. Jude General Hospital",
                date=next_week_str,
                time="10:00",
                duration=30,
                type="video",
                status="scheduled",
                reason="Telehealth Consultation for Test Results",
                video_call_link="https://meet.health.example.com/room-cardio-101",
                case_id=case1.id
            )
            db.add(apt1)
            db.add(apt2)
            logger.info("Seeded Appointments.")

        # 7. Prescriptions & Medications
        rx_res = await db.execute(select(Prescription).filter(Prescription.patient_id == p_user.id))
        if not rx_res.scalars().first():
            rx = Prescription(
                case_id=case1.id,
                patient_id=p_user.id,
                patient_name="Jane Doe",
                doctor_id=d_user.id,
                doctor_name="Dr. Rachel Goldberg",
                diagnosis="Mild Exercise-Induced Asthma & Sinus Bradycardia",
                notes="Take inhaler 15 minutes before strenuous physical activity.",
                status="active",
                ai_parsed=True,
                ai_parse_confidence=0.96,
                follow_up_date=(datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
            )
            db.add(rx)
            await db.flush()

            med1 = Medication(
                prescription_id=rx.id,
                name="Albuterol Sulfate Inhaler",
                dosage="90 mcg/actuation",
                frequency="2 puffs as needed",
                duration="90 days",
                special_instructions="Inhale 2 puffs 15 minutes prior to exercise or during acute shortness of breath.",
                status="active",
                start_date=datetime.utcnow().strftime("%Y-%m-%d"),
                end_date=(datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d"),
                scheduled_times=["08:00", "20:00"],
                taken_doses=10,
                total_doses=180
            )
            med2 = Medication(
                prescription_id=rx.id,
                name="Montelukast Sodium",
                dosage="10 mg",
                frequency="Once daily at bedtime",
                duration="30 days",
                special_instructions="Take 1 tablet by mouth daily in the evening.",
                status="active",
                start_date=datetime.utcnow().strftime("%Y-%m-%d"),
                end_date=(datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d"),
                scheduled_times=["21:00"],
                taken_doses=5,
                total_doses=30
            )
            db.add(med1)
            db.add(med2)
            logger.info("Seeded Prescriptions & Medications.")

        # 8. Clinical Reports
        rep_res = await db.execute(select(Report).filter(Report.patient_id == p_user.id))
        if not rep_res.scalars().first():
            rep1 = Report(
                patient_id=p_user.id,
                patient_name="Jane Doe",
                type="ai_report",
                title="AI Triage & Clinical Summary Report",
                summary="AI analysis confirms mild cardiovascular stress during physical exertion. Normal resting heart rate.",
                content="Full AI diagnostic breakdown: Resting HR: 68 bpm. SpO2: 98%. No acute arrhythmias detected.",
                doctor_name="Dr. Rachel Goldberg",
                hospital_name="St. Jude General Hospital",
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                status="ready",
                ai_generated=True,
                ai_confidence_score=0.95,
                tags=["Cardiology", "ECG", "AI Diagnosis"],
                vitals={"heart_rate": 68, "blood_pressure": "118/76", "oxygen_saturation": 98, "temperature": 36.6}
            )
            rep2 = Report(
                patient_id=p_user.id,
                patient_name="Jane Doe",
                type="lab_result",
                title="Complete Lipid Profile & Blood Panel",
                summary="Total Cholesterol: 185 mg/dL. HDL: 58 mg/dL. LDL: 105 mg/dL. Triglycerides: 110 mg/dL.",
                content="All metabolic indicators within optimal physiological range.",
                doctor_name="Dr. Rachel Goldberg",
                hospital_name="St. Jude General Hospital",
                date=(datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d"),
                status="ready",
                ai_generated=False,
                tags=["Lab Work", "Lipid Panel"],
                vitals={"total_cholesterol": 185, "hdl": 58, "ldl": 105}
            )
            db.add(rep1)
            db.add(rep2)
            logger.info("Seeded Clinical Reports.")

        # 9. Time Series Vitals
        v_res = await db.execute(select(VitalReading).filter(VitalReading.patient_id == p_user.id))
        if not v_res.scalars().first():
            now = datetime.utcnow()
            vitals_to_seed = [
                ("heart_rate", 72.0, "bpm", "normal", now - timedelta(hours=2)),
                ("heart_rate", 78.0, "bpm", "normal", now - timedelta(hours=6)),
                ("blood_pressure_systolic", 118.0, "mmHg", "normal", now - timedelta(hours=4)),
                ("blood_pressure_diastolic", 76.0, "mmHg", "normal", now - timedelta(hours=4)),
                ("oxygen_saturation", 98.5, "%", "normal", now - timedelta(hours=1)),
                ("temperature", 36.7, "°C", "normal", now - timedelta(hours=5)),
                ("blood_sugar", 95.0, "mg/dL", "normal", now - timedelta(hours=8)),
            ]
            for v_type, val, unit, status, ts in vitals_to_seed:
                vr = VitalReading(
                    patient_id=p_user.id,
                    type=v_type,
                    value=val,
                    unit=unit,
                    timestamp=ts.isoformat(),
                    status=status
                )
                db.add(vr)
            logger.info("Seeded Vital Readings.")

        # 10. Notifications
        n_res = await db.execute(select(NotificationItem).filter(NotificationItem.user_id == p_user.id))
        if not n_res.scalars().first():
            notif1 = NotificationItem(
                user_id=p_user.id,
                title="Appointment Confirmed",
                message="Your consultation with Dr. Rachel Goldberg is scheduled for today at 14:30.",
                type="appointment",
                timestamp=datetime.utcnow().isoformat(),
                read=False,
                priority="medium"
            )
            notif2 = NotificationItem(
                user_id=p_user.id,
                title="New Clinical Report Ready",
                message="Your AI Triage & Clinical Summary Report is now available for review.",
                type="report",
                timestamp=datetime.utcnow().isoformat(),
                read=True,
                priority="low"
            )
            db.add(notif1)
            db.add(notif2)
            logger.info("Seeded Notifications.")

        await db.commit()
        logger.info("All dummy data successfully seeded across all tables!")

if __name__ == "__main__":
    asyncio.run(seed_full_data())
