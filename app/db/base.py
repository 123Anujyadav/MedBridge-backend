# Import all models so they are registered on Base metadata
# for Alembic auto-migration discovery.
from app.db.base_class import Base  # noqa
from app.models.user import User  # noqa
from app.models.patient import Patient  # noqa
from app.models.doctor import Doctor  # noqa
from app.models.hospital import Hospital  # noqa
from app.models.case import Case, Symptom  # noqa
from app.models.prescription import Prescription, Medication  # noqa
from app.models.rx_verification import (  # noqa
    PrescriptionVerification,
    VerificationFinding,
)
from app.models.pharmacy import (  # noqa
    Pharmacy,
    PharmacyInventory,
    PharmacyDocument,
    PharmacyVerificationEvent,
)
from app.models.delivery import (  # noqa
    DeliveryPartner,
    DeliveryAssignment,
    DeliveryEvent,
)
from app.models.medicine_order import (  # noqa
    MedicineOrder,
    MedicineOrderItem,
    OrderStatusEvent,
)
from app.models.appointment import Appointment  # noqa
from app.models.report import Report  # noqa
from app.models.report_version import ReportVersion  # noqa
from app.models.notification import NotificationItem  # noqa
from app.models.emergency import EmergencyRequest, EmergencyStatusEvent  # noqa
from app.models.emergency_profile import EmergencyProfile  # noqa
from app.models.communication import CommunicationLog  # noqa
from app.models.audit import AuditLog  # noqa
from app.models.consent import ConsentRecord  # noqa
from app.models.vital_reading import VitalReading  # noqa
from app.models.intake import IntakeExtractedEntity, IntakeSessionRecord  # noqa
from app.models.assistant import AssistantConversation, AssistantMessage  # noqa
