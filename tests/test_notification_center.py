"""
Tests for the real-time notification centre.

The properties that matter most:

* **Ownership.** A notification belongs to exactly one user. Nobody can read,
  read-mark, dismiss or probe for another user's notifications — and a missing
  id and someone else's id must be indistinguishable, or the endpoint becomes
  an enumeration oracle.
* **No duplicates.** Every notification carries a `dedupe_key`. A retried
  request or two code paths observing one event must not produce two cards.
* **Critical first.** An urgent alert that has scrolled out of view has failed
  at its job, so priority beats recency in the ordering.
* **Delivery is targeted.** Notifications go out over `send_personal_message`,
  never `broadcast`, so a clinical alert cannot reach another doctor's socket.
* **No fabricated notifications.** Rows exist only where an event occurred.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text

from app.core.security import get_password_hash
from app.models.audit import AuditLog
from app.models.case import Case
from app.models.doctor import Doctor
from app.models.notification import NotificationItem
from app.models.patient import Patient
from app.models.user import User
from app.services.notifications import notification_service

pytestmark = pytest.mark.asyncio

PW = "password123"
DOC_A = "nc.doca@aronofy.com"
DOC_B = "nc.docb@aronofy.com"
PAT_A = "nc.pata@aronofy.com"


@pytest.fixture
async def estate(db):
    await db.execute(text("PRAGMA foreign_keys = OFF;"))
    for table in ("audit_logs", "notifications", "cases", "doctors", "patients", "users"):
        await db.execute(text(f"DELETE FROM {table};"))
    await db.execute(text("PRAGMA foreign_keys = ON;"))
    await db.flush()

    ids: dict[str, Any] = {}
    for k, e, r in (("doc_a", DOC_A, "doctor"), ("doc_b", DOC_B, "doctor"),
                    ("pat_a", PAT_A, "patient")):
        u = User(email=e, hashed_password=get_password_hash(PW), role=r, is_verified=True)
        db.add(u); await db.flush(); ids[k] = u.id

    db.add(Doctor(id=ids["doc_a"], first_name="Asha", last_name="Rao", phone="+911",
                  specialty="Neurology", hospital_name="Central",
                  license_number="LIC-NC-A", verification_status="verified"))
    db.add(Doctor(id=ids["doc_b"], first_name="Vikram", last_name="Sen", phone="+912",
                  specialty="Cardiology", hospital_name="East",
                  license_number="LIC-NC-B", verification_status="verified"))
    db.add(Patient(id=ids["pat_a"], first_name="Meera", last_name="Iyer", phone="+913",
                   date_of_birth="1992-03-14", gender="female",
                   allergies=[], chronic_conditions=[], medications=[]))
    await db.flush()

    case = Case(patient_id=ids["pat_a"], patient_name="Meera Iyer", patient_age=34,
                patient_gender="female", doctor_id=ids["doc_a"],
                doctor_name="Dr. Asha Rao", specialty="Neurology",
                symptom_summary="Headache.", urgency_level="high", status="routed",
                ai_extracted_symptoms=[], ai_confidence_score=0.0,
                attachments=[], notes="")
    unassigned = Case(patient_id=ids["pat_a"], patient_name="Meera Iyer",
                      patient_age=34, patient_gender="female", doctor_id=None,
                      doctor_name=None, specialty="General Medicine",
                      symptom_summary="Unrouted.", urgency_level="low",
                      status="ai_processing", ai_extracted_symptoms=[],
                      ai_confidence_score=0.0, attachments=[], notes="")
    db.add_all([case, unassigned])
    await db.flush()
    ids["case"], ids["unassigned"] = case.id, unassigned.id
    await db.commit()
    return ids


async def _login(client: AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _centre(client, headers, **params) -> dict:
    r = await client.get("/api/v1/shared/notifications/center",
                         params=params, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


async def _seed_notification(db, user_id, **kwargs) -> NotificationItem:
    defaults = dict(
        category="case", type="case_assigned", title="New Patient Case Assigned",
        message="Meera Iyer — Neurology.", priority="medium",
    )
    defaults.update(kwargs)
    item = await notification_service.notify(db, user_id=user_id, **defaults)
    await db.commit()
    return item


class TestCreationAndDedupe:
    async def test_creates_a_notification(self, db, estate):
        item = await _seed_notification(db, estate["doc_a"],
                                        dedupe_key="case_assigned:1")
        assert item is not None
        assert item.category == "case"
        assert item.read is False

    async def test_duplicate_dedupe_key_is_suppressed(self, db, estate):
        first = await _seed_notification(db, estate["doc_a"], dedupe_key="dup:1")
        second = await notification_service.notify(
            db, user_id=estate["doc_a"], category="case", type="case_assigned",
            title="Same event again", message="m", dedupe_key="dup:1",
        )
        await db.commit()

        assert first is not None
        assert second is None
        count = await db.scalar(
            select(NotificationItem.id).where(NotificationItem.dedupe_key == "dup:1")
        )
        assert count is not None

    async def test_same_key_for_a_different_user_is_allowed(self, db, estate):
        """Dedupe is per recipient; two doctors may both need the same alert."""
        a = await _seed_notification(db, estate["doc_a"], dedupe_key="shared:1")
        b = await _seed_notification(db, estate["doc_b"], dedupe_key="shared:1")
        assert a is not None and b is not None

    async def test_unassigned_case_notifies_nobody(self, db, estate):
        """An unrouted case must not land in an arbitrary doctor's inbox."""
        case = await db.get(Case, estate["unassigned"])
        result = await notification_service.notify_case_doctor(
            db, case=case, category="case", type="case_assigned",
            title="t", message="m",
        )
        await db.commit()
        assert result is None

    async def test_creation_is_audited(self, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="audit:1",
                                 case_id=estate["case"])
        rows = (await db.execute(
            select(AuditLog).where(AuditLog.event_type == "notification.created")
        )).scalars().all()
        assert len(rows) >= 1
        assert rows[0].resource == "Notification"


class TestCenterFeed:
    async def test_lists_only_the_callers_notifications(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="mine:1",
                                 title="MINE")
        await _seed_notification(db, estate["doc_b"], dedupe_key="theirs:1",
                                 title="THEIRS")

        headers = await _login(client, DOC_A)
        body = await _centre(client, headers)
        titles = {n["title"] for n in body["notifications"]}
        assert "MINE" in titles
        assert "THEIRS" not in titles

    async def test_critical_sorts_above_newer_items(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="low:1",
                                 title="LOW", priority="low")
        await _seed_notification(db, estate["doc_a"], dedupe_key="crit:1",
                                 title="CRITICAL", priority="critical")
        # A newer low-priority item must still sit below the critical one.
        await _seed_notification(db, estate["doc_a"], dedupe_key="low:2",
                                 title="NEWER_LOW", priority="low")

        headers = await _login(client, DOC_A)
        body = await _centre(client, headers)
        assert body["notifications"][0]["title"] == "CRITICAL"

    async def test_unread_and_critical_counts(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="c1",
                                 priority="critical")
        await _seed_notification(db, estate["doc_a"], dedupe_key="c2",
                                 priority="low")
        headers = await _login(client, DOC_A)
        body = await _centre(client, headers)
        assert body["unread_count"] == 2
        assert body["critical_count"] == 1

    async def test_grouping_collapses_similar_items(self, client, db, estate):
        for i in range(3):
            await _seed_notification(db, estate["doc_a"], dedupe_key=f"g{i}",
                                     group_key="report_uploaded",
                                     type="report_uploaded",
                                     title="Patient Uploaded Reports")
        headers = await _login(client, DOC_A)
        groups = (await _centre(client, headers))["groups"]
        assert any(g["group_key"] == "report_uploaded" and g["count"] == 3
                   for g in groups)

    async def test_a_single_item_is_not_reported_as_a_group(self, client, db, estate):
        """Presenting one notification as a group overstates the volume."""
        await _seed_notification(db, estate["doc_a"], dedupe_key="solo",
                                 group_key="solo_event")
        headers = await _login(client, DOC_A)
        groups = (await _centre(client, headers))["groups"]
        assert all(g["group_key"] != "solo_event" for g in groups)

    async def test_card_carries_workflow_context(self, client, db, estate):
        await _seed_notification(
            db, estate["doc_a"], dedupe_key="ctx:1", case_id=estate["case"],
            patient_id=estate["pat_a"], patient_name="Meera Iyer",
            action_url="/doctor/cases?case=x", action_label="Open Case",
        )
        headers = await _login(client, DOC_A)
        card = (await _centre(client, headers))["notifications"][0]

        assert card["patient_name"] == "Meera Iyer"
        assert card["case_id"] == str(estate["case"])
        assert card["case_short_id"] == str(estate["case"]).split("-")[0]
        assert card["action_label"] == "Open Case"

    async def test_no_case_means_no_short_id(self, client, db, estate):
        """A system alert has no case; nothing may be inferred for it."""
        await _seed_notification(db, estate["doc_a"], dedupe_key="sys:1",
                                 category="system", type="maintenance")
        headers = await _login(client, DOC_A)
        card = (await _centre(client, headers))["notifications"][0]
        assert card["case_id"] is None
        assert card["case_short_id"] is None
        assert card["patient_id"] is None


class TestFiltersAndSearch:
    async def test_category_filter(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="f1",
                                 category="ai", type="ai_ready", title="AI ONE")
        await _seed_notification(db, estate["doc_a"], dedupe_key="f2",
                                 category="appointment", type="appt", title="APPT")
        headers = await _login(client, DOC_A)
        body = await _centre(client, headers, category="ai")
        assert [n["title"] for n in body["notifications"]] == ["AI ONE"]

    async def test_unread_only_filter(self, client, db, estate):
        item = await _seed_notification(db, estate["doc_a"], dedupe_key="u1",
                                        title="READ_ME")
        await _seed_notification(db, estate["doc_a"], dedupe_key="u2",
                                 title="STILL_UNREAD")
        headers = await _login(client, DOC_A)
        await client.put(f"/api/v1/shared/notifications/{item.id}/read", headers=headers)

        body = await _centre(client, headers, unread_only=True)
        titles = {n["title"] for n in body["notifications"]}
        assert titles == {"STILL_UNREAD"}

    async def test_critical_only_filter(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="k1",
                                 priority="critical", title="CRIT")
        await _seed_notification(db, estate["doc_a"], dedupe_key="k2",
                                 priority="low", title="LOW")
        headers = await _login(client, DOC_A)
        body = await _centre(client, headers, critical_only=True)
        assert [n["title"] for n in body["notifications"]] == ["CRIT"]

    async def test_search_matches_patient_name(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="s1",
                                 patient_name="Meera Iyer", title="FOUND",
                                 message="Case assigned.")
        await _seed_notification(db, estate["doc_a"], dedupe_key="s2",
                                 patient_name="Someone Else", title="OTHER",
                                 message="Case assigned.")
        headers = await _login(client, DOC_A)
        body = await _centre(client, headers, search="Meera")
        assert [n["title"] for n in body["notifications"]] == ["FOUND"]

    async def test_search_with_no_match_is_empty(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="s3")
        headers = await _login(client, DOC_A)
        assert (await _centre(client, headers, search="zzz-none"))["total"] == 0

    async def test_date_range_excludes_out_of_window(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="d1")
        headers = await _login(client, DOC_A)
        body = await _centre(client, headers,
                             date_from="2000-01-01", date_to="2000-01-02")
        assert body["total"] == 0


class TestReadAndArchive:
    async def test_mark_read(self, client, db, estate):
        item = await _seed_notification(db, estate["doc_a"], dedupe_key="r1")
        headers = await _login(client, DOC_A)
        r = await client.put(f"/api/v1/shared/notifications/{item.id}/read",
                             headers=headers)
        assert r.status_code == 200
        assert (await _centre(client, headers))["unread_count"] == 0

    async def test_mark_all_read(self, client, db, estate):
        for i in range(3):
            await _seed_notification(db, estate["doc_a"], dedupe_key=f"a{i}")
        headers = await _login(client, DOC_A)
        r = await client.put("/api/v1/shared/notifications/read-all", headers=headers)
        assert r.status_code == 200 and r.json()["updated"] == 3
        assert (await _centre(client, headers))["unread_count"] == 0

    async def test_mark_selected_read(self, client, db, estate):
        first = await _seed_notification(db, estate["doc_a"], dedupe_key="sel1")
        await _seed_notification(db, estate["doc_a"], dedupe_key="sel2")
        headers = await _login(client, DOC_A)
        r = await client.put("/api/v1/shared/notifications/read-selected",
                             json={"notification_ids": [str(first.id)]}, headers=headers)
        assert r.status_code == 200 and r.json()["updated"] == 1
        assert (await _centre(client, headers))["unread_count"] == 1

    async def test_archive_hides_from_the_default_feed(self, client, db, estate):
        item = await _seed_notification(db, estate["doc_a"], dedupe_key="ar1",
                                        title="DISMISSED")
        headers = await _login(client, DOC_A)
        r = await client.put(f"/api/v1/shared/notifications/{item.id}/archive",
                             headers=headers)
        assert r.status_code == 200

        assert (await _centre(client, headers))["total"] == 0
        archived = await _centre(client, headers, include_archived=True)
        assert [n["title"] for n in archived["notifications"]] == ["DISMISSED"]

    async def test_opened_is_audited_separately_from_read(self, client, db, estate):
        item = await _seed_notification(db, estate["doc_a"], dedupe_key="op1",
                                        case_id=estate["case"])
        headers = await _login(client, DOC_A)
        r = await client.put(f"/api/v1/shared/notifications/{item.id}/opened",
                             headers=headers)
        assert r.status_code == 200

        events = {row.event_type for row in (await db.execute(
            select(AuditLog).where(AuditLog.resource == "Notification")
        )).scalars().all()}
        assert "notification.opened" in events
        assert "notification.read" in events


class TestOwnership:
    async def test_cannot_read_mark_another_users_notification(
        self, client, db, estate
    ):
        item = await _seed_notification(db, estate["doc_b"], dedupe_key="own1")
        headers = await _login(client, DOC_A)
        r = await client.put(f"/api/v1/shared/notifications/{item.id}/read",
                             headers=headers)
        assert r.status_code == 404

    async def test_cannot_archive_another_users_notification(self, client, db, estate):
        item = await _seed_notification(db, estate["doc_b"], dedupe_key="own2")
        headers = await _login(client, DOC_A)
        r = await client.put(f"/api/v1/shared/notifications/{item.id}/archive",
                             headers=headers)
        assert r.status_code == 404

    async def test_foreign_and_missing_ids_are_indistinguishable(
        self, client, db, estate
    ):
        """Otherwise the endpoint becomes a notification-enumeration oracle."""
        item = await _seed_notification(db, estate["doc_b"], dedupe_key="own3")
        headers = await _login(client, DOC_A)

        foreign = await client.put(f"/api/v1/shared/notifications/{item.id}/read",
                                   headers=headers)
        missing = await client.put(f"/api/v1/shared/notifications/{uuid.uuid4()}/read",
                                   headers=headers)
        assert foreign.status_code == missing.status_code == 404

    async def test_mark_selected_ignores_ids_the_caller_does_not_own(
        self, client, db, estate
    ):
        mine = await _seed_notification(db, estate["doc_a"], dedupe_key="own4")
        theirs = await _seed_notification(db, estate["doc_b"], dedupe_key="own5")
        headers = await _login(client, DOC_A)

        r = await client.put(
            "/api/v1/shared/notifications/read-selected",
            json={"notification_ids": [str(mine.id), str(theirs.id)]},
            headers=headers,
        )
        assert r.json()["updated"] == 1

        still_unread = await db.scalar(
            select(NotificationItem.read).where(NotificationItem.id == theirs.id)
        )
        assert still_unread is False

    async def test_mark_all_read_touches_only_the_caller(self, client, db, estate):
        await _seed_notification(db, estate["doc_a"], dedupe_key="own6")
        theirs = await _seed_notification(db, estate["doc_b"], dedupe_key="own7")
        headers = await _login(client, DOC_A)
        await client.put("/api/v1/shared/notifications/read-all", headers=headers)

        still_unread = await db.scalar(
            select(NotificationItem.read).where(NotificationItem.id == theirs.id)
        )
        assert still_unread is False


class TestDeliveryChannel:
    async def test_delivery_is_targeted_never_broadcast(self, db, estate, monkeypatch):
        """
        A clinical alert on every open socket would leak PHI across doctors.

        This asserts the transport choice, not just the payload.
        """
        from app.core import websocket as ws_module

        sent: list[tuple[dict, str]] = []
        broadcasts: list[dict] = []

        async def fake_personal(message, user_id):
            sent.append((message, user_id))

        async def fake_broadcast(message):
            broadcasts.append(message)

        monkeypatch.setattr(ws_module.websocket_manager, "send_personal_message",
                            fake_personal)
        monkeypatch.setattr(ws_module.websocket_manager, "broadcast", fake_broadcast)

        await _seed_notification(db, estate["doc_a"], dedupe_key="ws1",
                                 priority="critical")

        assert len(sent) == 1
        message, user_id = sent[0]
        assert user_id == str(estate["doc_a"])
        assert message["type"] == "NOTIFICATION_CREATED"
        assert broadcasts == []

    async def test_delivery_failure_does_not_break_creation(
        self, db, estate, monkeypatch
    ):
        """A notification is a side effect; it must not fail clinical work."""
        from app.core import websocket as ws_module

        async def boom(message, user_id):
            raise RuntimeError("socket down")

        monkeypatch.setattr(ws_module.websocket_manager, "send_personal_message", boom)

        item = await _seed_notification(db, estate["doc_a"], dedupe_key="ws2")
        assert item is not None
        assert item.delivered_at is None


class TestPagination:
    async def test_paginates_and_reports_has_more(self, client, db, estate):
        for i in range(5):
            await _seed_notification(db, estate["doc_a"], dedupe_key=f"p{i}")
        headers = await _login(client, DOC_A)

        first = await _centre(client, headers, limit=2)
        assert first["returned"] == 2 and first["has_more"] is True

        full = await _centre(client, headers, limit=100)
        assert full["has_more"] is False and full["returned"] == full["total"] == 5
