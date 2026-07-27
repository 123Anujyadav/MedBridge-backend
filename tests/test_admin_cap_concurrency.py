"""
The administrator cap, proved against a real PostgreSQL server.

These do not run on the suite's SQLite database, and that is the point. The
cap is enforced by a trigger and an advisory lock, neither of which SQLite has,
so a test that ran there would prove nothing about the thing that broke. The
original defect — an unlocked `SELECT COUNT(*)` inside the trigger — was
invisible to every existing test for exactly that reason.

Skipped automatically when no PostgreSQL server is reachable, so the suite
stays runnable on a laptop with nothing installed. Run them with:

    ALEMBIC_TEST_DSN=postgresql://... pytest tests/test_admin_cap_concurrency.py

Each test leaves the database as it found it: rows are namespaced `cap.%@x.com`
and deleted in a finally block, and no pre-existing account is modified except
through an explicitly restored toggle.
"""

import asyncio
import os
import uuid

import pytest

asyncpg = pytest.importorskip("asyncpg")

pytestmark = pytest.mark.asyncio

DSN = os.getenv(
    "ADMIN_CAP_TEST_DSN",
    "postgresql://postgres:postgrespassword@localhost:5432/aronofy_db",
)

MAX_ADMINS = 2
CONCURRENCY = 25
"""
Writers per stampede.

High enough that an unlocked count loses reliably — the original bug reproduced
at two — and low enough to stay well inside PostgreSQL's default
`max_connections` of 100 alongside the application's own pool.
"""


async def _connect():
    try:
        return await asyncio.wait_for(asyncpg.connect(DSN), timeout=5)
    except Exception as exc:  # server absent, wrong password, database missing
        pytest.skip(f"PostgreSQL not reachable for concurrency tests: {exc}")


async def _live_admins(conn) -> int:
    return await conn.fetchval(
        "SELECT COUNT(*) FROM users "
        "WHERE role='admin' AND is_active AND deleted_at IS NULL"
    )


async def _cleanup(conn) -> None:
    await conn.execute("DELETE FROM users WHERE email LIKE 'cap.%@x.com'")


@pytest.fixture
async def pg():
    """A connection, with the trigger present and the namespace clean."""
    conn = await _connect()
    installed = await conn.fetchval(
        "SELECT COUNT(*) FROM pg_trigger "
        "WHERE tgname = 'users_admin_account_cap_trigger'"
    )
    if not installed:
        pytest.skip("admin cap trigger not installed; run `alembic upgrade head`")

    await _cleanup(conn)
    try:
        yield conn
    finally:
        await _cleanup(conn)
        await conn.close()


@pytest.fixture
async def free_slots(pg):
    """
    Retire enough existing administrators to leave exactly `n` slots open, and
    put every one of them back afterwards.

    Teardown deletes the test's own administrators *before* restoring the real
    ones. The other order does not work: the cap is real, so reinstating an
    original while a test's administrator still occupies its slot is refused by
    the trigger, the restore is lost, and every later test in the file starts
    from a platform that is quietly one administrator short.
    """
    retired: list = []

    async def _make(n: int) -> None:
        live = await pg.fetch(
            "SELECT id FROM users "
            "WHERE role='admin' AND is_active AND deleted_at IS NULL"
        )
        to_retire = max(0, len(live) - (MAX_ADMINS - n))
        for row in live[:to_retire]:
            await pg.execute(
                "UPDATE users SET is_active=false WHERE id=$1", row["id"])
            retired.append(row["id"])

    try:
        yield _make
    finally:
        await _cleanup(pg)
        for uid in retired:
            await pg.execute("UPDATE users SET is_active=true WHERE id=$1", uid)


async def _race(op, count: int):
    """Run `count` copies of `op`, all released from a barrier together."""
    barrier = asyncio.Barrier(count)

    async def run(i):
        conn = await asyncpg.connect(DSN)
        try:
            tx = conn.transaction()
            await tx.start()
            await barrier.wait()          # every writer is inside a transaction
            try:
                await op(conn, i)
                await tx.commit()
                return "committed"
            except Exception as exc:
                await tx.rollback()
                return ("blocked" if "limit reached" in str(exc)
                        else f"error:{exc}")
        finally:
            await conn.close()

    return await asyncio.gather(*[run(i) for i in range(count)])


async def _insert_admin(conn, i):
    await conn.execute(
        "INSERT INTO users (id,email,hashed_password,role,is_active,is_verified) "
        "VALUES ($1,$2,'x','admin',true,true)",
        uuid.uuid4(), f"cap.race{i}@x.com",
    )


class TestConcurrentCreation:
    async def test_two_writers_one_slot_yields_one_administrator(
        self, pg, free_slots
    ):
        """
        The exact shape the audit reproduced: two transactions, one free slot,
        both committed, three administrators.
        """
        await free_slots(1)
        results = await _race(_insert_admin, 2)

        assert results.count("committed") == 1, results
        assert await _live_admins(pg) == MAX_ADMINS

    async def test_stampede_against_one_slot_admits_exactly_one(
        self, pg, free_slots
    ):
        await free_slots(1)
        results = await _race(_insert_admin, CONCURRENCY)

        assert results.count("committed") == 1, results
        assert results.count("blocked") == CONCURRENCY - 1, results
        assert await _live_admins(pg) == MAX_ADMINS

    async def test_stampede_against_a_full_platform_admits_nobody(
        self, pg, free_slots
    ):
        await free_slots(0)
        before = await _live_admins(pg)
        results = await _race(_insert_admin, CONCURRENCY)

        assert results.count("committed") == 0, results
        assert await _live_admins(pg) == before

    async def test_two_free_slots_admit_exactly_two(self, pg, free_slots):
        """The cap is a ceiling, not a freeze — it must still let people in."""
        await free_slots(2)
        results = await _race(_insert_admin, CONCURRENCY)

        assert results.count("committed") == 2, results
        assert await _live_admins(pg) == MAX_ADMINS


class TestConcurrentRestore:
    """
    Reactivating a retired administrator is a creation as far as the cap is
    concerned, and raced identically before the fix.
    """

    async def test_concurrent_reactivation_cannot_exceed_the_cap(
        self, pg, free_slots
    ):
        ids = []
        for i in range(CONCURRENCY):
            uid = uuid.uuid4()
            await pg.execute(
                "INSERT INTO users (id,email,hashed_password,role,is_active,is_verified) "
                "VALUES ($1,$2,'x','admin',false,true)", uid, f"cap.rest{i}@x.com")
            ids.append(uid)
        await free_slots(1)

        async def restore(conn, i):
            await conn.execute(
                "UPDATE users SET is_active=true WHERE id=$1", ids[i])

        results = await _race(restore, CONCURRENCY)
        assert results.count("committed") == 1, results
        assert await _live_admins(pg) == MAX_ADMINS

    async def test_undeleting_a_soft_deleted_admin_is_also_capped(
        self, pg, free_slots
    ):
        ids = []
        for i in range(10):
            uid = uuid.uuid4()
            await pg.execute(
                "INSERT INTO users (id,email,hashed_password,role,is_active,"
                "is_verified,deleted_at) VALUES ($1,$2,'x','admin',true,true,now())",
                uid, f"cap.undel{i}@x.com")
            ids.append(uid)
        await free_slots(0)

        async def undelete(conn, i):
            await conn.execute(
                "UPDATE users SET deleted_at=NULL WHERE id=$1", ids[i])

        results = await _race(undelete, 10)
        assert results.count("committed") == 0, results
        assert await _live_admins(pg) == MAX_ADMINS


class TestCapMechanics:
    async def test_rollback_frees_the_slot_and_releases_the_lock(
        self, pg, free_slots
    ):
        """
        A transaction-scoped advisory lock must not survive its transaction. If
        it leaked, the first rolled-back attempt would wedge every later one.
        """
        await free_slots(1)

        conn = await asyncpg.connect(DSN)
        try:
            tx = conn.transaction()
            await tx.start()
            await _insert_admin(conn, 900)
            await tx.rollback()
        finally:
            await conn.close()

        results = await _race(_insert_admin, 1)
        assert results == ["committed"], results
        assert await _live_admins(pg) == MAX_ADMINS

    async def test_a_retired_admin_frees_a_slot(self, pg, free_slots):
        await free_slots(0)
        assert (await _race(_insert_admin, 1))[0] == "blocked"

        victim = await pg.fetchval(
            "SELECT id FROM users "
            "WHERE role='admin' AND is_active AND deleted_at IS NULL LIMIT 1")
        await pg.execute("UPDATE users SET is_active=false WHERE id=$1", victim)
        try:
            assert (await _race(_insert_admin, 1))[0] == "committed"
        finally:
            # Same ordering rule as the fixture: clear the seat before the
            # original tries to sit back down in it.
            await _cleanup(pg)
            await pg.execute("UPDATE users SET is_active=true WHERE id=$1", victim)

    async def test_editing_an_existing_admin_is_never_refused(self, pg):
        """
        A live administrator changing their own email must not trip the cap —
        the count is unchanged and the trigger has to notice that.
        """
        victim = await pg.fetchval(
            "SELECT id FROM users "
            "WHERE role='admin' AND is_active AND deleted_at IS NULL LIMIT 1")
        original = await pg.fetchval(
            "SELECT is_verified FROM users WHERE id=$1", victim)
        try:
            await pg.execute(
                "UPDATE users SET is_verified = NOT is_verified WHERE id=$1", victim)
        finally:
            await pg.execute(
                "UPDATE users SET is_verified=$2 WHERE id=$1", victim, original)

    async def test_non_admin_writes_are_untouched_by_the_cap(self, pg):
        """
        Patients and doctors must not be serialised behind the administrator
        lock — it would turn every signup into a queue of one.
        """
        async def add_patient(conn, i):
            await conn.execute(
                "INSERT INTO users (id,email,hashed_password,role,is_active,"
                "is_verified) VALUES ($1,$2,'x','patient',true,true)",
                uuid.uuid4(), f"cap.pat{i}@x.com")

        results = await _race(add_patient, CONCURRENCY)
        assert results.count("committed") == CONCURRENCY, results


class TestDoctorInvariantsInPostgres:
    async def test_a_verified_doctor_cannot_lose_its_doctor_id(self, pg):
        victim = await pg.fetchval(
            "SELECT id FROM doctors WHERE verification_status='verified' "
            "AND doctor_code IS NOT NULL LIMIT 1")
        if victim is None:
            pytest.skip("no verified doctor to test against")

        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await pg.execute(
                "UPDATE doctors SET doctor_code=NULL WHERE id=$1", victim)

    async def test_an_unapproved_doctor_may_hold_no_doctor_id(self, pg):
        """
        The constraint must not block the backfill path or an unapproved
        clinician whose ID has not been issued yet.
        """
        victim = await pg.fetchval(
            "SELECT id FROM doctors WHERE verification_status <> 'verified' LIMIT 1")
        if victim is None:
            pytest.skip("no unapproved doctor to test against")

        original = await pg.fetchval(
            "SELECT doctor_code FROM doctors WHERE id=$1", victim)
        try:
            await pg.execute(
                "UPDATE doctors SET doctor_code=NULL WHERE id=$1", victim)
        finally:
            await pg.execute(
                "UPDATE doctors SET doctor_code=$2 WHERE id=$1", victim, original)

    async def test_doctor_code_format_is_enforced_by_the_column(self, pg):
        victim = await pg.fetchval("SELECT id FROM doctors LIMIT 1")
        original = await pg.fetchval(
            "SELECT doctor_code FROM doctors WHERE id=$1", victim)
        try:
            for bad in ("lowercas", "SHORT", "HAS-DASH"):
                with pytest.raises(asyncpg.exceptions.CheckViolationError):
                    await pg.execute(
                        "UPDATE doctors SET doctor_code=$2 WHERE id=$1", victim, bad)
        finally:
            await pg.execute(
                "UPDATE doctors SET doctor_code=$2 WHERE id=$1", victim, original)

    async def test_doctor_codes_are_unique(self, pg):
        rows = await pg.fetch(
            "SELECT id, doctor_code FROM doctors WHERE doctor_code IS NOT NULL LIMIT 2")
        if len(rows) < 2:
            pytest.skip("need two doctors")

        original = rows[1]["doctor_code"]
        try:
            with pytest.raises(asyncpg.exceptions.UniqueViolationError):
                await pg.execute(
                    "UPDATE doctors SET doctor_code=$2 WHERE id=$1",
                    rows[1]["id"], rows[0]["doctor_code"])
        finally:
            await pg.execute(
                "UPDATE doctors SET doctor_code=$2 WHERE id=$1",
                rows[1]["id"], original)
