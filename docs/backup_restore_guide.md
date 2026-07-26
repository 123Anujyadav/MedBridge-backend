# Disaster Recovery, Database Backup & Migration Rollback Guide

Enterprise operational guide for maintaining database persistence, executing backups, performing restores, and managing Alembic schema migration rollbacks.

---

## 1. Database Backup Procedure

### Automated Execution
Run the automated backup script from the `Backend` directory:
```bash
python scripts/backup_db.py
```
Backups are timestamped and saved in the `./backups` directory:
- PostgreSQL: `./backups/backup_postgres_YYYYMMDD_HHMMSS.sql`
- SQLite: `./backups/backup_sqlite_YYYYMMDD_HHMMSS.db`

### Scheduled Cron Job (Production Linux)
Add to system crontab (`crontab -e`) to execute daily at 02:00 AM UTC:
```cron
0 2 * * * cd /var/www/backend && ./venv/bin/python scripts/backup_db.py >> /var/log/db_backup.log 2>&1
```

---

## 2. Database Restore Procedure

### Restoring from Backup File
Execute the restore script passing the path to a target backup file:
```bash
python scripts/restore_db.py backups/backup_postgres_20260718_020000.sql
```

---

## 3. Migration Management & Rollback Strategy

Alembic handles schema migrations and rollback tracking.

### View Migration History & Current Head
```bash
alembic history --verbose
alembic current
```

### Apply New Migrations
```bash
alembic upgrade head
```

### Rollback Strategy

#### Roll Back 1 Revision
```bash
alembic downgrade -1
```

#### Roll Back to Specific Revision
```bash
alembic downgrade <revision_id>
```

#### Emergency Roll Back to Base (Empty Schema)
```bash
alembic downgrade base
```
