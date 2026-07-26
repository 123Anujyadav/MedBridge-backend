import os
import sys
import subprocess
import datetime
from app.core.config import settings

def backup_database():
    """
    Automated Database Backup Script for MedBridge Enterprise Stack.
    Supports PostgreSQL pg_dump backups and SQLite file backups.
    """
    db_url = settings.DATABASE_URL
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(os.getcwd(), "backups")
    os.makedirs(backup_dir, exist_ok=True)

    print(f"[{datetime.datetime.now().isoformat()}] Starting database backup routine...")

    if "sqlite" in db_url:
        import shutil
        db_file = db_url.split(":///")[-1]
        if os.path.exists(db_file):
            target_path = os.path.join(backup_dir, f"backup_sqlite_{timestamp}.db")
            shutil.copy2(db_file, target_path)
            print(f"SQLite database backed up successfully to: {target_path}")
            return target_path
        else:
            print(f"SQLite database file {db_file} not found.", file=sys.stderr)
            sys.exit(1)

    elif "postgresql" in db_url:
        target_path = os.path.join(backup_dir, f"backup_postgres_{timestamp}.sql")
        # Extract connection details from settings.DATABASE_URL
        # pg_dump command execution
        cmd = f"pg_dump {db_url} -F p -f {target_path}"
        try:
            res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"PostgreSQL database backed up successfully to: {target_path}")
            return target_path
        except subprocess.CalledProcessError as e:
            print(f"PostgreSQL pg_dump failed: {e.stderr}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unsupported database dialect in URL: {db_url}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    backup_database()
