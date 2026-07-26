import os
import sys
import subprocess
from app.core.config import settings

def restore_database(backup_file_path: str):
    """
    Automated Database Restore Script for MedBridge Enterprise Stack.
    """
    if not os.path.exists(backup_file_path):
        print(f"Error: Specified backup file does not exist: {backup_file_path}", file=sys.stderr)
        sys.exit(1)

    db_url = settings.DATABASE_URL
    print(f"Restoring database from backup: {backup_file_path}")

    if "sqlite" in db_url:
        import shutil
        db_file = db_url.split(":///")[-1]
        shutil.copy2(backup_file_path, db_file)
        print(f"SQLite database restored successfully from: {backup_file_path}")

    elif "postgresql" in db_url:
        cmd = f"psql {db_url} -f {backup_file_path}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"PostgreSQL database restored successfully from: {backup_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"PostgreSQL restore failed: {e.stderr}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unsupported database dialect in URL: {db_url}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/restore_db.py <path_to_backup_file>")
        sys.exit(1)
    restore_database(sys.argv[1])
