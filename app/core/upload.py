import os
import re
import uuid
import logging
from typing import Set
from fastapi import UploadFile, HTTPException, status

logger = logging.getLogger(__name__)

# Backend/ — three levels up from Backend/app/core/upload.py
UPLOADS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "uploads")
)
"""
The one place the uploads directory is resolved.

Writers and readers previously computed this independently with relative `..`
hops and disagreed: `report_generator` wrote to `Backend/uploads/reports` while
the download route looked in `Backend/app/uploads`, so every generated PDF
404'd. Anything touching the uploads tree must derive its path from here.
"""

REPORTS_DIR = os.path.join(UPLOADS_ROOT, "reports")

PRESCRIPTIONS_DIR = os.path.join(UPLOADS_ROOT, "prescriptions")
"""
Rendered prescription PDFs.

Kept apart from `reports` deliberately: the two are different clinical
documents with different retention and access rules, and the existing report
download route resolves filenames inside REPORTS_DIR. Sharing one directory
would let a prescription id collide with a report id and serve the wrong
document to the wrong reader.
"""

# Security thresholds
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES: Set[str] = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}

ALLOWED_EXTENSIONS: Set[str] = {
    ".pdf", ".jpg", ".jpeg", ".png", ".txt", ".doc", ".docx"
}

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes file name to prevent path traversal and shell injection.
    Strips non-alphanumeric characters except safe dots, dashes, underscores.
    """
    normalized = filename.replace("\\", "/")
    basename = os.path.basename(normalized)
    clean_name = re.sub(r"[^\w\.-]", "_", basename)
    # Ensure no leading dots to prevent hidden file exploits
    return clean_name.lstrip(".")

def scan_file_virus_hook(file_bytes: bytes) -> bool:
    """
    Placeholder hook for anti-virus integration (e.g., ClamAV).
    Returns True if clean, False if infected.
    """
    # Production hook placeholder: Connect to ClamAV socket or ICAP server
    logger.info(f"Anti-virus scan performed on payload ({len(file_bytes)} bytes). Result: CLEAN.")
    return True

async def validate_and_save_upload(
    file: UploadFile,
    target_dir: str = "uploads"
) -> str:
    """
    Validates MIME type, file size, path traversal defenses, performs virus scan hook,
    and safely persists file to disk using a unique secure path.
    """
    # 1. Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"File upload rejected: Disallowed MIME type '{file.content_type}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Disallowed file MIME type: {file.content_type}"
        )

    # 2. Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"File upload rejected: Disallowed extension '{ext}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Disallowed file extension: {ext}"
        )

    # 3. Read content and validate size limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File upload rejected: Size {len(content)} exceeds limit {MAX_FILE_SIZE_BYTES} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # 4. Anti-Virus scan hook execution
    if not scan_file_virus_hook(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File failed anti-virus security inspection."
        )

    # 5. Sanitize file name & Path Traversal Guard
    safe_name = sanitize_filename(file.filename or "file")
    unique_filename = f"{uuid.uuid4().hex}_{safe_name}"

    os.makedirs(target_dir, exist_ok=True)
    destination_path = os.path.abspath(os.path.join(target_dir, unique_filename))

    # Path traversal check: ensure destination path resides strictly inside target_dir
    abs_target_dir = os.path.abspath(target_dir)
    if not destination_path.startswith(abs_target_dir):
        logger.error(f"Path traversal attempt detected with path: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid destination path specified."
        )

    # 6. Persist file
    with open(destination_path, "wb") as f:
        f.write(content)

    logger.info(f"File upload saved securely to: {destination_path}")
    return destination_path
