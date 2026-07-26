"""
Profile photo validation, processing and storage.

Avatars are handled separately from clinical uploads because the threat model
and the lifecycle differ. A clinical document is stored as received and served
only through an authorised, ownership-checked route; a profile photo is decoded,
stripped and re-encoded, then served as a static image.

Three properties matter here:

* **Nothing executable survives.** The bytes are decoded by Pillow and written
  back out as a fresh WEBP. A payload disguised as an image (polyglot, appended
  archive, embedded script) does not survive re-encoding, and EXIF — which can
  carry both scripts and a patient's GPS coordinates — is dropped with it.
* **The caller never influences the path.** Stored names are random; the
  submitted filename is used only to check the extension.
* **Only the authenticated user's own row is written.** Enforced by the service
  layer, which always targets `current_user.id`.
"""

from __future__ import annotations

import io
import logging
import os
import re
import secrets
from typing import Set

from fastapi import HTTPException, UploadFile, status

from app.core.upload import UPLOADS_ROOT

logger = logging.getLogger(__name__)

AVATARS_DIR = os.path.join(UPLOADS_ROOT, "avatars")
"""Profile photos only. Clinical documents never land here, which is what makes
serving this directory as static images acceptable."""

AVATAR_URL_PREFIX = "/uploads/avatars/"

MAX_AVATAR_BYTES = 5 * 1024 * 1024
"""5 MB. A profile photo has no legitimate reason to be larger, and the ceiling
is checked before the image is decoded."""

MAX_SOURCE_DIMENSION = 12_000
"""Guards against decompression bombs: a small file can declare an enormous
canvas that only materialises during decode."""

AVATAR_SIZE = 512
THUMBNAIL_SIZE = 128
WEBP_QUALITY = 85

ALLOWED_AVATAR_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

ALLOWED_AVATAR_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Stored names are generated, so the pattern that reads them back can be exact.
STORED_AVATAR_PATTERN = re.compile(r"^[a-f0-9]{32}(_thumb)?\.webp$")

_MAGIC_SIGNATURES = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
)


def _sniff_image_type(content: bytes) -> str | None:
    """
    Identify the image from its own bytes.

    `Content-Type` and the file extension are both supplied by the caller, so
    neither is evidence of what the file actually contains.
    """
    for signature, kind in _MAGIC_SIGNATURES:
        if content.startswith(signature):
            return kind
    # WEBP is a RIFF container: 'RIFF' <4-byte size> 'WEBP'
    if len(content) >= 12 and content[0:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp"
    return None


def _reject(detail: str, code: int = status.HTTP_400_BAD_REQUEST) -> None:
    logger.warning("[AVATAR_REJECTED] %s", detail)
    raise HTTPException(status_code=code, detail=detail)


async def read_and_validate_avatar(file: UploadFile) -> bytes:
    """
    Read an uploaded profile photo and prove it is a real, supported image.

    Returns the raw bytes. Raises `HTTPException` with a message suitable for
    display if anything about the submission is wrong.
    """
    if file.content_type not in ALLOWED_AVATAR_MIME_TYPES:
        _reject(
            f"Unsupported image type '{file.content_type}'. "
            "Upload a JPEG, PNG or WEBP photo."
        )

    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_AVATAR_EXTENSIONS:
        shown = extension or "none"
        _reject(
            f"Unsupported file extension '{shown}'. "
            "Upload a JPEG, PNG or WEBP photo."
        )

    content = await file.read()
    if not content:
        _reject("The uploaded file is empty.")

    if len(content) > MAX_AVATAR_BYTES:
        # `HTTP_413_CONTENT_TOO_LARGE` is the current spelling of 413; the
        # `HTTP_413_REQUEST_ENTITY_TOO_LARGE` used elsewhere in this codebase is
        # deprecated in Starlette and emits a warning on access.
        _reject(
            "Image exceeds the maximum size of "
            f"{MAX_AVATAR_BYTES // (1024 * 1024)} MB.",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )

    if _sniff_image_type(content) is None:
        # Declared an image type but the bytes say otherwise — a renamed
        # executable or script reaches exactly this branch.
        _reject("The uploaded file is not a valid JPEG, PNG or WEBP image.")

    return content


def process_avatar(content: bytes) -> tuple[bytes, bytes]:
    """
    Decode, normalise and re-encode a photo into a display image and a thumbnail.

    Applies the EXIF orientation before discarding the metadata, so a portrait
    taken on a phone is not stored sideways, and centre-crops to a square so the
    existing circular avatar frames are filled rather than letterboxed.

    Returns `(avatar_webp, thumbnail_webp)`.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        # verify() consumes the file object, so the image is opened twice: once
        # to check integrity, once to actually work with.
        with Image.open(io.BytesIO(content)) as probe:
            probe.verify()

        with Image.open(io.BytesIO(content)) as image:
            if (
                image.width > MAX_SOURCE_DIMENSION
                or image.height > MAX_SOURCE_DIMENSION
            ):
                _reject(
                    "Image dimensions are too large. "
                    f"Each side must be under {MAX_SOURCE_DIMENSION} pixels."
                )

            image = ImageOps.exif_transpose(image)
            image = image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB")

            # ImageOps.fit centre-crops to the target aspect ratio instead of
            # squashing the subject.
            avatar = ImageOps.fit(
                image, (AVATAR_SIZE, AVATAR_SIZE), method=Image.LANCZOS
            )
            thumbnail = ImageOps.fit(
                image, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), method=Image.LANCZOS
            )

            return _encode_webp(avatar), _encode_webp(thumbnail)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("[AVATAR_DECODE_FAILED] %s", exc)
        _reject("The image could not be read. It may be corrupt or unsupported.")


def _encode_webp(image) -> bytes:
    buffer = io.BytesIO()
    # No EXIF/ICC is carried over: only pixels are written.
    image.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=6)
    return buffer.getvalue()


def store_avatar(avatar_bytes: bytes, thumbnail_bytes: bytes) -> str:
    """
    Write the processed images and return the URL path for `avatar_url`.

    The name is random rather than derived from the user id, so the stored URL
    reveals nothing about who a photo belongs to and a replacement never
    collides with the file it supersedes.
    """
    os.makedirs(AVATARS_DIR, exist_ok=True)

    token = secrets.token_hex(16)
    avatar_name = f"{token}.webp"
    thumbnail_name = f"{token}_thumb.webp"

    for name, payload in ((avatar_name, avatar_bytes),
                          (thumbnail_name, thumbnail_bytes)):
        destination = os.path.abspath(os.path.join(AVATARS_DIR, name))
        # The name is generated, so this can only fail if the constants above
        # are changed carelessly — which is exactly when it should fail.
        if not destination.startswith(os.path.abspath(AVATARS_DIR) + os.sep):
            _reject("Invalid avatar destination path.")
        with open(destination, "wb") as handle:
            handle.write(payload)

    logger.info(
        "[AVATAR_STORED] %s (%d bytes) + thumbnail (%d bytes)",
        avatar_name, len(avatar_bytes), len(thumbnail_bytes),
    )
    return f"{AVATAR_URL_PREFIX}{avatar_name}"


def resolve_stored_avatar(filename: str) -> str | None:
    """
    Map a requested avatar filename to a path on disk.

    Returns None when the name is not one this module could have written, so a
    traversal attempt (`../../uploads/reports/x.pdf`) is rejected on the shape
    of the name before the filesystem is touched at all.
    """
    if not STORED_AVATAR_PATTERN.match(filename or ""):
        return None

    path = os.path.abspath(os.path.join(AVATARS_DIR, filename))
    if not path.startswith(os.path.abspath(AVATARS_DIR) + os.sep):
        return None
    return path if os.path.exists(path) else None


def delete_avatar_files(avatar_url: str | None) -> None:
    """
    Remove a superseded photo and its thumbnail.

    Best-effort: a profile update must never fail because an old file was
    already gone. Anything not written by `store_avatar` is left untouched.
    """
    if not avatar_url or not avatar_url.startswith(AVATAR_URL_PREFIX):
        return

    filename = avatar_url[len(AVATAR_URL_PREFIX):]
    if not STORED_AVATAR_PATTERN.match(filename):
        return

    stem = filename[: -len(".webp")]
    for name in (f"{stem}.webp", f"{stem}_thumb.webp"):
        path = os.path.abspath(os.path.join(AVATARS_DIR, name))
        if not path.startswith(os.path.abspath(AVATARS_DIR) + os.sep):
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("[AVATAR_CLEANUP_FAILED] %s: %s", name, exc)
