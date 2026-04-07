"""
Cloudinary integration for persistent photo storage.

When CLOUDINARY_CLOUD_NAME / API_KEY / API_SECRET are set, all passport photos
are uploaded to Cloudinary instead of the local filesystem.  Capture photos used
for face-matching are still written to a temp file for the duration of a single
request and then deleted — they never need to be persistent.

If the three Cloudinary env vars are absent the module is disabled and the
callers fall back to local-disk storage automatically.
"""

import logging
import os
import tempfile
import uuid
from contextlib import contextmanager
from typing import Generator

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Initialise cloudinary SDK (only when credentials are present)
# --------------------------------------------------------------------------- #
CLOUDINARY_ENABLED: bool = bool(
    os.getenv("CLOUDINARY_CLOUD_NAME")
    and os.getenv("CLOUDINARY_API_KEY")
    and os.getenv("CLOUDINARY_API_SECRET")
)

if CLOUDINARY_ENABLED:
    import cloudinary
    import cloudinary.uploader

    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    logger.info("Cloudinary storage enabled (cloud: %s)", os.getenv("CLOUDINARY_CLOUD_NAME"))
else:
    logger.info("Cloudinary not configured — using local disk storage.")


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def upload_passport(content: bytes, employee_id: str) -> str:
    """
    Upload raw image bytes to Cloudinary.

    Returns the HTTPS ``secure_url`` of the stored photo.
    Raises ``RuntimeError`` when Cloudinary is not configured.
    """
    if not CLOUDINARY_ENABLED:
        raise RuntimeError("Cloudinary is not configured.")

    public_id = f"attendance/passports/{employee_id}_{uuid.uuid4().hex[:8]}"
    result = cloudinary.uploader.upload(
        content,
        public_id=public_id,
        resource_type="image",
        overwrite=True,
        # Auto-resize large images to save bandwidth; quality=auto compresses
        transformation=[{"width": 800, "crop": "limit", "quality": "auto:good"}],
    )
    secure_url: str = result["secure_url"]
    logger.info("Passport uploaded to Cloudinary: %s", secure_url)
    return secure_url


def delete_photo(secure_url: str) -> None:
    """
    Delete a Cloudinary-hosted photo by its ``secure_url``.

    Errors are logged but not re-raised — a failed delete must not break the
    primary request flow.
    """
    if not CLOUDINARY_ENABLED:
        return
    try:
        # URL pattern:  …/image/upload[/v<version>]/public/id.ext
        parts = secure_url.split("/upload/")
        if len(parts) != 2:
            logger.warning("Cannot parse Cloudinary URL for deletion: %s", secure_url)
            return

        path = parts[1]
        # Strip optional version segment  (e.g. "v1710000000/...")
        if path.startswith("v") and "/" in path:
            first, rest = path.split("/", 1)
            if first[1:].isdigit():
                path = rest

        public_id = os.path.splitext(path)[0]
        cloudinary.uploader.destroy(public_id, resource_type="image")
        logger.info("Deleted Cloudinary photo public_id=%s", public_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete Cloudinary photo '%s': %s", secure_url, exc)


def is_remote_url(path: str) -> bool:
    """Return True when *path* is an HTTP/HTTPS URL (i.e. a Cloudinary link)."""
    return path.startswith("http://") or path.startswith("https://")


@contextmanager
def resolve_photo_path(passport_path: str) -> Generator[str, None, None]:
    """
    Context manager that yields a local file path suitable for OpenCV.

    * If ``passport_path`` is already a local path — yields it directly;
      no cleanup needed.
    * If ``passport_path`` is a remote URL — downloads it to a temp file,
      yields the temp path, then deletes the temp file on exit.

    Usage::

        with resolve_photo_path(employee.passport_photo_path) as local_path:
            embedding = _get_embedding(local_path)
    """
    if not is_remote_url(passport_path):
        # Local file — pass through unchanged
        yield passport_path
        return

    tmp_path: str | None = None
    try:
        suffix = os.path.splitext(passport_path.split("?")[0])[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            resp = requests.get(passport_path, timeout=30, stream=True)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
        yield tmp_path
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
