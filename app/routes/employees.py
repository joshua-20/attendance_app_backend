import os
import uuid
import logging
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database import get_db
from ..models import Employee
from ..schemas import EmployeeResponse
from ..services.face_service import validate_face_in_image
from ..services.cloudinary_service import (
    CLOUDINARY_ENABLED,
    upload_passport,
    delete_photo,
    is_remote_url,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/employees", tags=["Employees"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
PASSPORT_DIR = os.path.join(UPLOAD_DIR, "passports")
ALLOWED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


def _validate_image_upload(file: UploadFile) -> None:
    """Raise HTTPException for invalid image files."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )


def _save_local(content: bytes, directory: str) -> str:
    """Persist bytes to a local directory and return the full path."""
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, f"{uuid.uuid4().hex}.jpg")
    with open(full_path, "wb") as fp:
        fp.write(content)
    return full_path


def _store_passport(content: bytes, employee_id: str) -> tuple[str, str | None]:
    """
    Persist a passport photo and return ``(stored_path, temp_path)``.

    * When Cloudinary is enabled the photo is uploaded to Cloudinary and
      ``stored_path`` is the ``secure_url``.  ``temp_path`` is a temporary
      local file that was used for face-validation; the caller must delete it.
    * When Cloudinary is disabled the photo is saved to local PASSPORT_DIR and
      ``stored_path`` is the local filesystem path.  ``temp_path`` is None.
    """
    if CLOUDINARY_ENABLED:
        # Write to temp file first so face_service can validate it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        cloudinary_url = upload_passport(content, employee_id)
        return cloudinary_url, tmp_path
    else:
        local_path = _save_local(content, PASSPORT_DIR)
        return local_path, None


def _remove_passport(stored_path: str) -> None:
    """Delete a passport photo from Cloudinary or local disk."""
    if not stored_path:
        return
    if is_remote_url(stored_path):
        delete_photo(stored_path)
    elif os.path.exists(stored_path):
        os.remove(stored_path)


# ---------------------------------------------------------------------------
# POST /api/employees/
# ---------------------------------------------------------------------------
@router.post("/", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
async def register_employee(
    employee_id: str = Form(..., description="Unique employee identifier"),
    name: str = Form(..., description="Full name"),
    email: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    passport_photo: UploadFile = File(..., description="Clear passport-style face photo"),
    db: Session = Depends(get_db),
):
    """
    Register a new employee and store their passport photo.

    The uploaded photo is validated to contain a detectable human face before
    being persisted. This face embedding is later used for attendance matching.
    """
    # Duplicate check
    if db.query(Employee).filter(Employee.employee_id == employee_id).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Employee with ID '{employee_id}' already exists.",
        )

    _validate_image_upload(passport_photo)

    content = await passport_photo.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passport photo exceeds the 10 MB size limit.",
        )

    stored_path, tmp_path = _store_passport(content, employee_id)

    # Validate face using temp file (Cloudinary) or the stored local file
    validation_path = tmp_path if tmp_path else stored_path
    if not validate_face_in_image(validation_path):
        # Clean up both temp and cloud/local copies on failure
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        _remove_passport(stored_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in the uploaded passport photo. Please upload a clear face photo.",
        )

    # Temp file no longer needed after validation
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    employee = Employee(
        employee_id=employee_id,
        name=name,
        email=email,
        department=department,
        role=role,
        passport_photo_path=stored_path,
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)
    logger.info(
        "Registered employee '%s' (id=%s) — photo stored at %s",
        name, employee_id, "Cloudinary" if CLOUDINARY_ENABLED else stored_path,
    )
    return employee


# ---------------------------------------------------------------------------
# GET /api/employees/
# ---------------------------------------------------------------------------
@router.get("/", response_model=List[EmployeeResponse])
def list_employees(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Return a paginated list of all registered employees."""
    return db.query(Employee).offset(skip).limit(limit).all()


# ---------------------------------------------------------------------------
# GET /api/employees/{employee_id}
# ---------------------------------------------------------------------------
@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: str, db: Session = Depends(get_db)):
    """Fetch a single employee by their employee_id string."""
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee '{employee_id}' not found.",
        )
    return employee


# ---------------------------------------------------------------------------
# PUT /api/employees/{employee_id}/photo
# ---------------------------------------------------------------------------
@router.put("/{employee_id}/photo", response_model=EmployeeResponse)
async def update_passport_photo(
    employee_id: str,
    passport_photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Replace an employee's stored passport photo."""
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee '{employee_id}' not found.",
        )

    _validate_image_upload(passport_photo)
    content = await passport_photo.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Photo exceeds 10 MB size limit.",
        )

    new_stored_path, tmp_path = _store_passport(content, employee_id)

    validation_path = tmp_path if tmp_path else new_stored_path
    if not validate_face_in_image(validation_path):
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        _remove_passport(new_stored_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in the uploaded photo.",
        )

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Remove the old passport photo
    _remove_passport(employee.passport_photo_path or "")

    employee.passport_photo_path = new_stored_path
    db.commit()
    db.refresh(employee)
    return employee


# ---------------------------------------------------------------------------
# DELETE /api/employees/{employee_id}
# ---------------------------------------------------------------------------
@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_employee(employee_id: str, db: Session = Depends(get_db)):
    """Remove an employee and their associated passport photo."""
    employee = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Employee '{employee_id}' not found.",
        )

    _remove_passport(employee.passport_photo_path or "")

    db.delete(employee)
    db.commit()
