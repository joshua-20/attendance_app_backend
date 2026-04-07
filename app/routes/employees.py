import os
import uuid
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database import get_db
from ..models import Employee
from ..schemas import EmployeeResponse
from ..services.face_service import validate_face_in_image

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


def _save_upload(content: bytes, directory: str) -> tuple[str, str]:
    """Persist uploaded bytes to disk and return (safe_filename, full_path)."""
    os.makedirs(directory, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex}.jpg"
    full_path = os.path.join(directory, safe_filename)
    with open(full_path, "wb") as fp:
        fp.write(content)
    return safe_filename, full_path


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

    _, passport_path = _save_upload(content, PASSPORT_DIR)

    # Ensure the photo contains a face before committing to the database
    if not validate_face_in_image(passport_path):
        os.remove(passport_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in the uploaded passport photo. Please upload a clear face photo.",
        )

    employee = Employee(
        employee_id=employee_id,
        name=name,
        email=email,
        department=department,
        role=role,
        passport_photo_path=passport_path,
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)
    logger.info("Registered employee '%s' (id=%s)", name, employee_id)
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

    _, new_path = _save_upload(content, PASSPORT_DIR)

    if not validate_face_in_image(new_path):
        os.remove(new_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in the uploaded photo.",
        )

    # Remove the old passport photo from disk
    old_path = employee.passport_photo_path
    if old_path and os.path.exists(old_path):
        os.remove(old_path)

    employee.passport_photo_path = new_path
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

    if employee.passport_photo_path and os.path.exists(employee.passport_photo_path):
        os.remove(employee.passport_photo_path)

    db.delete(employee)
    db.commit()
