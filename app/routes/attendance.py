import datetime
import logging
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database import get_db
from ..models import Attendance, Employee
from ..schemas import AttendanceResponse, FaceVerificationResponse
from ..services.face_service import count_faces_in_image, find_best_match, validate_face_in_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
CAPTURES_DIR = os.path.join(UPLOAD_DIR, "captures")
ALLOWED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


def _validate_image_upload(file: UploadFile) -> None:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )


def _save_capture(content: bytes) -> str:
    """Persist capture bytes to disk and return the full path."""
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    safe_filename = f"{uuid.uuid4().hex}.jpg"
    full_path = os.path.join(CAPTURES_DIR, safe_filename)
    with open(full_path, "wb") as fp:
        fp.write(content)
    return full_path


# ---------------------------------------------------------------------------
# POST /api/attendance/check-in
# ---------------------------------------------------------------------------
@router.post("/check-in", response_model=FaceVerificationResponse)
async def check_in(
    face_capture: UploadFile = File(
        ...,
        description="Live face photo captured from the mobile app",
    ),
    db: Session = Depends(get_db),
):
    """
    Identify an employee from a live face capture and record attendance.

    Flow
    ----
    1. Receive face image from mobile app.
    2. Validate that a face is present in the image.
    3. Compare the face against every employee's stored passport photo.
    4. On match → record check-in (or check-out if already checked in today).
    5. Return the matched employee details and attendance record.
    """
    _validate_image_upload(face_capture)

    content = await face_capture.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image exceeds 10 MB.",
        )

    capture_path = _save_capture(content)

    # --- Face presence check ---
    if not validate_face_in_image(capture_path):
        os.remove(capture_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No face detected in the captured image. Please retake the photo.",
        )

    # --- Multiple-face check (security: only one person may check in at a time) ---
    if count_faces_in_image(capture_path) > 1:
        os.remove(capture_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Multiple faces detected. Ensure only one person is visible.",
        )

    # --- Load all employees that have a passport photo ---
    employees = (
        db.query(Employee).filter(Employee.passport_photo_path.isnot(None)).all()
    )
    if not employees:
        os.remove(capture_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No employee records with passport photos found.",
        )

    employee_pairs = [(emp.id, emp.passport_photo_path) for emp in employees]

    # --- Match face ---
    match_result = find_best_match(capture_path, employee_pairs)

    if not match_result:
        # Keep the capture for audit purposes even on failed match
        logger.info("Face capture saved at %s — no match found.", capture_path)
        return FaceVerificationResponse(
            success=False,
            message="Face not recognised. Access denied.",
            confidence_score=0.0,
        )

    matched_db_id, confidence = match_result
    matched_employee = db.query(Employee).filter(Employee.id == matched_db_id).first()

    # --- Attendance logic (check-in / check-out) ---
    today = datetime.date.today()
    now = datetime.datetime.utcnow()

    existing = (
        db.query(Attendance)
        .filter(Attendance.employee_id == matched_db_id, Attendance.date == today)
        .first()
    )

    if existing:
        if existing.check_out_time is None:
            # Second scan of the day → check-out
            existing.check_out_time = now
            existing.capture_photo_path = capture_path
            db.commit()
            db.refresh(existing)
            return FaceVerificationResponse(
                success=True,
                message=f"Check-out recorded for {matched_employee.name}.",
                employee=matched_employee,
                attendance=existing,
                confidence_score=round(confidence, 4),
            )
        else:
            return FaceVerificationResponse(
                success=True,
                message=f"{matched_employee.name} has already completed attendance for today.",
                employee=matched_employee,
                attendance=existing,
                confidence_score=round(confidence, 4),
            )

    # First scan of the day → check-in
    attendance = Attendance(
        employee_id=matched_db_id,
        date=today,
        check_in_time=now,
        confidence_score=round(confidence, 4),
        capture_photo_path=capture_path,
        status="present",
    )
    db.add(attendance)
    db.commit()
    db.refresh(attendance)
    logger.info(
        "Check-in: employee=%s confidence=%.4f",
        matched_employee.employee_id,
        confidence,
    )
    return FaceVerificationResponse(
        success=True,
        message=f"Check-in recorded for {matched_employee.name}.",
        employee=matched_employee,
        attendance=attendance,
        confidence_score=round(confidence, 4),
    )


# ---------------------------------------------------------------------------
# GET /api/attendance/today
# ---------------------------------------------------------------------------
@router.get("/today", response_model=List[AttendanceResponse])
def today_attendance(db: Session = Depends(get_db)):
    """Return all attendance records for the current date."""
    today = datetime.date.today()
    return (
        db.query(Attendance)
        .filter(Attendance.date == today)
        .order_by(Attendance.check_in_time)
        .all()
    )


# ---------------------------------------------------------------------------
# GET /api/attendance/
# ---------------------------------------------------------------------------
@router.get("/", response_model=List[AttendanceResponse])
def list_attendance(
    date: Optional[datetime.date] = None,
    employee_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Return a paginated list of attendance records.

    Filters
    -------
    - `date`        — exact date (YYYY-MM-DD)
    - `employee_id` — the employee's string ID (e.g. "EMP001")
    """
    query = db.query(Attendance)

    if date:
        query = query.filter(Attendance.date == date)

    if employee_id:
        emp = db.query(Employee).filter(Employee.employee_id == employee_id).first()
        if emp:
            query = query.filter(Attendance.employee_id == emp.id)
        else:
            return []  # unknown employee_id → empty result

    return (
        query.order_by(Attendance.date.desc(), Attendance.check_in_time.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# GET /api/attendance/{attendance_id}
# ---------------------------------------------------------------------------
@router.get("/{attendance_id}", response_model=AttendanceResponse)
def get_attendance_record(attendance_id: int, db: Session = Depends(get_db)):
    """Fetch a single attendance record by its numeric ID."""
    record = db.query(Attendance).filter(Attendance.id == attendance_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attendance record {attendance_id} not found.",
        )
    return record
