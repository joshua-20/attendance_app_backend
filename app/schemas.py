import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator


# --- Employee schemas ---

class EmployeeBase(BaseModel):
    employee_id: str
    name: str
    email: Optional[EmailStr] = None
    department: Optional[str] = None
    role: Optional[str] = None


class EmployeeResponse(EmployeeBase):
    id: int
    passport_photo_path: Optional[str] = None
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


# --- Attendance schemas ---

class AttendanceResponse(BaseModel):
    id: int
    employee_id: int
    date: datetime.date
    check_in_time: Optional[datetime.datetime] = None
    check_out_time: Optional[datetime.datetime] = None
    confidence_score: Optional[float] = None
    status: str
    employee: EmployeeResponse

    model_config = {"from_attributes": True}


# --- Face verification response ---

class FaceVerificationResponse(BaseModel):
    success: bool
    message: str
    employee: Optional[EmployeeResponse] = None
    attendance: Optional[AttendanceResponse] = None
    confidence_score: Optional[float] = None
