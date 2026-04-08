import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator


# ---------------------------------------------------------------------------
# Auth schemas
# ---------------------------------------------------------------------------

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    admin_id: int
    username: str

class AdminCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


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


class AttendancePeriod(BaseModel):
    """Query helper — not a DB model."""
    period: str = "today"   # "today" | "week" | "month"


# --- Face verification response ---

class FaceVerificationResponse(BaseModel):
    success: bool
    message: str
    employee: Optional[EmployeeResponse] = None
    attendance: Optional[AttendanceResponse] = None
    confidence_score: Optional[float] = None
