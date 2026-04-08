import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Date, Boolean
from sqlalchemy.orm import relationship
from .database import Base


class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(150), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    department = Column(String(100), nullable=True)
    role = Column(String(100), nullable=True)
    passport_photo_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    attendance_records = relationship("Attendance", back_populates="employee", cascade="all, delete-orphan")


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    date = Column(Date, default=datetime.date.today, nullable=False)
    check_in_time = Column(DateTime, nullable=True)
    check_out_time = Column(DateTime, nullable=True)
    confidence_score = Column(Float, nullable=True)
    capture_photo_path = Column(String(500), nullable=True)
    status = Column(String(20), default="present", nullable=False)

    employee = relationship("Employee", back_populates="attendance_records")
