"""
Admin authentication routes.

POST /api/auth/login   — returns a JWT
POST /api/auth/seed    — creates the first admin (only works when no admin exists)
GET  /api/auth/me      — returns current admin profile
"""
import logging
import os

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import create_access_token, hash_password, require_admin, verify_password
from ..database import get_db
from ..models import Admin
from ..schemas import AdminCreate, AdminLoginRequest, TokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------
@router.post("/login", response_model=TokenResponse)
def login(body: AdminLoginRequest, db: Session = Depends(get_db)):
    """Authenticate an admin and return a JWT access token."""
    admin = db.query(Admin).filter(
        Admin.username == body.username,
        Admin.is_active == True,
    ).first()

    if not admin or not verify_password(body.password, admin.hashed_password):
        logger.warning("ADMIN LOGIN FAILED  username=%s", body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    token = create_access_token(admin.id, admin.username)
    logger.info("ADMIN LOGIN SUCCESS  username=%s  id=%s", admin.username, admin.id)
    return TokenResponse(
        access_token=token,
        admin_id=admin.id,
        username=admin.username,
    )


# ---------------------------------------------------------------------------
# POST /api/auth/seed
# Creates the very first admin.  Disabled once any admin exists.
# The initial password can also be set via ADMIN_SEED_PASSWORD env var.
# ---------------------------------------------------------------------------
@router.post("/seed", status_code=status.HTTP_201_CREATED)
def seed_admin(body: AdminCreate, db: Session = Depends(get_db)):
    """
    Bootstrap the first admin account.
    This endpoint is only available when **no** admin exists in the database.
    Remove or disable it after initial setup.
    """
    if db.query(Admin).count() > 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An admin already exists. Use /api/auth/login instead.",
        )

    admin = Admin(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    logger.info("ADMIN SEEDED  username=%s", admin.username)
    return {"message": f"Admin '{admin.username}' created successfully."}


# ---------------------------------------------------------------------------
# GET /api/auth/me
# ---------------------------------------------------------------------------
@router.get("/me")
def me(admin: Admin = Depends(require_admin)):
    """Return the currently authenticated admin's profile."""
    return {
        "id": admin.id,
        "username": admin.username,
        "email": admin.email,
        "created_at": admin.created_at,
    }
