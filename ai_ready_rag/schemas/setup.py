"""Setup wizard schemas."""

from pydantic import BaseModel, field_validator

from ai_ready_rag.services.settings_service import get_security_setting


class SetupStatusResponse(BaseModel):
    """Response for setup status check."""

    setup_complete: bool
    setup_required: bool  # True if not complete AND not bypassed


class CompleteSetupRequest(BaseModel):
    """Request to complete setup by changing admin password."""

    current_password: str
    new_password: str
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        min_length = get_security_setting("password_min_length", 12)
        if len(v) < min_length:
            raise ValueError(f"Password must be at least {min_length} characters long")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_must_match(cls, v: str, info) -> str:
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class CompleteSetupResponse(BaseModel):
    """Response after completing setup."""

    success: bool
    message: str
