"""Authentication schemas."""

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserBasicResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserBasicResponse
    setup_required: bool = False  # True if admin needs to complete first-run setup


class SetupRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str
