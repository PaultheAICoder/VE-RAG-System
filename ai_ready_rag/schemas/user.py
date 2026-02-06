"""User management schemas."""

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    display_name: str
    password: str
    role: str = "user"


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    display_name: str | None = None
    role: str | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    must_reset_password: bool
    tags: list[dict] = []

    class Config:
        from_attributes = True


class TagAssignment(BaseModel):
    tag_ids: list[str]
