"""Tag management schemas."""

from pydantic import BaseModel


class TagCreate(BaseModel):
    name: str
    display_name: str
    description: str | None = None
    color: str = "#6B7280"
    owner_id: str | None = None


class TagUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    color: str | None = None
    owner_id: str | None = None


class TagFacetItem(BaseModel):
    """A single tag within a facet namespace."""

    name: str
    display: str
    count: int


class TagResponse(BaseModel):
    id: str
    name: str
    display_name: str
    description: str | None
    color: str
    owner_id: str | None
    is_system: bool

    class Config:
        from_attributes = True
