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


class TagFacetsResponse(BaseModel):
    """Tag facets grouped by namespace with ordering."""

    facets: dict[str, list[TagFacetItem]]
    namespace_order: list[str]


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


class DeleteAllTagsRequest(BaseModel):
    """Request to delete all tags."""

    confirm: bool


class DeleteAllTagsResponse(BaseModel):
    """Response after deleting all tags."""

    deleted_count: int
    skipped_system_count: int
    success: bool
