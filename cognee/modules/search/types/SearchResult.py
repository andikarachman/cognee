from uuid import UUID
from pydantic import BaseModel, Field
from typing import Any, Optional, List


class SearchResultDataset(BaseModel):
    id: UUID
    name: str


class BoundingBox(BaseModel):
    """Bounding box for layout positioning."""

    x_min: float = Field(..., description="Normalized x minimum (0-1)")
    y_min: float = Field(..., description="Normalized y minimum (0-1)")
    x_max: float = Field(..., description="Normalized x maximum (0-1)")
    y_max: float = Field(..., description="Normalized y maximum (0-1)")
    confidence: Optional[float] = Field(None, description="Detection confidence")


class LayoutMetadata(BaseModel):
    """Layout metadata for a chunk."""

    page_number: Optional[int] = Field(None, description="Page number")
    bbox: Optional[BoundingBox] = Field(None, description="Primary bounding box")
    layout_type: Optional[str] = Field(None, description="Layout type (text, table, etc.)")
    confidence: Optional[float] = Field(None, description="OCR confidence score")
    page_width: Optional[int] = Field(None, description="Page width in pixels")
    page_height: Optional[int] = Field(None, description="Page height in pixels")


class DocumentReference(BaseModel):
    """Reference to source document."""

    id: UUID
    name: str
    mime_type: Optional[str] = None


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk in search results."""

    text: str = Field(..., description="Chunk text content")
    chunk_id: Optional[UUID] = Field(None, description="Unique chunk identifier")
    chunk_index: Optional[int] = Field(None, description="Chunk index in document")
    score: Optional[float] = Field(None, description="Search relevance score")
    layout: Optional[LayoutMetadata] = Field(None, description="Layout metadata (if available)")


class SearchResult(BaseModel):
    search_result: Any  # Backward compatible
    dataset_id: Optional[UUID]
    dataset_name: Optional[str]
    chunks: Optional[List[ChunkMetadata]] = Field(
        None,
        description="Structured chunk metadata with layout info (for CHUNKS search type)",
    )
    document: Optional[DocumentReference] = Field(
        None,
        description="Source document reference",
    )
