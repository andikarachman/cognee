"""Layout-aware chunk model with bounding box metadata."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator
from cognee.modules.data.processing.document_types import Document
from .DocumentChunk import DocumentChunk


class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 range) with optional pixel coordinates."""

    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized x minimum (0-1)")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized y minimum (0-1)")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized x maximum (0-1)")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized y maximum (0-1)")

    pixel_x_min: Optional[int] = Field(None, description="Absolute x minimum in pixels")
    pixel_y_min: Optional[int] = Field(None, description="Absolute y minimum in pixels")
    pixel_x_max: Optional[int] = Field(None, description="Absolute x maximum in pixels")
    pixel_y_max: Optional[int] = Field(None, description="Absolute y maximum in pixels")

    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")

    @model_validator(mode="after")
    def validate_coordinate_ordering(self) -> "BoundingBox":
        """Validate that min coordinates are less than max coordinates."""
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        return self

    @property
    def area(self) -> float:
        """Calculate normalized area (0-1 range)."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self) -> tuple[float, float]:
        """Get center point (normalized coordinates)."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class PageDimensions(BaseModel):
    """Page dimensions in pixels."""

    width: int = Field(..., gt=0, description="Page width in pixels")
    height: int = Field(..., gt=0, description="Page height in pixels")
    dpi: Optional[int] = Field(None, description="Page DPI (if available)")


class LayoutType(str, Enum):
    """Layout element types."""

    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    CODE = "code"
    UNKNOWN = "unknown"


class LayoutChunk(DocumentChunk):
    """
    Extends DocumentChunk with layout and bounding box metadata.

    Stores OCR/layout information including:
    - Bounding boxes for spatial positioning
    - Page numbers for document navigation
    - Layout types for semantic understanding
    - OCR confidence scores
    - Reading order and column information

    Compatible with DocumentChunk for backward compatibility.
    """

    # Override DocumentChunk required fields with defaults for backward compatibility
    chunk_size: int = Field(default=0, description="Size of chunk in tokens (0 if not applicable)")
    chunk_index: int = Field(
        default=0, description="Index of chunk in document (0 if not applicable)"
    )
    cut_type: str = Field(default="layout", description="Type of cut used for chunking")
    is_part_of: Optional[Document] = Field(default=None, description="Parent document reference")

    # Layout-specific fields
    word_count: Optional[int] = Field(default=None, description="Number of words in this chunk")
    chunk_id: Optional[str] = Field(
        default=None, description="Optional human-readable chunk identifier"
    )

    bounding_boxes: List[BoundingBox] = Field(
        default_factory=list,
        description="List of bounding boxes for this chunk",
    )

    page_number: Optional[int] = Field(
        None,
        ge=1,
        description="Primary page number (for single-page chunks)",
    )

    page_numbers: Optional[List[int]] = Field(
        None,
        description="Page numbers (for multi-page chunks)",
    )

    page_dimensions: Optional[PageDimensions] = Field(
        None,
        description="Page dimensions in pixels",
    )

    layout_type: LayoutType = Field(
        default=LayoutType.TEXT,
        description="Type of layout element",
    )

    ocr_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Average OCR confidence score",
    )

    reading_order: Optional[int] = Field(
        None,
        ge=0,
        description="Reading order index on page",
    )

    column_index: Optional[int] = Field(
        None,
        ge=0,
        description="Column index for multi-column layouts",
    )

    @property
    def primary_bbox(self) -> Optional[BoundingBox]:
        """
        Get the primary (largest by area) bounding box.

        Returns:
            Largest bounding box or None if no boxes exist
        """
        if not self.bounding_boxes:
            return None
        return max(self.bounding_boxes, key=lambda b: b.area)

    @property
    def has_layout_info(self) -> bool:
        """Check if chunk has layout metadata."""
        return bool(self.bounding_boxes) or self.page_number is not None

    def get_page_numbers(self) -> List[int]:
        """
        Get all page numbers this chunk spans.

        Returns:
            List of unique page numbers (sorted)
        """
        if self.page_numbers:
            return sorted(set(self.page_numbers))
        elif self.page_number:
            return [self.page_number]
        else:
            return []

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )
