"""Pydantic models for structured OCR output format."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class OCRFormatType(str, Enum):
    """OCR output format type."""

    BLOCK = "block"  # Layout-aware format with grouped blocks
    FLAT = "flat"  # Flat format without layout grouping


class OCRBoundingBox(BaseModel):
    """
    Bounding box coordinates in normalized (0-1) range.

    Normalized coordinates allow consistent representation across different
    image/page sizes.
    """

    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized x minimum (0-1)")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized y minimum (0-1)")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized x maximum (0-1)")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized y maximum (0-1)")


class OCRSourceInfo(BaseModel):
    """Provenance information for OCR output."""

    loader: str = Field(..., description="Name of the loader that produced the OCR output")
    ocr_engine: str = Field(
        default="paddleocr", description="OCR engine used (e.g., paddleocr, tesseract)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="ISO 8601 timestamp when OCR was performed",
    )
    use_structure: bool = Field(
        default=False, description="Whether layout-aware OCR (PPStructureV3) was used"
    )


class OCRDocumentInfo(BaseModel):
    """Document-level metadata."""

    total_pages: int = Field(..., ge=1, description="Total number of pages in document")
    content_hash: Optional[str] = Field(None, description="Hash of original file content")
    source_filename: Optional[str] = Field(None, description="Original filename if available")


class OCRTextElementOutput(BaseModel):
    """Single text element extracted from OCR."""

    text: str = Field(..., description="Extracted text content")
    bbox: OCRBoundingBox = Field(..., description="Bounding box of text element")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="OCR confidence score for this element"
    )
    layout_type: str = Field(default="text", description="Layout type classification")


class OCRLayoutBlockOutput(BaseModel):
    """
    Layout block containing grouped text elements.

    Used in block format when layout-aware OCR (PPStructureV3) is enabled.
    """

    layout_type: str = Field(..., description="Type of layout region (e.g., paragraph, title)")
    bbox: OCRBoundingBox = Field(..., description="Bounding box of layout region")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Layout detection confidence"
    )
    elements: List[OCRTextElementOutput] = Field(
        default_factory=list, description="Text elements within this layout block"
    )


class OCRPageOutput(BaseModel):
    """OCR output for a single page."""

    page_number: int = Field(..., ge=1, description="1-based page number")
    width: Optional[int] = Field(None, ge=1, description="Page width in pixels")
    height: Optional[int] = Field(None, ge=1, description="Page height in pixels")

    # For block format (layout-aware)
    layout_blocks: Optional[List[OCRLayoutBlockOutput]] = Field(
        None, description="Layout blocks (used in block format)"
    )

    # For flat format (non-layout-aware)
    elements: Optional[List[OCRTextElementOutput]] = Field(
        None, description="Text elements (used in flat format)"
    )


class OCROutputDocument(BaseModel):
    """
    Root document model for structured OCR output.

    This is the JSON schema for OCR output files. It provides:
    - Version identifier for compatibility checking
    - Clear separation between text content and metadata
    - Support for both layout-aware (block) and flat formats
    - Clean plain_text field for embedding (no metadata noise)

    Example (block format):
        {
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {
                "loader": "ocr_image_loader",
                "ocr_engine": "paddleocr",
                "timestamp": "2025-01-15T10:30:00Z"
            },
            "document": {
                "total_pages": 1,
                "content_hash": "abc123..."
            },
            "pages": [...],
            "plain_text": "Hello world"
        }
    """

    cognee_ocr_format: str = Field(
        default="1.0", description="Format version for compatibility checking"
    )
    format_type: OCRFormatType = Field(
        ..., description="Format type: 'block' for layout-aware, 'flat' for non-layout"
    )
    source: OCRSourceInfo = Field(..., description="Provenance information")
    document: OCRDocumentInfo = Field(..., description="Document-level metadata")
    pages: List[OCRPageOutput] = Field(..., description="List of pages with OCR results")
    plain_text: str = Field(
        ..., description="Clean text content without metadata (for embedding/search)"
    )

    def get_all_text_elements(self) -> List[OCRTextElementOutput]:
        """
        Get all text elements across all pages.

        Returns:
            Flattened list of all text elements
        """
        elements = []
        for page in self.pages:
            if page.layout_blocks:
                for block in page.layout_blocks:
                    elements.extend(block.elements)
            elif page.elements:
                elements.extend(page.elements)
        return elements

    def get_text_by_page(self, page_number: int) -> str:
        """
        Get plain text for a specific page.

        Args:
            page_number: 1-based page number

        Returns:
            Text content for the page
        """
        for page in self.pages:
            if page.page_number == page_number:
                texts = []
                if page.layout_blocks:
                    for block in page.layout_blocks:
                        for elem in block.elements:
                            texts.append(elem.text)
                elif page.elements:
                    for elem in page.elements:
                        texts.append(elem.text)
                return " ".join(texts)
        return ""


# Convenience constants
OCR_FORMAT_VERSION = "1.0"
