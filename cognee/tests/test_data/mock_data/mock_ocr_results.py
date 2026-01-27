"""Mock OCR results and test data generators for testing."""

from typing import List, Optional
from cognee.infrastructure.ocr.PaddleOCRAdapter import (
    BoundingBox,
    OCRTextElement,
    OCRPageResult,
    OCRDocumentResult,
)
from cognee.modules.chunking.models.LayoutChunk import (
    BoundingBox as ChunkBoundingBox,
    PageDimensions,
    LayoutType,
)


def create_mock_bbox(
    x_min: float = 0.1,
    y_min: float = 0.2,
    x_max: float = 0.8,
    y_max: float = 0.9,
    pixel_x_min: Optional[int] = None,
    pixel_y_min: Optional[int] = None,
    pixel_x_max: Optional[int] = None,
    pixel_y_max: Optional[int] = None,
    confidence: float = 0.95,
) -> BoundingBox:
    """Generate test BoundingBox for OCR adapter."""
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        pixel_x_min=pixel_x_min or int(x_min * 1000),
        pixel_y_min=pixel_y_min or int(y_min * 1000),
        pixel_x_max=pixel_x_max or int(x_max * 1000),
        pixel_y_max=pixel_y_max or int(y_max * 1000),
        confidence=confidence,
    )


def create_mock_chunk_bbox(
    x_min: float = 0.1,
    y_min: float = 0.2,
    x_max: float = 0.8,
    y_max: float = 0.9,
    pixel_x_min: Optional[int] = None,
    pixel_y_min: Optional[int] = None,
    pixel_x_max: Optional[int] = None,
    pixel_y_max: Optional[int] = None,
    confidence: float = 1.0,
) -> ChunkBoundingBox:
    """Generate test BoundingBox for LayoutChunk."""
    return ChunkBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        pixel_x_min=pixel_x_min,
        pixel_y_min=pixel_y_min,
        pixel_x_max=pixel_x_max,
        pixel_y_max=pixel_y_max,
        confidence=confidence,
    )


def create_mock_ocr_element(
    text: str = "Sample text",
    bbox: Optional[BoundingBox] = None,
    page_number: int = 1,
    confidence: float = 0.95,
) -> OCRTextElement:
    """Generate test OCRTextElement."""
    if bbox is None:
        bbox = create_mock_bbox(confidence=confidence)

    return OCRTextElement(
        text=text,
        bbox=bbox,
        confidence=confidence,
        page_number=page_number,
    )


def create_mock_ocr_page_result(
    page_number: int = 1,
    num_elements: int = 5,
    width: int = 1000,
    height: int = 1400,
) -> OCRPageResult:
    """Generate OCRPageResult for testing."""
    elements = []
    for i in range(num_elements):
        y_offset = i * 0.1
        elements.append(
            create_mock_ocr_element(
                text=f"Line {i + 1} text content",
                bbox=create_mock_bbox(
                    x_min=0.1,
                    y_min=0.1 + y_offset,
                    x_max=0.9,
                    y_max=0.15 + y_offset,
                ),
                page_number=page_number,
            )
        )

    return OCRPageResult(
        page_number=page_number,
        elements=elements,
        page_width=width,
        page_height=height,
    )


def create_mock_ocr_result(
    num_pages: int = 2,
    elements_per_page: int = 5,
) -> OCRDocumentResult:
    """Generate OCRDocumentResult for testing."""
    pages = []
    for page_num in range(1, num_pages + 1):
        pages.append(
            create_mock_ocr_page_result(
                page_number=page_num,
                num_elements=elements_per_page,
            )
        )

    return OCRDocumentResult(
        pages=pages,
        total_pages=num_pages,
    )


def create_test_layout_text(num_lines: int = 5, page_number: int = 1) -> str:
    """Generate text with layout metadata markers."""
    lines = []
    for i in range(num_lines):
        y_offset = i * 0.1
        line = (
            f"Line {i + 1} text content "
            f"[page={page_number}, "
            f"bbox=(0.1,{0.1 + y_offset:.2f},0.9,{0.15 + y_offset:.2f}), "
            f"type=text, "
            f"confidence=0.95]"
        )
        lines.append(line)

    return "\n".join(lines)


def create_mock_page_dimensions(
    width: int = 1000,
    height: int = 1400,
    dpi: Optional[int] = 150,
) -> PageDimensions:
    """Generate test PageDimensions."""
    return PageDimensions(width=width, height=height, dpi=dpi)


def assert_bbox_normalized(bbox):
    """Validate bbox coordinates are in 0-1 range and properly ordered."""
    assert 0 <= bbox.x_min <= 1, f"x_min {bbox.x_min} not in 0-1 range"
    assert 0 <= bbox.y_min <= 1, f"y_min {bbox.y_min} not in 0-1 range"
    assert 0 <= bbox.x_max <= 1, f"x_max {bbox.x_max} not in 0-1 range"
    assert 0 <= bbox.y_max <= 1, f"y_max {bbox.y_max} not in 0-1 range"
    assert bbox.x_min < bbox.x_max, f"x_min {bbox.x_min} >= x_max {bbox.x_max}"
    assert bbox.y_min < bbox.y_max, f"y_min {bbox.y_min} >= y_max {bbox.y_max}"


def assert_has_valid_layout(chunk_metadata):
    """Validate chunk layout metadata structure."""
    assert chunk_metadata.layout is not None, "Layout metadata is None"

    if chunk_metadata.layout.bbox:
        assert_bbox_normalized(chunk_metadata.layout.bbox)

    if chunk_metadata.layout.page_number is not None:
        assert chunk_metadata.layout.page_number >= 1, "Page number must be >= 1"
