"""PaddleOCR adapter for OCR processing."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for text with normalized (0-1) and pixel coordinates."""

    x_min: float  # Normalized 0-1
    y_min: float
    x_max: float
    y_max: float
    pixel_x_min: Optional[int] = None
    pixel_y_min: Optional[int] = None
    pixel_x_max: Optional[int] = None
    pixel_y_max: Optional[int] = None
    confidence: float = 1.0

    @property
    def area(self) -> float:
        """Calculate normalized bbox area (0-1 range)."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bbox (normalized coordinates)."""
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        return (center_x, center_y)


@dataclass
class OCRTextElement:
    """Single text element extracted from OCR."""

    text: str
    bbox: BoundingBox
    confidence: float
    page_number: int


@dataclass
class OCRPageResult:
    """OCR result for a single page."""

    page_number: int
    elements: List[OCRTextElement]
    page_width: int
    page_height: int


@dataclass
class OCRDocumentResult:
    """OCR result for entire document."""

    pages: List[OCRPageResult]
    total_pages: int


class PaddleOCRAdapter:
    """
    Adapter for PaddleOCR functionality.

    Provides OCR processing for images and PDFs with bounding box extraction.
    Lazy initialization ensures PaddleOCR is only loaded when actually used.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
    ):
        """
        Initialize PaddleOCR adapter.

        Args:
            lang: Language code for OCR (e.g., 'en', 'ch', 'fr')
            use_gpu: Whether to use GPU acceleration
            min_confidence: Minimum confidence threshold for text detection
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """Lazy initialization of PaddleOCR engine."""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR

                logger.info(f"Initializing PaddleOCR with lang={self.lang}, use_gpu={self.use_gpu}")
                self._ocr_engine = PaddleOCR(
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
            except ImportError as e:
                raise ImportError(
                    "PaddleOCR is required for OCR processing. "
                    "Install with: pip install cognee[ocr]"
                ) from e

        return self._ocr_engine

    def _normalize_bbox(
        self,
        bbox_coords: List[List[float]],
        page_width: int,
        page_height: int,
    ) -> BoundingBox:
        """
        Normalize bounding box coordinates to 0-1 range.

        Args:
            bbox_coords: PaddleOCR bbox format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            page_width: Page width in pixels
            page_height: Page height in pixels

        Returns:
            BoundingBox with normalized and pixel coordinates
        """
        # Extract min/max from 4-point polygon
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]

        pixel_x_min = int(min(x_coords))
        pixel_y_min = int(min(y_coords))
        pixel_x_max = int(max(x_coords))
        pixel_y_max = int(max(y_coords))

        # Normalize to 0-1 range
        x_min = pixel_x_min / page_width if page_width > 0 else 0
        y_min = pixel_y_min / page_height if page_height > 0 else 0
        x_max = pixel_x_max / page_width if page_width > 0 else 1
        y_max = pixel_y_max / page_height if page_height > 0 else 1

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            pixel_x_min=pixel_x_min,
            pixel_y_min=pixel_y_min,
            pixel_x_max=pixel_x_max,
            pixel_y_max=pixel_y_max,
        )

    async def process_image(
        self,
        image_path: str,
        page_number: int = 1,
    ) -> OCRPageResult:
        """
        Process a single image with OCR.

        Args:
            image_path: Path to image file
            page_number: Page number for tracking (default 1 for images)

        Returns:
            OCRPageResult with extracted text and bounding boxes

        Raises:
            ImportError: If PaddleOCR not installed
            Exception: If OCR processing fails
        """
        try:
            from PIL import Image

            ocr_engine = self._get_ocr_engine()

            # Load image to get dimensions
            with Image.open(image_path) as img:
                page_width, page_height = img.size

            logger.info(f"Processing image {image_path} with PaddleOCR")
            result = ocr_engine.ocr(image_path, cls=True)

            elements = []
            if result and result[0]:
                for line in result[0]:
                    bbox_coords = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]

                    # Skip low confidence results
                    if confidence < self.min_confidence:
                        continue

                    bbox = self._normalize_bbox(bbox_coords, page_width, page_height)
                    bbox.confidence = confidence

                    elements.append(
                        OCRTextElement(
                            text=text,
                            bbox=bbox,
                            confidence=confidence,
                            page_number=page_number,
                        )
                    )

            logger.info(f"Extracted {len(elements)} text elements from image")
            return OCRPageResult(
                page_number=page_number,
                elements=elements,
                page_width=page_width,
                page_height=page_height,
            )

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise

    async def process_pdf(self, pdf_path: str) -> OCRDocumentResult:
        """
        Process PDF with OCR by converting to images first.

        Args:
            pdf_path: Path to PDF file

        Returns:
            OCRDocumentResult with all pages processed

        Raises:
            ImportError: If required libraries not installed
            Exception: If PDF processing fails
        """
        try:
            from pdf2image import convert_from_path

            logger.info(f"Converting PDF {pdf_path} to images for OCR")
            images = convert_from_path(pdf_path)

            pages = []
            for page_num, image in enumerate(images, start=1):
                # Save image temporarily
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path, "PNG")

                # Process with OCR
                page_result = await self.process_image(temp_path, page_number=page_num)
                pages.append(page_result)

                # Cleanup temp file
                import os

                os.unlink(temp_path)

                logger.info(
                    f"Processed page {page_num}/{len(images)} "
                    f"({len(page_result.elements)} elements)"
                )

            return OCRDocumentResult(
                pages=pages,
                total_pages=len(pages),
            )

        except ImportError as e:
            raise ImportError(
                "pdf2image is required for PDF OCR. Install with: pip install cognee[ocr]"
            ) from e
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
