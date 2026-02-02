"""OCR infrastructure for document processing."""

from .config import OCRConfig, get_ocr_config, is_ppstructure_available
from .PaddleOCRAdapter import (
    BoundingBox,
    OCRTextElement,
    OCRLayoutElement,
    OCRPageResult,
    OCRDocumentResult,
    PaddleOCRAdapter,
)
from .models import (
    OCRFormatType,
    OCRBoundingBox,
    OCRSourceInfo,
    OCRDocumentInfo,
    OCRTextElementOutput,
    OCRLayoutBlockOutput,
    OCRPageOutput,
    OCROutputDocument,
    OCR_FORMAT_VERSION,
)

__all__ = [
    # Config
    "OCRConfig",
    "get_ocr_config",
    # Availability checks
    "is_paddleocr_available",
    "is_pdfplumber_available",
    "is_ppstructure_available",
    # PaddleOCR adapter types
    "BoundingBox",
    "OCRTextElement",
    "OCRLayoutElement",
    "OCRPageResult",
    "OCRDocumentResult",
    "PaddleOCRAdapter",
    # Structured output models
    "OCRFormatType",
    "OCRBoundingBox",
    "OCRSourceInfo",
    "OCRDocumentInfo",
    "OCRTextElementOutput",
    "OCRLayoutBlockOutput",
    "OCRPageOutput",
    "OCROutputDocument",
    "OCR_FORMAT_VERSION",
]


def is_paddleocr_available() -> bool:
    """Check if PaddleOCR is installed and available."""
    try:
        import paddleocr

        return True
    except ImportError:
        return False


def is_pdfplumber_available() -> bool:
    """Check if pdfplumber is installed and available."""
    try:
        import pdfplumber

        return True
    except ImportError:
        return False
