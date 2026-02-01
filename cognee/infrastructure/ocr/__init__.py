"""OCR infrastructure for document processing."""

from .config import OCRConfig, get_ocr_config, is_ppstructure_available
from .PaddleOCRAdapter import (
    BoundingBox,
    OCRTextElement,
    OCRPageResult,
    OCRDocumentResult,
    PaddleOCRAdapter,
)

__all__ = [
    "OCRConfig",
    "get_ocr_config",
    "BoundingBox",
    "OCRTextElement",
    "OCRPageResult",
    "OCRDocumentResult",
    "PaddleOCRAdapter",
    "is_paddleocr_available",
    "is_pdfplumber_available",
    "is_ppstructure_available",
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
