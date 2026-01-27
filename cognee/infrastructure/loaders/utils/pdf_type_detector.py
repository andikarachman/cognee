"""PDF type detection utility to determine if PDF is digital or scanned."""

from enum import Enum
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


class PDFType(Enum):
    """PDF type classification."""

    DIGITAL = "digital"  # Has extractable text
    SCANNED = "scanned"  # Requires OCR
    HYBRID = "hybrid"  # Mix of both


def detect_pdf_type(pdf_path: str, sample_pages: int = 3) -> PDFType:
    """
    Detect if PDF is digital (extractable text) or scanned (requires OCR).

    Strategy:
    1. Extract text from first N pages using pdfplumber/PyPDF
    2. Count extractable characters per page
    3. If avg chars/page > 100: DIGITAL
    4. If avg chars/page < 20: SCANNED
    5. Else: HYBRID (use OCR to be safe)

    Args:
        pdf_path: Path to PDF file
        sample_pages: Number of pages to sample (default 3)

    Returns:
        PDFType enum indicating whether PDF is digital, scanned, or hybrid

    Raises:
        ImportError: If pdfplumber is not installed
        Exception: If PDF cannot be read
    """
    try:
        import pdfplumber
    except ImportError as e:
        # Fallback to pypdf if pdfplumber not available
        logger.warning("pdfplumber not available, falling back to pypdf for detection")
        try:
            from pypdf import PdfReader

            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                total_chars = 0
                pages_checked = 0

                for i in range(min(sample_pages, len(reader.pages))):
                    text = reader.pages[i].extract_text() or ""
                    total_chars += len(text.strip())
                    pages_checked += 1

                avg_chars_per_page = (
                    total_chars / pages_checked if pages_checked > 0 else 0
                )

                if avg_chars_per_page > 100:
                    logger.info(
                        f"PDF {pdf_path} detected as DIGITAL "
                        f"(avg {avg_chars_per_page:.1f} chars/page)"
                    )
                    return PDFType.DIGITAL
                elif avg_chars_per_page < 20:
                    logger.info(
                        f"PDF {pdf_path} detected as SCANNED "
                        f"(avg {avg_chars_per_page:.1f} chars/page)"
                    )
                    return PDFType.SCANNED
                else:
                    logger.info(
                        f"PDF {pdf_path} detected as HYBRID "
                        f"(avg {avg_chars_per_page:.1f} chars/page)"
                    )
                    return PDFType.HYBRID

        except ImportError:
            raise ImportError(
                "Either pdfplumber or pypdf is required for PDF type detection. "
                "Install with: pip install cognee[ocr]"
            ) from e

    try:
        total_chars = 0
        pages_checked = 0

        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(sample_pages, len(pdf.pages))

            for i in range(pages_to_check):
                text = pdf.pages[i].extract_text() or ""
                total_chars += len(text.strip())
                pages_checked += 1

        avg_chars_per_page = total_chars / pages_checked if pages_checked > 0 else 0

        if avg_chars_per_page > 100:
            logger.info(
                f"PDF {pdf_path} detected as DIGITAL "
                f"(avg {avg_chars_per_page:.1f} chars/page)"
            )
            return PDFType.DIGITAL
        elif avg_chars_per_page < 20:
            logger.info(
                f"PDF {pdf_path} detected as SCANNED "
                f"(avg {avg_chars_per_page:.1f} chars/page)"
            )
            return PDFType.SCANNED
        else:
            logger.info(
                f"PDF {pdf_path} detected as HYBRID "
                f"(avg {avg_chars_per_page:.1f} chars/page)"
            )
            return PDFType.HYBRID

    except Exception as e:
        logger.error(f"Failed to detect PDF type for {pdf_path}: {e}")
        # On error, assume HYBRID to use OCR (safer default)
        logger.warning("Defaulting to HYBRID type due to detection error")
        return PDFType.HYBRID
