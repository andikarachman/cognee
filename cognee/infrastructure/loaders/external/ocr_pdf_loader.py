"""PDF loader with OCR support for scanned PDFs."""

from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata

logger = get_logger(__name__)


class OcrPdfLoader(LoaderInterface):
    """
    PDF loader with automatic type detection and OCR support.

    Detects if PDF is digital (has extractable text) or scanned (requires OCR):
    - Digital PDFs: Delegates to PdfPlumberLoader (fast, native text extraction)
    - Scanned PDFs: Uses PaddleOCR for text extraction with bounding boxes
    - Hybrid PDFs: Uses OCR (safer default)

    Fallback chain: OCR -> pdfplumber -> PyPDF
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    @property
    def loader_name(self) -> str:
        return "ocr_pdf_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True
        return False

    def _format_ocr_result(self, ocr_result, file_metadata: dict) -> str:
        """
        Format OCR result as text with inline bbox metadata.

        Args:
            ocr_result: OCRDocumentResult from PaddleOCRAdapter
            file_metadata: File metadata dictionary

        Returns:
            Formatted text content with bbox metadata
        """
        content_parts = []

        for page_result in ocr_result.pages:
            content_parts.append(f"Page {page_result.page_number}:")

            for element in page_result.elements:
                # Format: "text [page=N, bbox=(x,y,x,y), type=text, confidence=C]"
                bbox = element.bbox
                formatted_line = (
                    f"{element.text} "
                    f"[page={element.page_number}, "
                    f"bbox=({bbox.x_min:.3f},{bbox.y_min:.3f},"
                    f"{bbox.x_max:.3f},{bbox.y_max:.3f}), "
                    f"type=text, "
                    f"confidence={element.confidence:.3f}]"
                )
                content_parts.append(formatted_line)

            content_parts.append("")  # Blank line between pages

        return "\n".join(content_parts)

    async def load(
        self,
        file_path: str,
        lang: str = "en",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        force_ocr: bool = False,
        **kwargs,
    ) -> str:
        """
        Load PDF file with automatic type detection and OCR.

        Args:
            file_path: Path to the PDF file
            lang: OCR language code (default: 'en')
            use_gpu: Whether to use GPU for OCR (default: False)
            min_confidence: Minimum OCR confidence threshold (default: 0.5)
            force_ocr: Force OCR even for digital PDFs (default: False)
            **kwargs: Additional arguments

        Returns:
            Path to stored file with layout-aware text

        Raises:
            ImportError: If required libraries not installed
            Exception: If PDF processing fails
        """
        try:
            from cognee.infrastructure.loaders.utils.pdf_type_detector import (
                detect_pdf_type,
                PDFType,
            )
            from cognee.infrastructure.ocr import PaddleOCRAdapter, is_paddleocr_available

            # Detect PDF type (unless forcing OCR)
            if not force_ocr:
                pdf_type = detect_pdf_type(file_path)
                logger.info(f"Detected PDF type: {pdf_type.value}")

                # If digital PDF, delegate to PdfPlumberLoader for faster processing
                if pdf_type == PDFType.DIGITAL:
                    logger.info("Using PdfPlumberLoader for digital PDF")
                    try:
                        from .pdfplumber_loader import PdfPlumberLoader

                        pdfplumber_loader = PdfPlumberLoader()
                        return await pdfplumber_loader.load(file_path, **kwargs)
                    except Exception as e:
                        logger.warning(
                            f"PdfPlumberLoader failed: {e}, falling back to OCR"
                        )
                        # Continue to OCR fallback
            else:
                logger.info("Forcing OCR processing")

            # Check if PaddleOCR is available
            if not is_paddleocr_available():
                raise ImportError(
                    "PaddleOCR is required for scanned PDF processing. "
                    "Install with: pip install cognee[ocr]"
                )

            # Use OCR for scanned/hybrid PDFs
            logger.info(f"Processing PDF with OCR (lang={lang}, use_gpu={use_gpu})")

            with open(file_path, "rb") as file:
                file_metadata = await get_file_metadata(file)
                # Name file based on hash with OCR prefix
                storage_file_name = "ocr_text_" + file_metadata["content_hash"] + ".txt"

                # Initialize OCR adapter
                ocr_adapter = PaddleOCRAdapter(
                    lang=lang,
                    use_gpu=use_gpu,
                    min_confidence=min_confidence,
                )

                # Process PDF with OCR
                ocr_result = await ocr_adapter.process_pdf(file_path)

                # Format result
                full_content = self._format_ocr_result(ocr_result, file_metadata)

                if not full_content.strip():
                    logger.warning(
                        f"No content extracted from {file_path} with OCR, "
                        "trying fallback loaders"
                    )
                    # Fallback to pypdf
                    return await self._fallback_to_pypdf(file_path)

                # Store result
                storage_config = get_storage_config()
                data_root_directory = storage_config["data_root_directory"]
                storage = get_file_storage(data_root_directory)

                full_file_path = await storage.store(storage_file_name, full_content)

                logger.info(
                    f"Successfully processed PDF with OCR: {file_path} -> {full_file_path}"
                )
                return full_file_path

        except ImportError as e:
            logger.error(f"Required libraries not available: {e}")
            logger.info("Falling back to PyPdfLoader")
            return await self._fallback_to_pypdf(file_path)

        except Exception as e:
            logger.error(f"Failed to process PDF {file_path} with OCR: {e}")
            logger.info("Falling back to PyPdfLoader")
            return await self._fallback_to_pypdf(file_path)

    async def _fallback_to_pypdf(self, file_path: str) -> str:
        """
        Fallback to PyPdfLoader if OCR fails.

        Args:
            file_path: Path to PDF file

        Returns:
            Path to stored file
        """
        try:
            from .pypdf_loader import PyPdfLoader

            logger.info("Using PyPdfLoader as fallback")
            pypdf_loader = PyPdfLoader()
            return await pypdf_loader.load(file_path)
        except Exception as e:
            logger.error(f"PyPdfLoader fallback also failed: {e}")
            raise Exception(f"All PDF loading methods failed for {file_path}") from e
