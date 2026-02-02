"""PDF loader with OCR support for scanned PDFs."""

import json
from datetime import datetime, timezone
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

    Output is stored as structured JSON with cognee_ocr_format header.
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

    def _extract_plain_text(self, ocr_result) -> str:
        """
        Extract clean plain text from multi-page OCR result.

        Args:
            ocr_result: OCRDocumentResult from PaddleOCRAdapter

        Returns:
            Plain text content without metadata
        """
        page_texts = []

        for page_result in ocr_result.pages:
            texts = []

            # Use layout_elements if available (layout-aware mode)
            if page_result.layout_elements:
                for layout_elem in page_result.layout_elements:
                    for element in layout_elem.text_elements:
                        texts.append(element.text)
            else:
                # Use flat elements
                for element in page_result.elements:
                    texts.append(element.text)

            if texts:
                page_texts.append(" ".join(texts))

        return "\n\n".join(page_texts)

    def _build_ocr_output(
        self,
        ocr_result,
        file_metadata: dict,
        use_structure: bool = False,
    ) -> dict:
        """
        Build structured OCR output document for multi-page PDF.

        Args:
            ocr_result: OCRDocumentResult from PaddleOCRAdapter
            file_metadata: File metadata dictionary
            use_structure: Whether layout-aware OCR was used

        Returns:
            Dictionary conforming to OCROutputDocument schema
        """
        from cognee.infrastructure.ocr.models import (
            OCRFormatType,
            OCR_FORMAT_VERSION,
        )

        # Check if any page has layout elements to determine format type
        has_any_layout = any(
            page.layout_elements is not None and len(page.layout_elements) > 0
            for page in ocr_result.pages
        )
        format_type = OCRFormatType.BLOCK if has_any_layout else OCRFormatType.FLAT

        # Build source info
        source = {
            "loader": self.loader_name,
            "ocr_engine": "paddleocr",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "use_structure": use_structure,
        }

        # Build document info
        document = {
            "total_pages": ocr_result.total_pages,
            "content_hash": file_metadata.get("content_hash"),
            "source_filename": file_metadata.get("name"),
        }

        # Build page outputs
        pages = []
        for page_result in ocr_result.pages:
            page_output = {
                "page_number": page_result.page_number,
                "width": page_result.page_width,
                "height": page_result.page_height,
            }

            has_layout = (
                page_result.layout_elements is not None and len(page_result.layout_elements) > 0
            )

            if has_layout:
                # Block format with layout blocks
                layout_blocks = []
                for layout_elem in page_result.layout_elements:
                    block = {
                        "layout_type": layout_elem.layout_type,
                        "bbox": {
                            "x_min": layout_elem.bbox.x_min,
                            "y_min": layout_elem.bbox.y_min,
                            "x_max": layout_elem.bbox.x_max,
                            "y_max": layout_elem.bbox.y_max,
                        },
                        "confidence": layout_elem.confidence,
                        "elements": [],
                    }

                    for elem in layout_elem.text_elements:
                        block["elements"].append(
                            {
                                "text": elem.text,
                                "bbox": {
                                    "x_min": elem.bbox.x_min,
                                    "y_min": elem.bbox.y_min,
                                    "x_max": elem.bbox.x_max,
                                    "y_max": elem.bbox.y_max,
                                },
                                "confidence": elem.confidence,
                                "layout_type": elem.layout_type,
                            }
                        )

                    layout_blocks.append(block)

                page_output["layout_blocks"] = layout_blocks
            else:
                # Flat format with elements
                elements = []
                for elem in page_result.elements:
                    elements.append(
                        {
                            "text": elem.text,
                            "bbox": {
                                "x_min": elem.bbox.x_min,
                                "y_min": elem.bbox.y_min,
                                "x_max": elem.bbox.x_max,
                                "y_max": elem.bbox.y_max,
                            },
                            "confidence": elem.confidence,
                            "layout_type": getattr(elem, "layout_type", "text"),
                        }
                    )

                page_output["elements"] = elements

            pages.append(page_output)

        # Extract plain text for embedding
        plain_text = self._extract_plain_text(ocr_result)

        return {
            "cognee_ocr_format": OCR_FORMAT_VERSION,
            "format_type": format_type.value,
            "source": source,
            "document": document,
            "pages": pages,
            "plain_text": plain_text,
        }

    async def load(
        self,
        file_path: str,
        lang: str = "en",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        force_ocr: bool = False,
        use_structure: bool = True,
        structure_config: dict = None,
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
            use_structure: Whether to use PPStructureV3 for layout-aware OCR (default: True)
            structure_config: Optional configuration dict for PPStructureV3 (default: None)
            **kwargs: Additional arguments

        Returns:
            Path to stored JSON file with layout-aware text and metadata

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
                        logger.warning(f"PdfPlumberLoader failed: {e}, falling back to OCR")
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
                # Name file based on hash with OCR prefix - now using .json extension
                storage_file_name = "ocr_" + file_metadata["content_hash"] + ".json"

                # Initialize OCR adapter
                ocr_adapter = PaddleOCRAdapter(
                    lang=lang,
                    use_gpu=use_gpu,
                    min_confidence=min_confidence,
                    use_structure=use_structure,
                    structure_config=structure_config,
                )

                # Process PDF with OCR
                ocr_result = await ocr_adapter.process_pdf(file_path)

                # Build structured output
                ocr_output = self._build_ocr_output(
                    ocr_result,
                    file_metadata,
                    use_structure=use_structure,
                )

                if not ocr_output["plain_text"].strip():
                    logger.warning(
                        f"No content extracted from {file_path} with OCR, trying fallback loaders"
                    )
                    # Fallback to pypdf
                    return await self._fallback_to_pypdf(file_path)

                # Store result as JSON
                storage_config = get_storage_config()
                data_root_directory = storage_config["data_root_directory"]
                storage = get_file_storage(data_root_directory)

                full_content = json.dumps(ocr_output, indent=2, ensure_ascii=False)
                full_file_path = await storage.store(storage_file_name, full_content)

                logger.info(f"Successfully processed PDF with OCR: {file_path} -> {full_file_path}")
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
