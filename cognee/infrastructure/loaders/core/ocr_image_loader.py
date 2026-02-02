"""Image loader with OCR support."""

import json
from datetime import datetime, timezone
from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata

logger = get_logger(__name__)


class OcrImageLoader(LoaderInterface):
    """
    Image loader with OCR support.

    Uses PaddleOCR to extract text and bounding boxes from images
    (PNG, JPG, JPEG, TIFF, BMP, etc.). Falls back to ImageLoader
    (LLM vision) if OCR fails or returns empty results.

    Output is stored as structured JSON with cognee_ocr_format header.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"]

    @property
    def supported_mime_types(self) -> List[str]:
        return [
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/bmp",
            "image/gif",
            "image/webp",
        ]

    @property
    def loader_name(self) -> str:
        return "ocr_image_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        if extension in self.supported_extensions or mime_type in self.supported_mime_types:
            return True
        return False

    def _extract_plain_text(self, page_result) -> str:
        """
        Extract clean plain text from OCR result.

        Args:
            page_result: OCRPageResult from PaddleOCRAdapter

        Returns:
            Plain text content without metadata
        """
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

        return " ".join(texts)

    def _build_ocr_output(
        self,
        page_result,
        file_metadata: dict,
        use_structure: bool = False,
    ) -> dict:
        """
        Build structured OCR output document.

        Args:
            page_result: OCRPageResult from PaddleOCRAdapter
            file_metadata: File metadata dictionary
            use_structure: Whether layout-aware OCR was used

        Returns:
            Dictionary conforming to OCROutputDocument schema
        """
        from cognee.infrastructure.ocr.models import (
            OCRFormatType,
            OCR_FORMAT_VERSION,
        )

        # Determine format type based on whether layout_elements are present
        has_layout = (
            page_result.layout_elements is not None and len(page_result.layout_elements) > 0
        )
        format_type = OCRFormatType.BLOCK if has_layout else OCRFormatType.FLAT

        # Build source info
        source = {
            "loader": self.loader_name,
            "ocr_engine": "paddleocr",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "use_structure": use_structure,
        }

        # Build document info
        document = {
            "total_pages": 1,
            "content_hash": file_metadata.get("content_hash"),
            "source_filename": file_metadata.get("name"),
        }

        # Build page output
        page_output = {
            "page_number": page_result.page_number,
            "width": page_result.page_width,
            "height": page_result.page_height,
        }

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

        # Extract plain text for embedding
        plain_text = self._extract_plain_text(page_result)

        return {
            "cognee_ocr_format": OCR_FORMAT_VERSION,
            "format_type": format_type.value,
            "source": source,
            "document": document,
            "pages": [page_output],
            "plain_text": plain_text,
        }

    async def load(
        self,
        file_path: str,
        lang: str = "en",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        disable_llm_fallback: bool = False,
        use_structure: bool = True,
        structure_config: dict = None,
        **kwargs,
    ) -> str:
        """
        Load image file and extract text with OCR.

        Args:
            file_path: Path to the image file
            lang: OCR language code (default: 'en')
            use_gpu: Whether to use GPU for OCR (default: False)
            min_confidence: Minimum OCR confidence threshold (default: 0.5)
            disable_llm_fallback: Disable LLM vision fallback for testing (default: False)
            use_structure: Whether to use PPStructureV3 for layout-aware OCR (default: True)
            structure_config: Optional configuration dict for PPStructureV3 (default: None)
            **kwargs: Additional arguments

        Returns:
            Path to stored JSON file with OCR-extracted text and metadata

        Raises:
            ImportError: If PaddleOCR not installed
            Exception: If image processing fails
        """
        try:
            from cognee.infrastructure.ocr import PaddleOCRAdapter, is_paddleocr_available

            # Check if PaddleOCR is available
            if not is_paddleocr_available():
                raise ImportError(
                    "PaddleOCR is required for image OCR. Install with: pip install cognee[ocr]"
                )

            logger.info(f"Processing image with OCR: {file_path}")

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

                # Process image with OCR
                page_result = await ocr_adapter.process_image(file_path, page_number=1)

                # Build structured output
                ocr_output = self._build_ocr_output(
                    page_result,
                    file_metadata,
                    use_structure=use_structure,
                )

                if not ocr_output["plain_text"].strip():
                    if disable_llm_fallback:
                        # Return empty string for testing purposes
                        logger.info(
                            f"No content extracted from {file_path} with OCR "
                            "(LLM fallback disabled for testing)"
                        )
                        return ""
                    else:
                        # Production behavior: fallback to LLM vision
                        logger.warning(
                            f"No content extracted from {file_path} with OCR, "
                            "trying fallback to LLM vision"
                        )
                        return await self._fallback_to_image_loader(file_path)

                # Store result as JSON
                storage_config = get_storage_config()
                data_root_directory = storage_config["data_root_directory"]
                storage = get_file_storage(data_root_directory)

                full_content = json.dumps(ocr_output, indent=2, ensure_ascii=False)
                full_file_path = await storage.store(storage_file_name, full_content)

                logger.info(
                    f"Successfully processed image with OCR: {file_path} -> {full_file_path}"
                )
                return full_file_path

        except ImportError as e:
            logger.error(f"PaddleOCR not available: {e}")
            logger.info("Falling back to ImageLoader (LLM vision)")
            return await self._fallback_to_image_loader(file_path)

        except Exception as e:
            logger.error(f"Failed to process image {file_path} with OCR: {e}")
            logger.info("Falling back to ImageLoader (LLM vision)")
            return await self._fallback_to_image_loader(file_path)

    async def _fallback_to_image_loader(self, file_path: str) -> str:
        """
        Fallback to ImageLoader (LLM vision) if OCR fails.

        Args:
            file_path: Path to image file

        Returns:
            Path to stored file
        """
        try:
            from cognee.infrastructure.loaders.core.image_loader import ImageLoader

            logger.info("Using ImageLoader (LLM vision) as fallback")
            image_loader = ImageLoader()
            return await image_loader.load(file_path)
        except Exception as e:
            logger.error(f"ImageLoader fallback also failed: {e}")
            raise Exception(f"All image loading methods failed for {file_path}") from e
