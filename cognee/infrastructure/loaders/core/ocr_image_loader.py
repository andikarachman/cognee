"""Image loader with OCR support."""

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

    def _format_ocr_result(self, page_result, file_metadata: dict) -> str:
        """
        Format OCR result as text with inline bbox metadata.

        Args:
            page_result: OCRPageResult from PaddleOCRAdapter
            file_metadata: File metadata dictionary

        Returns:
            Formatted text content with bbox metadata
        """
        content_parts = []

        for element in page_result.elements:
            # Format: "text [page=1, bbox=(x,y,x,y), type=TYPE, confidence=C]"
            bbox = element.bbox
            # Use layout_type from element (defaults to 'text')
            layout_type = getattr(element, 'layout_type', 'text')
            formatted_line = (
                f"{element.text} "
                f"[page=1, "
                f"bbox=({bbox.x_min:.3f},{bbox.y_min:.3f},"
                f"{bbox.x_max:.3f},{bbox.y_max:.3f}), "
                f"type={layout_type}, "
                f"confidence={element.confidence:.3f}]"
            )
            content_parts.append(formatted_line)

        return "\n".join(content_parts)

    async def load(
        self,
        file_path: str,
        lang: str = "en",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        disable_llm_fallback: bool = False,
        use_structure: bool = False,
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
            use_structure: Whether to use PPStructureV3 for layout-aware OCR (default: False)
            structure_config: Optional configuration dict for PPStructureV3 (default: None)
            **kwargs: Additional arguments

        Returns:
            Path to stored file with OCR-extracted text

        Raises:
            ImportError: If PaddleOCR not installed
            Exception: If image processing fails
        """
        try:
            from cognee.infrastructure.ocr import PaddleOCRAdapter, is_paddleocr_available

            # Check if PaddleOCR is available
            if not is_paddleocr_available():
                raise ImportError(
                    "PaddleOCR is required for image OCR. "
                    "Install with: pip install cognee[ocr]"
                )

            logger.info(f"Processing image with OCR: {file_path}")

            with open(file_path, "rb") as file:
                file_metadata = await get_file_metadata(file)
                # Name file based on hash with OCR prefix
                storage_file_name = "ocr_text_" + file_metadata["content_hash"] + ".txt"

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

                # Format result
                full_content = self._format_ocr_result(page_result, file_metadata)

                if not full_content.strip():
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

                # Store result
                storage_config = get_storage_config()
                data_root_directory = storage_config["data_root_directory"]
                storage = get_file_storage(data_root_directory)

                full_file_path = await storage.store(storage_file_name, full_content)

                logger.info(
                    f"Successfully processed image with OCR: "
                    f"{file_path} -> {full_file_path}"
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
