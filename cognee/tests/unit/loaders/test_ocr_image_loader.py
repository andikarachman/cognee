"""Unit tests for OcrImageLoader (image OCR loader)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from cognee.infrastructure.loaders.core.ocr_image_loader import OcrImageLoader
from cognee.tests.test_data.mock_data.mock_ocr_results import (
    create_mock_ocr_page_result,
)


class TestImageOCRProcessing:
    """Tests for image OCR processing."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_process_image_success(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class
    ):
        """Test successful image OCR processing."""
        # Mock OCR adapter
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_page_result(page_number=1, num_elements=5)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "test.png"}

        loader = OcrImageLoader()
        result = await loader.load("test.png")

        # Should use OCR
        mock_ocr_instance.process_image.assert_called_once_with("test.png", page_number=1)

        # Should store result
        mock_storage.store.assert_called_once()

        # Verify filename format
        call_args = mock_storage.store.call_args
        assert "ocr_text_" in call_args[1]["filename"]
        assert call_args[1]["filename"].endswith(".txt")

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_ocr_text_formatting(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class
    ):
        """Test that OCR results are formatted with metadata markers."""
        # Mock OCR
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_page_result(page_number=1, num_elements=3)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage to capture formatted text
        formatted_text = None

        async def store_side_effect(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[0].decode("utf-8") if isinstance(args[0], bytes) else args[0]
            return "stored_file_id"

        mock_storage = AsyncMock()
        mock_storage.store.side_effect = store_side_effect
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "test.png"}

        loader = OcrImageLoader()
        await loader.load("test.png")

        # Verify formatted text structure
        assert formatted_text is not None
        assert "[page=1" in formatted_text  # Images are treated as single page
        assert "bbox=" in formatted_text
        assert "type=" in formatted_text
        assert "confidence=" in formatted_text

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_empty_ocr_result(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class
    ):
        """Test handling of empty OCR results."""
        # Mock OCR returning empty result
        mock_ocr_instance = Mock()
        from cognee.infrastructure.ocr.PaddleOCRAdapter import OCRPageResult

        mock_ocr_instance.process_image.return_value = OCRPageResult(
            page_number=1,
            elements=[],
            page_width=1000,
            page_height=1000,
        )
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "blank.png"}

        loader = OcrImageLoader()
        result = await loader.load("blank.png")

        # Should still store (empty result)
        mock_storage.store.assert_called_once()


class TestFallbackToImageLoader:
    """Tests for fallback to ImageLoader (LLM vision)."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.ImageLoader")
    async def test_ocr_failure_fallback_to_image_loader(
        self, mock_image_loader_class, mock_ocr_class
    ):
        """Test fallback to ImageLoader when OCR fails."""
        # Mock OCR failure
        mock_ocr_instance = Mock()
        mock_ocr_instance.process_image.side_effect = Exception("OCR failed")
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock ImageLoader fallback
        mock_image_loader_instance = AsyncMock()
        mock_image_loader_instance.load.return_value = ["llm_vision_result.txt"]
        mock_image_loader_class.return_value = mock_image_loader_instance

        loader = OcrImageLoader()
        result = await loader.load("test.png")

        # Should fallback to ImageLoader
        mock_image_loader_instance.load.assert_called_once_with("test.png")
        assert result == ["llm_vision_result.txt"]

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.ImageLoader")
    async def test_import_error_fallback(
        self, mock_image_loader_class, mock_ocr_class
    ):
        """Test fallback when PaddleOCR is not available."""
        # Mock import error
        mock_ocr_class.side_effect = ImportError("PaddleOCR not installed")

        # Mock ImageLoader fallback
        mock_image_loader_instance = AsyncMock()
        mock_image_loader_instance.load.return_value = ["fallback_result.txt"]
        mock_image_loader_class.return_value = mock_image_loader_instance

        loader = OcrImageLoader()
        result = await loader.load("test.png")

        # Should fallback to ImageLoader
        mock_image_loader_instance.load.assert_called_once()
        assert result == ["fallback_result.txt"]


class TestImageFormatsSupport:
    """Tests for various image format support."""

    @pytest.mark.parametrize(
        "filename,expected_handle",
        [
            ("test.png", True),
            ("test.PNG", True),
            ("test.jpg", True),
            ("test.jpeg", True),
            ("test.JPEG", True),
            ("test.tiff", True),
            ("test.tif", True),
            ("test.bmp", True),
            ("test.pdf", False),
            ("test.txt", False),
            ("test.docx", False),
        ],
    )
    def test_can_handle_image_formats(self, filename, expected_handle):
        """Test can_handle method for various image formats."""
        loader = OcrImageLoader()
        assert loader.can_handle(filename) == expected_handle

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_process_different_formats(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class
    ):
        """Test processing different image formats."""
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_page_result()
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        loader = OcrImageLoader()

        # Test various formats
        for ext in ["png", "jpg", "jpeg", "tiff"]:
            filename = f"test.{ext}"
            mock_get_metadata.return_value = {"name": filename}

            await loader.load(filename)

            # Should process each format
            mock_ocr_instance.process_image.assert_called_with(filename, page_number=1)


class TestLoaderConfiguration:
    """Tests for loader configuration."""

    def test_loader_name(self):
        """Test loader has correct name."""
        loader = OcrImageLoader()
        assert loader.loader_name == "ocr_image_loader"

    def test_loader_initialization_with_config(self):
        """Test loader initialization with custom config."""
        loader = OcrImageLoader(
            min_confidence=0.9,
            use_gpu=True,
            lang="es",
        )

        assert loader.min_confidence == 0.9
        assert loader.use_gpu is True
        assert loader.lang == "es"

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_custom_confidence_filtering(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class
    ):
        """Test that custom confidence threshold is passed to OCR adapter."""
        # Mock OCR adapter class to capture initialization
        captured_min_confidence = None

        def ocr_init_side_effect(*args, **kwargs):
            nonlocal captured_min_confidence
            captured_min_confidence = kwargs.get("min_confidence")
            mock_instance = Mock()
            mock_instance.process_image.return_value = create_mock_ocr_page_result()
            return mock_instance

        mock_ocr_class.side_effect = ocr_init_side_effect

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "test.png"}

        loader = OcrImageLoader(min_confidence=0.85)
        await loader.load("test.png")

        # Verify custom confidence was passed to OCR adapter
        assert captured_min_confidence == 0.85


class TestOutputFormat:
    """Tests for output format consistency."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_output_format_consistency(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class
    ):
        """Test that output format is consistent with other OCR loaders."""
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_page_result(num_elements=2)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        formatted_text = None

        async def store_side_effect(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[0].decode("utf-8") if isinstance(args[0], bytes) else args[0]
            return "stored_file_id"

        mock_storage = AsyncMock()
        mock_storage.store.side_effect = store_side_effect
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "test.png"}

        loader = OcrImageLoader()
        await loader.load("test.png")

        # Output should have same format as OcrPdfLoader
        assert "[page=1" in formatted_text
        assert "bbox=(" in formatted_text
        assert "type=" in formatted_text
        assert "]" in formatted_text  # Closing bracket for metadata
