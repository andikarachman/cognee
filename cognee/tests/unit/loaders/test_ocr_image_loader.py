"""Unit tests for OcrImageLoader (image OCR loader)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, mock_open
from cognee.infrastructure.loaders.core.ocr_image_loader import OcrImageLoader
from cognee.tests.test_data.mock_data.mock_ocr_results import (
    create_mock_ocr_page_result,
)


class TestImageOCRProcessing:
    """Tests for image OCR processing."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_process_image_success(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_file_open
    ):
        """Test successful image OCR processing."""
        # Mock OCR adapter
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_page_result(page_number=1, num_elements=5)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "test.png",
            "file_path": "test.png",
            "mime_type": "image/png",
            "extension": ".png",
            "content_hash": "test_hash_123",
            "file_size": 1024,
        }

        loader = OcrImageLoader()
        await loader.load("test.png")

        # Should use OCR
        mock_ocr_instance.process_image.assert_called_once_with("test.png", page_number=1)

        # Should store result
        mock_storage.store.assert_called_once()

        # Verify filename format (first positional argument)
        call_args = mock_storage.store.call_args
        filename = call_args[0][0]  # First positional argument is the filename
        assert "ocr_text_" in filename
        assert filename.endswith(".txt")

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_ocr_text_formatting(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_file_open
    ):
        """Test that OCR results are formatted with metadata markers."""
        # Mock OCR
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_page_result(page_number=1, num_elements=3)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage to capture formatted text
        formatted_text = None

        async def store_side_effect(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1].decode("utf-8") if isinstance(args[1], bytes) else args[1]
            return "stored_file_id"

        mock_storage = AsyncMock()
        mock_storage.store.side_effect = store_side_effect
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {
            "name": "test.png",
            "file_path": "test.png",
            "mime_type": "image/png",
            "extension": ".png",
            "content_hash": "test_hash_123",
            "file_size": 1024,
        }

        loader = OcrImageLoader()
        await loader.load("test.png")

        # Verify formatted text structure
        assert formatted_text is not None
        assert "[page=1" in formatted_text  # Images are treated as single page
        assert "bbox=" in formatted_text
        assert "type=" in formatted_text
        assert "confidence=" in formatted_text

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    @patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader")
    async def test_empty_ocr_result(
        self, mock_image_loader_class, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_file_open
    ):
        """Test handling of empty OCR results."""
        # Mock OCR returning empty result
        mock_ocr_instance = AsyncMock()
        from cognee.infrastructure.ocr.PaddleOCRAdapter import OCRPageResult

        mock_ocr_instance.process_image.return_value = OCRPageResult(
            page_number=1,
            elements=[],
            page_width=1000,
            page_height=1000,
        )
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock ImageLoader fallback
        mock_image_loader_instance = AsyncMock()
        mock_image_loader_instance.load.return_value = "fallback_result.txt"
        mock_image_loader_class.return_value = mock_image_loader_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "blank.png",
            "file_path": "blank.png",
            "mime_type": "image/png",
            "extension": ".png",
            "content_hash": "blank_hash_123",
            "file_size": 1024,
        }

        loader = OcrImageLoader()
        result = await loader.load("blank.png")

        # Should fallback to ImageLoader
        mock_image_loader_instance.load.assert_called_once_with("blank.png")
        assert result == "fallback_result.txt"


class TestFallbackToImageLoader:
    """Tests for fallback to ImageLoader (LLM vision)."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader")
    async def test_ocr_failure_fallback_to_image_loader(
        self, mock_image_loader_class, mock_ocr_class, mock_is_available
    ):
        """Test fallback to ImageLoader when OCR fails."""
        # Mock OCR failure
        mock_ocr_instance = AsyncMock()
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
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=False)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader")
    async def test_import_error_fallback(
        self, mock_image_loader_class, mock_ocr_class, mock_is_available
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
        "extension,mime_type,expected_handle",
        [
            ("png", "image/png", True),
            ("PNG", "image/png", True),
            ("jpg", "image/jpeg", True),
            ("jpeg", "image/jpeg", True),
            ("JPEG", "image/jpeg", True),
            ("tiff", "image/tiff", True),
            ("tif", "image/tiff", True),
            ("bmp", "image/bmp", True),
            ("pdf", "application/pdf", False),
            ("txt", "text/plain", False),
            ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", False),
        ],
    )
    def test_can_handle_image_formats(self, extension, mime_type, expected_handle):
        """Test can_handle method for various image formats."""
        loader = OcrImageLoader()
        assert loader.can_handle(extension, mime_type) == expected_handle

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_process_different_formats(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_file_open
    ):
        """Test processing different image formats."""
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_page_result()
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}

        loader = OcrImageLoader()

        # Test various formats
        for ext in ["png", "jpg", "jpeg", "tiff"]:
            filename = f"test.{ext}"
            mock_get_metadata.return_value = {
                "name": filename,
                "file_path": filename,
                "mime_type": "image/png",
                "extension": "." + filename.split(".")[-1],
                "content_hash": f"{filename}_hash",
                "file_size": 1024,
            }

            await loader.load(filename)

            # Should process each format
            mock_ocr_instance.process_image.assert_called_with(filename, page_number=1)


class TestLoaderConfiguration:
    """Tests for loader configuration."""

    def test_loader_name(self):
        """Test loader has correct name."""
        loader = OcrImageLoader()
        assert loader.loader_name == "ocr_image_loader"

    def test_loader_initialization(self):
        """Test loader can be initialized without config (config goes to load() method)."""
        loader = OcrImageLoader()

        # Loader should have properties
        assert loader.loader_name == "ocr_image_loader"
        assert len(loader.supported_extensions) > 0
        assert len(loader.supported_mime_types) > 0

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_custom_confidence_filtering(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_is_available, mock_file_open
    ):
        """Test that custom confidence threshold is passed to OCR adapter."""
        # Mock OCR adapter class to capture initialization
        captured_min_confidence = None

        def ocr_init_side_effect(*args, **kwargs):
            nonlocal captured_min_confidence
            captured_min_confidence = kwargs.get("min_confidence")
            mock_instance = AsyncMock()
            mock_instance.process_image.return_value = create_mock_ocr_page_result()
            return mock_instance

        mock_ocr_class.side_effect = ocr_init_side_effect

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {
            "name": "test.png",
            "file_path": "test.png",
            "mime_type": "image/png",
            "extension": ".png",
            "content_hash": "test_hash_123",
            "file_size": 1024,
        }

        loader = OcrImageLoader()
        await loader.load("test.png", min_confidence=0.85)

        # Verify custom confidence was passed to OCR adapter
        assert captured_min_confidence == 0.85


class TestOutputFormat:
    """Tests for output format consistency."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata")
    async def test_output_format_consistency(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_file_open
    ):
        """Test that output format is consistent with other OCR loaders."""
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_page_result(num_elements=2)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        formatted_text = None

        async def store_side_effect(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1].decode("utf-8") if isinstance(args[1], bytes) else args[1]
            return "stored_file_id"

        mock_storage = AsyncMock()
        mock_storage.store.side_effect = store_side_effect
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "test.png",
            "file_path": "test.png",
            "mime_type": "image/png",
            "extension": ".png",
            "content_hash": "test_hash_123",
            "file_size": 1024,
        }

        loader = OcrImageLoader()
        await loader.load("test.png")

        # Output should have same format as OcrPdfLoader
        assert "[page=1" in formatted_text
        assert "bbox=(" in formatted_text
        assert "type=" in formatted_text
        assert "]" in formatted_text  # Closing bracket for metadata
