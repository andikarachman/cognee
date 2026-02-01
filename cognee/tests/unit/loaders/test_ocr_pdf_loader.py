"""Unit tests for OcrPdfLoader (scanned PDF loader with OCR)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from cognee.infrastructure.loaders.external.ocr_pdf_loader import OcrPdfLoader
from cognee.infrastructure.loaders.utils.pdf_type_detector import PDFType
from cognee.tests.test_data.mock_data.mock_ocr_results import create_mock_ocr_result


class TestPDFTypeDelegation:
    """Tests for PDF type detection and delegation logic."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.PdfPlumberLoader")
    async def test_digital_pdf_delegates_to_pdfplumber(
        self, mock_pdfplumber_class, mock_detect
    ):
        """Test that digital PDFs delegate to PdfPlumberLoader."""
        mock_detect.return_value = PDFType.DIGITAL

        # Mock PdfPlumberLoader instance
        mock_pdfplumber_instance = AsyncMock()
        mock_pdfplumber_instance.load.return_value = ["layout_text_result.txt"]
        mock_pdfplumber_class.return_value = mock_pdfplumber_instance

        loader = OcrPdfLoader()
        result = await loader.load("digital.pdf")

        # Should delegate to PdfPlumberLoader
        mock_pdfplumber_instance.load.assert_called_once_with("digital.pdf")
        assert result == ["layout_text_result.txt"]

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_scanned_pdf_uses_ocr(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_detect, mock_file_open
    ):
        """Test that scanned PDFs use PaddleOCRAdapter."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR adapter
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1, elements_per_page=3)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "scanned.pdf",
            "file_path": "scanned.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "content_hash": "scanned_hash_123",
            "file_size": 1024,
        }

        loader = OcrPdfLoader()
        await loader.load("scanned.pdf")

        # Should use OCR
        mock_ocr_instance.process_pdf.assert_called_once_with("scanned.pdf")

        # Should store result
        mock_storage.store.assert_called_once()

        # Verify filename format (first positional argument)
        call_args = mock_storage.store.call_args
        filename = call_args[0][0]  # First positional argument is the filename
        assert "ocr_text_" in filename

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_hybrid_pdf_uses_ocr(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_is_available, mock_detect, mock_file_open
    ):
        """Test that hybrid PDFs use OCR (safe default)."""
        mock_detect.return_value = PDFType.HYBRID

        # Mock OCR
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {
            "name": "hybrid.pdf",
            "file_path": "hybrid.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "content_hash": "hybrid_hash_123",
            "file_size": 1024,
        }

        loader = OcrPdfLoader()
        await loader.load("hybrid.pdf")

        # Should use OCR for hybrid PDFs
        mock_ocr_instance.process_pdf.assert_called_once()


class TestOCRProcessing:
    """Tests for OCR text extraction and formatting."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_ocr_text_formatting(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_detect, mock_file_open
    ):
        """Test that OCR results are formatted with metadata markers."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR result
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_result(num_pages=2, elements_per_page=2)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
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
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "test.pdf",
            "file_path": "test.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "content_hash": "test_hash_123",
            "file_size": 1024,
        }

        loader = OcrPdfLoader()
        await loader.load("test.pdf")

        # Verify formatted text has correct structure
        assert formatted_text is not None
        assert "[page=" in formatted_text
        assert "bbox=" in formatted_text
        assert "type=" in formatted_text
        assert "confidence=" in formatted_text

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_multi_page_ocr(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_detect, mock_file_open
    ):
        """Test OCR processing of multiple pages."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR with 3 pages
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_result(num_pages=3, elements_per_page=5)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "multi.pdf",
            "file_path": "multi.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "content_hash": "multi_hash_123",
            "file_size": 1024,
        }

        loader = OcrPdfLoader()
        await loader.load("multi.pdf")

        # Should process all pages
        mock_ocr_instance.process_pdf.assert_called_once()


class TestFallbackLogic:
    """Tests for fallback chain when OCR fails."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.PyPdfLoader")
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.PdfPlumberLoader")
    async def test_ocr_failure_fallback_to_pdfplumber(
        self, mock_pdfplumber_class, mock_ocr_class, mock_is_available, mock_detect, mock_pypdf_loader_class, mock_pypdf_open, mock_file_open
    ):
        """Test fallback to PdfPlumberLoader when OCR fails."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR failure
        mock_ocr_instance = AsyncMock()
        mock_ocr_instance.process_pdf.side_effect = Exception("OCR failed")
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock PyPdfLoader as fallback when OCR fails
        mock_pypdf_instance = AsyncMock()
        mock_pypdf_instance.load.return_value = ["fallback_result.txt"]
        mock_pypdf_loader_class.return_value = mock_pypdf_instance

        loader = OcrPdfLoader()
        result = await loader.load("test.pdf")

        # Should fallback to PyPdfLoader
        mock_pypdf_instance.load.assert_called_once_with("test.pdf")
        assert result == ["fallback_result.txt"]

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.PdfPlumberLoader")
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.PyPdfLoader")
    async def test_double_fallback_to_pypdf(
        self, mock_pypdf_class, mock_pdfplumber_class, mock_ocr_class, mock_is_available, mock_detect
    ):
        """Test fallback to PyPdfLoader when both OCR and PdfPlumber fail."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR failure
        mock_ocr_instance = AsyncMock()
        mock_ocr_instance.process_pdf.side_effect = Exception("OCR failed")
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock PdfPlumber failure
        mock_pdfplumber_instance = AsyncMock()
        mock_pdfplumber_instance.load.side_effect = Exception("PdfPlumber failed")
        mock_pdfplumber_class.return_value = mock_pdfplumber_instance

        # Mock PyPdf fallback
        mock_pypdf_instance = AsyncMock()
        mock_pypdf_instance.load.return_value = ["final_fallback.txt"]
        mock_pypdf_class.return_value = mock_pypdf_instance

        loader = OcrPdfLoader()
        result = await loader.load("test.pdf")

        # Should fallback to PyPdfLoader
        mock_pypdf_instance.load.assert_called_once()
        assert result == ["final_fallback.txt"]

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.PyPdfLoader")
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_storage_config")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_empty_ocr_result_fallback(
        self, mock_get_metadata, mock_get_storage, mock_get_config, mock_ocr_class, mock_is_available, mock_detect, mock_pypdf_loader_class, mock_pypdf_open, mock_file_open
    ):
        """Test fallback when OCR returns empty results."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR returning empty result
        mock_ocr_instance = AsyncMock()
        from cognee.infrastructure.ocr.PaddleOCRAdapter import OCRDocumentResult

        mock_ocr_instance.process_pdf.return_value = OCRDocumentResult(
            pages=[], total_pages=0
        )
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock PyPdfLoader fallback
        mock_pypdf_instance = AsyncMock()
        mock_pypdf_instance.load.return_value = "fallback_result.txt"
        mock_pypdf_loader_class.return_value = mock_pypdf_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_config.return_value = {"data_root_directory": "/tmp/test"}
        mock_get_metadata.return_value = {
            "name": "empty.pdf",
            "file_path": "empty.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "content_hash": "empty_hash_123",
            "file_size": 1024,
        }

        loader = OcrPdfLoader()
        result = await loader.load("empty.pdf")

        # Should fallback to PyPdfLoader
        mock_pypdf_instance.load.assert_called_once_with("empty.pdf")
        assert result == "fallback_result.txt"


class TestForceOCROption:
    """Tests for force_ocr parameter."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.open", new_callable=mock_open)
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.detect_pdf_type")
    @patch("cognee.infrastructure.ocr.is_paddleocr_available", return_value=True)
    @patch("cognee.infrastructure.ocr.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_force_ocr_bypasses_detection(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_is_available, mock_detect, mock_file_open
    ):
        """Test that force_ocr=True bypasses PDF type detection."""
        # Mock OCR
        mock_ocr_instance = AsyncMock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {
            "name": "test.pdf",
            "file_path": "test.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "content_hash": "test_hash_123",
            "file_size": 1024,
        }

        loader = OcrPdfLoader()
        await loader.load("test.pdf", force_ocr=True)

        # Should NOT call detect_pdf_type
        mock_detect.assert_not_called()

        # Should use OCR
        mock_ocr_instance.process_pdf.assert_called_once()


class TestLoaderMetadata:
    """Tests for loader metadata and capabilities."""

    def test_can_handle_method(self):
        """Test can_handle identifies PDF files."""
        loader = OcrPdfLoader()

        # PDF loader requires BOTH extension AND mime_type to match (uses 'and' logic)
        assert loader.can_handle("pdf", "application/pdf") is True
        # Either one alone is not enough
        assert loader.can_handle("", "application/pdf") is False
        assert loader.can_handle("pdf", "") is False
        # Non-PDF files
        assert loader.can_handle("txt", "text/plain") is False
        assert loader.can_handle("png", "image/png") is False

    def test_loader_name(self):
        """Test loader has correct name."""
        loader = OcrPdfLoader()
        assert loader.loader_name == "ocr_pdf_loader"

    def test_loader_initialization(self):
        """Test loader can be initialized without config (config goes to load() method)."""
        loader = OcrPdfLoader()

        # Loader should have properties
        assert loader.loader_name == "ocr_pdf_loader"
        assert len(loader.supported_extensions) > 0
        assert len(loader.supported_mime_types) > 0
