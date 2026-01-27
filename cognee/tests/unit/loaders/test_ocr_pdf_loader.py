"""Unit tests for OcrPdfLoader (scanned PDF loader with OCR)."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from cognee.infrastructure.loaders.external.ocr_pdf_loader import OcrPdfLoader
from cognee.infrastructure.loaders.utils.pdf_type_detector import PDFType
from cognee.tests.test_data.mock_data.mock_ocr_results import create_mock_ocr_result


class TestPDFTypeDelegation:
    """Tests for PDF type detection and delegation logic."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PdfPlumberLoader")
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
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_scanned_pdf_uses_ocr(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_detect
    ):
        """Test that scanned PDFs use PaddleOCRAdapter."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR adapter
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1, elements_per_page=3)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "scanned.pdf"}

        loader = OcrPdfLoader()
        result = await loader.load("scanned.pdf")

        # Should use OCR
        mock_ocr_instance.process_pdf.assert_called_once_with("scanned.pdf")

        # Should store result
        mock_storage.store.assert_called_once()

        # Verify filename format
        call_args = mock_storage.store.call_args
        assert "ocr_text_" in call_args[1]["filename"]

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_hybrid_pdf_uses_ocr(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_detect
    ):
        """Test that hybrid PDFs use OCR (safe default)."""
        mock_detect.return_value = PDFType.HYBRID

        # Mock OCR
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "hybrid.pdf"}

        loader = OcrPdfLoader()
        await loader.load("hybrid.pdf")

        # Should use OCR for hybrid PDFs
        mock_ocr_instance.process_pdf.assert_called_once()


class TestOCRProcessing:
    """Tests for OCR text extraction and formatting."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_ocr_text_formatting(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_detect
    ):
        """Test that OCR results are formatted with metadata markers."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR result
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_result(num_pages=2, elements_per_page=2)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
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
        mock_get_metadata.return_value = {"name": "test.pdf"}

        loader = OcrPdfLoader()
        await loader.load("test.pdf")

        # Verify formatted text has correct structure
        assert formatted_text is not None
        assert "[page=" in formatted_text
        assert "bbox=" in formatted_text
        assert "type=" in formatted_text
        assert "confidence=" in formatted_text

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_multi_page_ocr(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_detect
    ):
        """Test OCR processing of multiple pages."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR with 3 pages
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_result(num_pages=3, elements_per_page=5)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "multi.pdf"}

        loader = OcrPdfLoader()
        await loader.load("multi.pdf")

        # Should process all pages
        mock_ocr_instance.process_pdf.assert_called_once()


class TestFallbackLogic:
    """Tests for fallback chain when OCR fails."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PdfPlumberLoader")
    async def test_ocr_failure_fallback_to_pdfplumber(
        self, mock_pdfplumber_class, mock_ocr_class, mock_detect
    ):
        """Test fallback to PdfPlumberLoader when OCR fails."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR failure
        mock_ocr_instance = Mock()
        mock_ocr_instance.process_pdf.side_effect = Exception("OCR failed")
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock PdfPlumberLoader fallback
        mock_pdfplumber_instance = AsyncMock()
        mock_pdfplumber_instance.load.return_value = ["fallback_result.txt"]
        mock_pdfplumber_class.return_value = mock_pdfplumber_instance

        loader = OcrPdfLoader()
        result = await loader.load("test.pdf")

        # Should fallback to PdfPlumberLoader
        mock_pdfplumber_instance.load.assert_called_once()
        assert result == ["fallback_result.txt"]

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PdfPlumberLoader")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PyPdfLoader")
    async def test_double_fallback_to_pypdf(
        self, mock_pypdf_class, mock_pdfplumber_class, mock_ocr_class, mock_detect
    ):
        """Test fallback to PyPdfLoader when both OCR and PdfPlumber fail."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR failure
        mock_ocr_instance = Mock()
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
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_empty_ocr_result_fallback(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_detect
    ):
        """Test fallback when OCR returns empty results."""
        mock_detect.return_value = PDFType.SCANNED

        # Mock OCR returning empty result
        mock_ocr_instance = Mock()
        from cognee.infrastructure.ocr.PaddleOCRAdapter import OCRDocumentResult

        mock_ocr_instance.process_pdf.return_value = OCRDocumentResult(
            pages=[], total_pages=0
        )
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "empty.pdf"}

        loader = OcrPdfLoader()

        # Empty OCR result should still store (empty text)
        await loader.load("empty.pdf")
        mock_storage.store.assert_called_once()


class TestForceOCROption:
    """Tests for force_ocr parameter."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.detect_pdf_type")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.get_file_metadata")
    async def test_force_ocr_bypasses_detection(
        self, mock_get_metadata, mock_get_storage, mock_ocr_class, mock_detect
    ):
        """Test that force_ocr=True bypasses PDF type detection."""
        # Mock OCR
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage
        mock_get_metadata.return_value = {"name": "test.pdf"}

        loader = OcrPdfLoader(force_ocr=True)
        await loader.load("test.pdf")

        # Should NOT call detect_pdf_type
        mock_detect.assert_not_called()

        # Should use OCR
        mock_ocr_instance.process_pdf.assert_called_once()


class TestLoaderMetadata:
    """Tests for loader metadata and capabilities."""

    def test_can_handle_method(self):
        """Test can_handle identifies PDF files."""
        loader = OcrPdfLoader()

        assert loader.can_handle("test.pdf") is True
        assert loader.can_handle("test.PDF") is True
        assert loader.can_handle("test.txt") is False
        assert loader.can_handle("test.png") is False

    def test_loader_name(self):
        """Test loader has correct name."""
        loader = OcrPdfLoader()
        assert loader.loader_name == "ocr_pdf_loader"

    def test_loader_initialization_with_config(self):
        """Test loader initialization with custom config."""
        loader = OcrPdfLoader(
            min_confidence=0.85,
            use_gpu=True,
            lang="fr",
            force_ocr=True,
        )

        assert loader.min_confidence == 0.85
        assert loader.use_gpu is True
        assert loader.lang == "fr"
        assert loader.force_ocr is True
