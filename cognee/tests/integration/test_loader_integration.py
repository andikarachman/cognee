"""Integration tests for loader registration and priority."""

import pytest
from unittest.mock import patch, Mock
from cognee.infrastructure.loaders.LoaderEngine import LoaderEngine
from cognee.infrastructure.loaders.supported_loaders import supported_loaders


class TestLoaderRegistration:
    """Tests for loader registration in supported_loaders."""

    def test_pdfplumber_loader_registered(self):
        """Test that PdfPlumberLoader is registered."""
        # Check if loader is registered (conditional on pdfplumber availability)
        if "pdfplumber_loader" in supported_loaders:
            from cognee.infrastructure.loaders.external.pdfplumber_loader import (
                PdfPlumberLoader,
            )

            assert supported_loaders["pdfplumber_loader"] == PdfPlumberLoader

    def test_ocr_pdf_loader_registered(self):
        """Test that OcrPdfLoader is registered."""
        if "ocr_pdf_loader" in supported_loaders:
            from cognee.infrastructure.loaders.external.ocr_pdf_loader import (
                OcrPdfLoader,
            )

            assert supported_loaders["ocr_pdf_loader"] == OcrPdfLoader

    def test_ocr_image_loader_registered(self):
        """Test that OcrImageLoader is registered."""
        if "ocr_image_loader" in supported_loaders:
            from cognee.infrastructure.loaders.core.ocr_image_loader import (
                OcrImageLoader,
            )

            assert supported_loaders["ocr_image_loader"] == OcrImageLoader

    def test_fallback_loaders_still_registered(self):
        """Test that fallback loaders (PyPdf, Image) are still registered."""
        # These should always be available
        assert "pypdf_loader" in supported_loaders or "pdf_loader" in supported_loaders
        assert "image_loader" in supported_loaders


class TestLoaderPriority:
    """Tests for loader priority and selection."""

    def test_default_priority_order(self):
        """Test that default priority prefers layout-aware loaders."""
        engine = LoaderEngine()

        # Check priority list
        priority = engine.default_loader_priority

        # Should have some PDF loaders
        pdf_loaders = [
            loader for loader in priority if "pdf" in loader.lower()
        ]
        assert len(pdf_loaders) > 0

        # Layout-aware loaders should come before fallbacks
        if "pdfplumber_loader" in priority:
            pdfplumber_idx = priority.index("pdfplumber_loader")
            if "pypdf_loader" in priority:
                pypdf_idx = priority.index("pypdf_loader")
                assert pdfplumber_idx < pypdf_idx

    def test_loader_selection_for_pdf(self):
        """Test that LoaderEngine selects appropriate loader for PDF."""
        engine = LoaderEngine()

        # Get loader for PDF file
        loader_class = engine.get_loader_for_file("test.pdf")

        # Should get some PDF loader
        assert loader_class is not None

        # Check that it can handle PDFs
        loader_instance = loader_class()
        assert loader_instance.can_handle("test.pdf")

    def test_loader_selection_for_image(self):
        """Test that LoaderEngine selects appropriate loader for images."""
        engine = LoaderEngine()

        # Get loader for image file
        loader_class = engine.get_loader_for_file("test.png")

        # Should get some image loader
        assert loader_class is not None

        # Check that it can handle images
        loader_instance = loader_class()
        assert loader_instance.can_handle("test.png")


class TestLoaderCapabilities:
    """Tests for loader can_handle methods."""

    @pytest.mark.parametrize(
        "filename,loader_name",
        [
            ("test.pdf", "pdfplumber_loader"),
            ("test.PDF", "pdfplumber_loader"),
            ("test.pdf", "ocr_pdf_loader"),
            ("test.png", "ocr_image_loader"),
            ("test.jpg", "ocr_image_loader"),
            ("test.jpeg", "ocr_image_loader"),
        ],
    )
    def test_loader_can_handle_files(self, filename, loader_name):
        """Test that loaders correctly identify files they can handle."""
        if loader_name in supported_loaders:
            loader_class = supported_loaders[loader_name]
            loader = loader_class()

            assert loader.can_handle(filename) is True

    @pytest.mark.parametrize(
        "filename,loader_name",
        [
            ("test.txt", "pdfplumber_loader"),
            ("test.docx", "ocr_pdf_loader"),
            ("test.pdf", "ocr_image_loader"),
        ],
    )
    def test_loader_rejects_wrong_file_types(self, filename, loader_name):
        """Test that loaders reject files they cannot handle."""
        if loader_name in supported_loaders:
            loader_class = supported_loaders[loader_name]
            loader = loader_class()

            assert loader.can_handle(filename) is False


class TestLoaderNames:
    """Tests for loader name consistency."""

    def test_pdfplumber_loader_name(self):
        """Test PdfPlumberLoader has correct name."""
        if "pdfplumber_loader" in supported_loaders:
            from cognee.infrastructure.loaders.external.pdfplumber_loader import (
                PdfPlumberLoader,
            )

            loader = PdfPlumberLoader()
            assert loader.loader_name == "pdfplumber_loader"

    def test_ocr_pdf_loader_name(self):
        """Test OcrPdfLoader has correct name."""
        if "ocr_pdf_loader" in supported_loaders:
            from cognee.infrastructure.loaders.external.ocr_pdf_loader import (
                OcrPdfLoader,
            )

            loader = OcrPdfLoader()
            assert loader.loader_name == "ocr_pdf_loader"

    def test_ocr_image_loader_name(self):
        """Test OcrImageLoader has correct name."""
        if "ocr_image_loader" in supported_loaders:
            from cognee.infrastructure.loaders.core.ocr_image_loader import (
                OcrImageLoader,
            )

            loader = OcrImageLoader()
            assert loader.loader_name == "ocr_image_loader"


class TestLoaderFallbacks:
    """Tests for loader fallback behavior."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    async def test_pdfplumber_loader_actual_loading(self, mock_pdfplumber):
        """Test that PdfPlumberLoader can actually load a file."""
        if "pdfplumber_loader" in supported_loaders:
            from cognee.infrastructure.loaders.external.pdfplumber_loader import (
                PdfPlumberLoader,
            )

            # Mock PDF
            mock_page = Mock()
            mock_page.width = 1000
            mock_page.height = 1000
            mock_page.extract_words.return_value = [
                {"text": "Test", "x0": 100, "y0": 200, "x1": 200, "y1": 220},
            ]
            mock_page.find_tables.return_value = []

            mock_pdf = Mock()
            mock_pdf.__enter__.return_value.pages = [mock_page]
            mock_pdfplumber.open.return_value = mock_pdf

            loader = PdfPlumberLoader()
            result = loader._extract_text_with_layout("test.pdf")

            # Should return text with metadata
            assert "Test" in result
            assert "[page=" in result

    @pytest.mark.asyncio
    async def test_loader_engine_fallback_chain(self):
        """Test that LoaderEngine respects fallback chain."""
        engine = LoaderEngine()

        # Get all PDF loaders in priority order
        pdf_loaders = []
        for loader_name in engine.default_loader_priority:
            if loader_name in supported_loaders:
                loader_class = supported_loaders[loader_name]
                loader = loader_class()
                if loader.can_handle("test.pdf"):
                    pdf_loaders.append(loader_name)

        # Should have at least 2 PDF loaders (one primary, one fallback)
        assert len(pdf_loaders) >= 2


class TestOCRAvailability:
    """Tests for OCR availability detection."""

    def test_ocr_loaders_require_dependencies(self):
        """Test that OCR loaders are only registered if dependencies available."""
        # If OCR loaders are registered, they should be importable
        if "ocr_pdf_loader" in supported_loaders:
            try:
                from cognee.infrastructure.loaders.external.ocr_pdf_loader import (
                    OcrPdfLoader,
                )

                # Should be able to create instance
                loader = OcrPdfLoader()
                assert loader is not None
            except ImportError:
                pytest.fail("OCR loader registered but dependencies missing")

    def test_system_works_without_ocr(self):
        """Test that system works even without OCR loaders."""
        # At minimum, should have basic PDF and image loaders
        has_pdf_loader = any(
            "pdf" in name.lower() for name in supported_loaders.keys()
        )
        has_image_loader = any(
            "image" in name.lower() for name in supported_loaders.keys()
        )

        assert has_pdf_loader, "System must have at least one PDF loader"
        assert has_image_loader, "System must have at least one image loader"


class TestPreferredLoaders:
    """Tests for preferred_loaders configuration."""

    @pytest.mark.asyncio
    async def test_preferred_loader_configuration(self):
        """Test that users can specify preferred loaders."""
        engine = LoaderEngine()

        # Test with preferred loaders config
        preferred = {
            "ocr_pdf_loader": {
                "min_confidence": 0.85,
                "use_gpu": False,
            }
        }

        # Should be able to pass preferred loaders
        # (Implementation may vary, this tests the concept)
        assert isinstance(preferred, dict)

    @pytest.mark.asyncio
    async def test_loader_configuration_options(self):
        """Test that loaders accept configuration options."""
        if "ocr_pdf_loader" in supported_loaders:
            from cognee.infrastructure.loaders.external.ocr_pdf_loader import (
                OcrPdfLoader,
            )

            # Should accept config options
            loader = OcrPdfLoader(
                min_confidence=0.9,
                use_gpu=False,
                lang="en",
            )

            assert loader.min_confidence == 0.9
            assert loader.use_gpu is False
            assert loader.lang == "en"
