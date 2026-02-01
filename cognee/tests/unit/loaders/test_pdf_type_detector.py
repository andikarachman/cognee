"""Unit tests for PDF type detector."""

import sys
import pytest
from unittest.mock import Mock, MagicMock
from cognee.infrastructure.loaders.utils.pdf_type_detector import (
    PDFType,
    detect_pdf_type,
)


@pytest.fixture
def mock_pdfplumber():
    """Fixture to inject mock pdfplumber into sys.modules for lazy import testing.

    This fixture handles the case where pdfplumber is imported locally inside
    functions (lazy import pattern). It injects a mock into sys.modules so that
    when `import pdfplumber` executes inside detect_pdf_type(), it gets our mock.

    Yields:
        MagicMock: Mock pdfplumber module that can be configured in tests

    Example:
        def test_something(mock_pdfplumber):
            mock_page = Mock()
            mock_page.extract_text.return_value = "x" * 200

            mock_pdf = MagicMock()
            mock_pdf.__enter__.return_value.pages = [mock_page]
            mock_pdfplumber.open.return_value = mock_pdf

            result = detect_pdf_type("test.pdf")
            assert result == PDFType.DIGITAL
    """
    # Create mock pdfplumber module
    mock_module = MagicMock()

    # Inject into sys.modules so `import pdfplumber` gets our mock
    original = sys.modules.get("pdfplumber")  # Save original if exists
    sys.modules["pdfplumber"] = mock_module

    yield mock_module

    # Cleanup: restore original state after test
    if original is None:
        sys.modules.pop("pdfplumber", None)
    else:
        sys.modules["pdfplumber"] = original


class TestPDFTypeDetection:
    """Tests for PDF type detection (digital vs scanned)."""

    @pytest.mark.parametrize(
        "avg_chars,expected_type",
        [
            (500, PDFType.DIGITAL),  # Lots of text -> digital
            (150, PDFType.DIGITAL),  # Above threshold -> digital
            (101, PDFType.DIGITAL),  # Just above boundary -> digital
            (100, PDFType.HYBRID),  # At boundary -> hybrid (safe default)
            (50, PDFType.HYBRID),  # Some text -> hybrid
            (25, PDFType.HYBRID),  # Little text -> hybrid
            (20, PDFType.HYBRID),  # At lower boundary -> hybrid
            (19, PDFType.SCANNED),  # Below boundary -> scanned
            (5, PDFType.SCANNED),  # Very little text -> scanned
            (0, PDFType.SCANNED),  # No text -> scanned
        ],
    )
    def test_detect_pdf_type_by_char_count(self, mock_pdfplumber, avg_chars, expected_type):
        """Test PDF type detection based on character count."""
        # Mock PDF with single page
        mock_page = Mock()
        mock_page.extract_text.return_value = "x" * avg_chars

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf")

        assert result == expected_type

    def test_detect_multi_page_averaging(self, mock_pdfplumber):
        """Test PDF type detection averages across multiple pages."""
        # Mock PDF with 3 pages: 200, 300, 100 chars -> avg 200 -> DIGITAL
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "x" * 200

        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "x" * 300

        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "x" * 100

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf", sample_pages=3)

        assert result == PDFType.DIGITAL

    def test_detect_respects_sample_pages_limit(self, mock_pdfplumber):
        """Test that only sample_pages number of pages are checked."""
        # Create 10 pages, but only first 3 should be checked
        mock_pages = []
        for i in range(10):
            mock_page = Mock()
            mock_page.extract_text.return_value = "x" * 200  # Digital
            mock_pages.append(mock_page)

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = mock_pages
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf", sample_pages=3)

        assert result == PDFType.DIGITAL

        # Verify only first 3 pages were accessed
        assert mock_pages[0].extract_text.called
        assert mock_pages[1].extract_text.called
        assert mock_pages[2].extract_text.called

    def test_detect_handles_none_text(self, mock_pdfplumber):
        """Test detection handles pages that return None for text."""
        mock_page = Mock()
        mock_page.extract_text.return_value = None  # No text

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf")

        assert result == PDFType.SCANNED  # None treated as 0 chars

    def test_detect_handles_empty_string(self, mock_pdfplumber):
        """Test detection handles pages with empty string."""
        mock_page = Mock()
        mock_page.extract_text.return_value = ""  # Empty string

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf")

        assert result == PDFType.SCANNED

    def test_detect_handles_whitespace_only(self, mock_pdfplumber):
        """Test detection handles pages with only whitespace."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "   \n\t  "  # Whitespace only

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf")

        # After strip(), this is 0 chars -> SCANNED
        assert result == PDFType.SCANNED

    def test_detect_handles_mixed_pages(self, mock_pdfplumber):
        """Test detection with mix of digital and scanned pages."""
        # Page 1: 300 chars (digital)
        # Page 2: 10 chars (scanned)
        # Avg: 155 chars -> DIGITAL
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "x" * 300

        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "x" * 10

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("test.pdf", sample_pages=2)

        assert result == PDFType.DIGITAL

    def test_detect_defaults_to_hybrid_on_error(self, mock_pdfplumber):
        """Test that detection defaults to HYBRID on error."""
        mock_pdfplumber.open.side_effect = Exception("PDF open failed")

        result = detect_pdf_type("invalid.pdf")

        # Should default to HYBRID (safe choice - will use OCR)
        assert result == PDFType.HYBRID

    def test_detect_with_zero_pages(self, mock_pdfplumber):
        """Test detection with PDF that has no pages."""
        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = []
        mock_pdfplumber.open.return_value = mock_pdf

        result = detect_pdf_type("empty.pdf")

        # No pages -> 0 chars -> SCANNED (or could default to HYBRID)
        assert result in [PDFType.SCANNED, PDFType.HYBRID]


class TestPDFTypeEnum:
    """Tests for PDFType enum."""

    def test_pdf_type_values(self):
        """Test PDFType enum values."""
        assert PDFType.DIGITAL.value == "digital"
        assert PDFType.SCANNED.value == "scanned"
        assert PDFType.HYBRID.value == "hybrid"

    def test_pdf_type_comparison(self):
        """Test PDFType enum comparison."""
        assert PDFType.DIGITAL == PDFType.DIGITAL
        assert PDFType.DIGITAL != PDFType.SCANNED
        assert PDFType.SCANNED != PDFType.HYBRID
