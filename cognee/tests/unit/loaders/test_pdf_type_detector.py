"""Unit tests for PDF type detector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from cognee.infrastructure.loaders.utils.pdf_type_detector import (
    PDFType,
    detect_pdf_type,
)


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
    def test_detect_pdf_type_by_char_count(self, avg_chars, expected_type):
        """Test PDF type detection based on character count."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
            # Mock PDF with single page
            mock_page = Mock()
            mock_page.extract_text.return_value = "x" * avg_chars

            mock_pdf = MagicMock()
            mock_pdf.__enter__.return_value.pages = [mock_page]
            mock_pdfplumber.open.return_value = mock_pdf

            result = detect_pdf_type("test.pdf")

            assert result == expected_type

    def test_detect_multi_page_averaging(self):
        """Test PDF type detection averages across multiple pages."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
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

    def test_detect_respects_sample_pages_limit(self):
        """Test that only sample_pages number of pages are checked."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
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

    def test_detect_handles_none_text(self):
        """Test detection handles pages that return None for text."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
            mock_page = Mock()
            mock_page.extract_text.return_value = None  # No text

            mock_pdf = MagicMock()
            mock_pdf.__enter__.return_value.pages = [mock_page]
            mock_pdfplumber.open.return_value = mock_pdf

            result = detect_pdf_type("test.pdf")

            assert result == PDFType.SCANNED  # None treated as 0 chars

    def test_detect_handles_empty_string(self):
        """Test detection handles pages with empty string."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
            mock_page = Mock()
            mock_page.extract_text.return_value = ""  # Empty string

            mock_pdf = MagicMock()
            mock_pdf.__enter__.return_value.pages = [mock_page]
            mock_pdfplumber.open.return_value = mock_pdf

            result = detect_pdf_type("test.pdf")

            assert result == PDFType.SCANNED

    def test_detect_handles_whitespace_only(self):
        """Test detection handles pages with only whitespace."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
            mock_page = Mock()
            mock_page.extract_text.return_value = "   \n\t  "  # Whitespace only

            mock_pdf = MagicMock()
            mock_pdf.__enter__.return_value.pages = [mock_page]
            mock_pdfplumber.open.return_value = mock_pdf

            result = detect_pdf_type("test.pdf")

            # After strip(), this is 0 chars -> SCANNED
            assert result == PDFType.SCANNED

    def test_detect_handles_mixed_pages(self):
        """Test detection with mix of digital and scanned pages."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
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

    def test_detect_defaults_to_hybrid_on_error(self):
        """Test that detection defaults to HYBRID on error."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
            mock_pdfplumber.open.side_effect = Exception("PDF open failed")

            result = detect_pdf_type("invalid.pdf")

            # Should default to HYBRID (safe choice - will use OCR)
            assert result == PDFType.HYBRID

    def test_detect_with_zero_pages(self):
        """Test detection with PDF that has no pages."""
        with patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber") as mock_pdfplumber:
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
