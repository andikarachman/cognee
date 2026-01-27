"""Unit tests for PdfPlumberLoader (digital PDF loader)."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from cognee.infrastructure.loaders.external.pdfplumber_loader import PdfPlumberLoader


class TestBoundingBoxNormalization:
    """Tests for bbox normalization in PdfPlumberLoader."""

    def test_normalize_bbox_method(self):
        """Test _normalize_bbox method."""
        loader = PdfPlumberLoader()

        # Test with page dimensions 1000x1400
        bbox = loader._normalize_bbox(
            x0=100,
            y0=200,
            x1=800,
            y1=900,
            page_width=1000,
            page_height=1400,
        )

        assert abs(bbox.x_min - 0.1) < 0.01  # 100/1000
        assert abs(bbox.y_min - 0.143) < 0.01  # 200/1400
        assert abs(bbox.x_max - 0.8) < 0.01  # 800/1000
        assert abs(bbox.y_max - 0.643) < 0.01  # 900/1400


class TestLayoutTypeDetection:
    """Tests for layout type detection."""

    @pytest.mark.parametrize(
        "y_pos,page_height,expected_type",
        [
            (50, 1000, "header"),  # Top 10%
            (80, 1000, "header"),  # Top 10%
            (100, 1000, "text"),  # Just below header threshold
            (500, 1000, "text"),  # Middle
            (900, 1000, "footer"),  # Bottom 10%
            (920, 1000, "footer"),  # Bottom 10%
            (880, 1000, "text"),  # Just above footer threshold
        ],
    )
    def test_detect_layout_type_by_position(self, y_pos, page_height, expected_type):
        """Test layout type detection based on Y position."""
        loader = PdfPlumberLoader()

        layout_type = loader._detect_layout_type(
            x0=100,
            y0=y_pos,
            x1=800,
            y1=y_pos + 50,
            page_width=1000,
            page_height=page_height,
            is_in_table=False,
        )

        assert layout_type == expected_type

    def test_detect_table_layout(self):
        """Test that tables are detected correctly."""
        loader = PdfPlumberLoader()

        layout_type = loader._detect_layout_type(
            x0=100,
            y0=500,
            x1=800,
            y1=600,
            page_width=1000,
            page_height=1000,
            is_in_table=True,
        )

        assert layout_type == "table"


class TestWordExtraction:
    """Tests for word and line extraction from PDF."""

    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    def test_extract_words_basic(self, mock_pdfplumber):
        """Test basic word extraction with bbox."""
        # Mock pdfplumber page
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1400
        mock_page.extract_words.return_value = [
            {"text": "Hello", "x0": 100, "y0": 200, "x1": 200, "y1": 250},
            {"text": "World", "x0": 210, "y0": 200, "x1": 310, "y1": 250},
        ]
        mock_page.find_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        loader = PdfPlumberLoader()
        result = loader._extract_text_with_layout("test.pdf")

        assert "Hello" in result
        assert "World" in result
        assert "[page=1" in result
        assert "bbox=" in result
        assert "type=" in result

    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    def test_line_grouping(self, mock_pdfplumber):
        """Test that words with similar Y-coordinates are grouped into lines."""
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            # Line 1 (y=200-210)
            {"text": "First", "x0": 100, "y0": 200, "x1": 200, "y1": 210},
            {"text": "line", "x0": 210, "y0": 202, "x1": 280, "y1": 212},
            # Line 2 (y=300-310)
            {"text": "Second", "x0": 100, "y0": 300, "x1": 200, "y1": 310},
            {"text": "line", "x0": 210, "y0": 302, "x1": 280, "y1": 312},
        ]
        mock_page.find_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        loader = PdfPlumberLoader()
        result = loader._extract_text_with_layout("test.pdf")

        # Should have "First line" on one line
        assert "First line" in result or ("First" in result and "line" in result)

    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    def test_table_detection(self, mock_pdfplumber):
        """Test table detection and marking."""
        mock_table = Mock()
        mock_table.bbox = (100, 300, 800, 600)  # x0, y0, x1, y1
        mock_table.extract.return_value = [
            ["Header1", "Header2"],
            ["Cell1", "Cell2"],
        ]

        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            {"text": "Header1", "x0": 100, "y0": 300, "x1": 200, "y1": 320},
            {"text": "Cell1", "x0": 100, "y0": 350, "x1": 180, "y1": 370},
        ]
        mock_page.find_tables.return_value = [mock_table]

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        loader = PdfPlumberLoader()
        result = loader._extract_text_with_layout("test.pdf")

        # Should have table marker
        assert "type=table" in result

    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    def test_multi_page_extraction(self, mock_pdfplumber):
        """Test extraction from multiple pages."""
        mock_page1 = Mock()
        mock_page1.width = 1000
        mock_page1.height = 1000
        mock_page1.extract_words.return_value = [
            {"text": "Page1", "x0": 100, "y0": 200, "x1": 200, "y1": 220},
        ]
        mock_page1.find_tables.return_value = []

        mock_page2 = Mock()
        mock_page2.width = 1000
        mock_page2.height = 1000
        mock_page2.extract_words.return_value = [
            {"text": "Page2", "x0": 100, "y0": 200, "x1": 200, "y1": 220},
        ]
        mock_page2.find_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value = mock_pdf

        loader = PdfPlumberLoader()
        result = loader._extract_text_with_layout("test.pdf")

        assert "Page1" in result
        assert "Page2" in result
        assert "[page=1" in result
        assert "[page=2" in result


class TestLoaderIntegration:
    """Tests for full loader integration."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.get_file_metadata")
    async def test_load_method_success(
        self, mock_get_metadata, mock_get_storage, mock_pdfplumber
    ):
        """Test full load() method with storage."""
        # Mock PDF
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            {"text": "Test", "x0": 100, "y0": 200, "x1": 200, "y1": 220},
        ]
        mock_page.find_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        # Mock storage
        mock_storage = AsyncMock()
        mock_get_storage.return_value = mock_storage

        # Mock metadata
        mock_get_metadata.return_value = {"name": "test.pdf"}

        loader = PdfPlumberLoader()
        result = await loader.load("test.pdf")

        # Should return list with one item (the stored text file)
        assert isinstance(result, list)
        assert len(result) == 1

        # Verify storage was called
        mock_storage.store.assert_called_once()

        # Verify filename format
        call_args = mock_storage.store.call_args
        assert "layout_text_" in call_args[1]["filename"]
        assert call_args[1]["filename"].endswith(".txt")

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    async def test_load_handles_empty_pdf(self, mock_pdfplumber):
        """Test loading empty PDF."""
        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = []
        mock_pdfplumber.open.return_value = mock_pdf

        loader = PdfPlumberLoader()
        result = loader._extract_text_with_layout("empty.pdf")

        assert result == ""  # Empty string for empty PDF

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    async def test_load_handles_error(self, mock_pdfplumber):
        """Test error handling during load."""
        mock_pdfplumber.open.side_effect = Exception("PDF open failed")

        loader = PdfPlumberLoader()

        with pytest.raises(Exception):
            loader._extract_text_with_layout("invalid.pdf")

    def test_output_format(self):
        """Test that output has correct metadata format."""
        loader = PdfPlumberLoader()

        # Create a test line output
        line_output = loader._format_line_output(
            text="Sample text",
            bbox_tuple=(0.1, 0.2, 0.8, 0.3),
            layout_type="text",
            page_number=1,
        )

        assert "Sample text" in line_output
        assert "[page=1" in line_output
        assert "bbox=(0.1" in line_output
        assert "type=text" in line_output

    def test_can_handle_method(self):
        """Test can_handle method identifies PDF files."""
        loader = PdfPlumberLoader()

        assert loader.can_handle("test.pdf") is True
        assert loader.can_handle("test.PDF") is True
        assert loader.can_handle("test.txt") is False
        assert loader.can_handle("test.docx") is False

    def test_loader_name(self):
        """Test loader has correct name."""
        loader = PdfPlumberLoader()
        assert loader.loader_name == "pdfplumber_loader"
