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
            bbox=(100, 200, 800, 900),
            page_width=1000,
            page_height=1400,
        )

        # Returns tuple (x_min, y_min, x_max, y_max), not object
        assert abs(bbox[0] - 0.1) < 0.01  # x_min: 100/1000
        assert abs(bbox[1] - 0.143) < 0.01  # y_min: 200/1400
        assert abs(bbox[2] - 0.8) < 0.01  # x_max: 800/1000
        assert abs(bbox[3] - 0.643) < 0.01  # y_max: 900/1400


class TestLayoutTypeDetection:
    """Tests for layout type detection."""

    @pytest.mark.parametrize(
        "y_pos,page_height,expected_type",
        [
            (30, 1000, "header"),  # y1=80 < 100 (top 10%)
            (40, 1000, "header"),  # y1=90 < 100 (top 10%)
            (100, 1000, "text"),  # y1=150 >= 100 (below header threshold)
            (500, 1000, "text"),  # Middle
            (860, 1000, "text"),  # y0=860 <= 900 (above footer threshold)
            (920, 1000, "footer"),  # y0=920 > 900 (bottom 10%)
            (950, 1000, "footer"),  # y0=950 > 900 (bottom 10%)
        ],
    )
    def test_detect_layout_type_by_position(self, y_pos, page_height, expected_type):
        """Test layout type detection based on Y position."""
        loader = PdfPlumberLoader()

        layout_type = loader._detect_layout_type(
            bbox=(100, y_pos, 800, y_pos + 50),
            page_height=page_height,
            is_in_table=False,
        )

        assert layout_type == expected_type

    def test_detect_table_layout(self):
        """Test that tables are detected correctly."""
        loader = PdfPlumberLoader()

        layout_type = loader._detect_layout_type(
            bbox=(100, 500, 800, 600),
            page_height=1000,
            is_in_table=True,
        )

        assert layout_type == "table"


class TestWordExtraction:
    """Tests for word and line extraction from PDF."""

    def test_extract_words_basic(self):
        """Test basic word extraction with bbox."""
        # Mock pdfplumber page
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1400
        mock_page.extract_words.return_value = [
            {"text": "Hello", "x0": 100, "top": 200, "x1": 200, "bottom": 250},
            {"text": "World", "x0": 210, "top": 200, "x1": 310, "bottom": 250},
        ]
        mock_page.find_tables.return_value = []

        loader = PdfPlumberLoader()
        result = loader._extract_words_with_layout(mock_page, page_num=1)

        # result is List[str], join to check content
        result_text = "\n".join(result)
        assert "Hello" in result_text
        assert "World" in result_text
        assert "[page=1" in result_text
        assert "bbox=" in result_text
        assert "type=" in result_text

    def test_line_grouping(self):
        """Test that words with similar Y-coordinates are grouped into lines."""
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            # Line 1 (y=200-210)
            {"text": "First", "x0": 100, "top": 200, "x1": 200, "bottom": 210},
            {"text": "line", "x0": 210, "top": 202, "x1": 280, "bottom": 212},
            # Line 2 (y=300-310)
            {"text": "Second", "x0": 100, "top": 300, "x1": 200, "bottom": 310},
            {"text": "line", "x0": 210, "top": 302, "x1": 280, "bottom": 312},
        ]
        mock_page.find_tables.return_value = []

        loader = PdfPlumberLoader()
        result = loader._extract_words_with_layout(mock_page, page_num=1)

        # Should have two lines
        assert len(result) == 2
        result_text = "\n".join(result)
        # Should have "First line" on one line
        assert "First line" in result_text

    def test_table_detection(self):
        """Test table detection and marking."""
        mock_table = Mock()
        mock_table.bbox = (100, 300, 800, 600)  # x0, y0, x1, y1

        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            {"text": "Header1", "x0": 100, "top": 300, "x1": 200, "bottom": 320},
            {"text": "Cell1", "x0": 100, "top": 350, "x1": 180, "bottom": 370},
        ]
        mock_page.find_tables.return_value = [mock_table]

        loader = PdfPlumberLoader()
        result = loader._extract_words_with_layout(mock_page, page_num=1)

        result_text = "\n".join(result)
        # Should have table marker
        assert "type=table" in result_text

    def test_multi_page_extraction(self):
        """Test extraction from multiple pages."""
        mock_page1 = Mock()
        mock_page1.width = 1000
        mock_page1.height = 1000
        mock_page1.extract_words.return_value = [
            {"text": "Page1", "x0": 100, "top": 200, "x1": 200, "bottom": 220},
        ]
        mock_page1.find_tables.return_value = []

        mock_page2 = Mock()
        mock_page2.width = 1000
        mock_page2.height = 1000
        mock_page2.extract_words.return_value = [
            {"text": "Page2", "x0": 100, "top": 200, "x1": 200, "bottom": 220},
        ]
        mock_page2.find_tables.return_value = []

        loader = PdfPlumberLoader()
        result1 = loader._extract_words_with_layout(mock_page1, page_num=1)
        result2 = loader._extract_words_with_layout(mock_page2, page_num=2)

        result_text1 = "\n".join(result1)
        result_text2 = "\n".join(result2)

        assert "Page1" in result_text1
        assert "Page2" in result_text2
        assert "[page=1" in result_text1
        assert "[page=2" in result_text2


class TestLoaderIntegration:
    """Tests for full loader integration."""

    @pytest.mark.asyncio
    @patch("builtins.open", create=True)
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.get_file_storage")
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.get_file_metadata")
    async def test_load_method_success(
        self, mock_get_metadata, mock_get_storage, mock_open
    ):
        """Test full load() method with storage."""
        import sys

        # Mock file object
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_open.return_value = mock_file

        # Mock PDF
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            {"text": "Test", "x0": 100, "top": 200, "x1": 200, "bottom": 220},
        ]
        mock_page.find_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdf.__exit__.return_value = None

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.store.return_value = "/path/to/layout_text_hash.txt"
        mock_get_storage.return_value = mock_storage

        # Mock metadata
        mock_get_metadata.return_value = {"name": "test.pdf", "content_hash": "hash123"}

        # Inject mock pdfplumber into sys.modules
        original_pdfplumber = sys.modules.get('pdfplumber')
        sys.modules['pdfplumber'] = mock_pdfplumber

        try:
            loader = PdfPlumberLoader()
            result = await loader.load("test.pdf")

            # Should return the stored file path
            assert result == "/path/to/layout_text_hash.txt"

            # Verify storage was called
            mock_storage.store.assert_called_once()

            # Verify filename format - storage.store() called with positional args
            call_args = mock_storage.store.call_args
            filename = call_args[0][0]  # First positional arg
            assert "layout_text_" in filename
            assert filename.endswith(".txt")
        finally:
            # Restore original pdfplumber module
            if original_pdfplumber is not None:
                sys.modules['pdfplumber'] = original_pdfplumber
            else:
                sys.modules.pop('pdfplumber', None)

    @pytest.mark.asyncio
    async def test_load_handles_empty_pdf(self):
        """Test loading empty page."""
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = []
        mock_page.find_tables.return_value = []

        loader = PdfPlumberLoader()
        result = loader._extract_words_with_layout(mock_page, page_num=1)

        assert result == []  # Empty list for empty page

    @pytest.mark.asyncio
    async def test_load_handles_error(self):
        """Test error handling during extraction."""
        mock_page = Mock()
        mock_page.extract_words.side_effect = Exception("Extract failed")

        loader = PdfPlumberLoader()

        with pytest.raises(Exception):
            loader._extract_words_with_layout(mock_page, page_num=1)

    def test_output_format(self):
        """Test that output has correct metadata format."""
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1000
        mock_page.extract_words.return_value = [
            {"text": "Sample", "x0": 100, "top": 200, "x1": 200, "bottom": 220},
            {"text": "text", "x0": 210, "top": 200, "x1": 280, "bottom": 220},
        ]
        mock_page.find_tables.return_value = []

        loader = PdfPlumberLoader()
        result = loader._extract_words_with_layout(mock_page, page_num=1)

        # Should have one line with both words
        assert len(result) == 1
        line_output = result[0]

        assert "Sample text" in line_output
        assert "[page=1" in line_output
        assert "bbox=" in line_output
        assert "type=" in line_output

    def test_can_handle_method(self):
        """Test can_handle method identifies PDF files."""
        loader = PdfPlumberLoader()

        # PDF files (extension should be lowercase)
        assert loader.can_handle("pdf", "application/pdf") is True

        # Non-PDF files
        assert loader.can_handle("txt", "text/plain") is False
        assert loader.can_handle("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") is False

        # Wrong mime type
        assert loader.can_handle("pdf", "text/plain") is False

    def test_loader_name(self):
        """Test loader has correct name."""
        loader = PdfPlumberLoader()
        assert loader.loader_name == "pdfplumber_loader"
