"""Unit tests for LayoutChunk and related models."""

import pytest
from pydantic import ValidationError
from cognee.shared.data_models import BoundingBox
from cognee.modules.chunking.models.LayoutChunk import (
    PageDimensions,
    LayoutType,
    LayoutChunk,
)
from cognee.modules.chunking.models import DocumentChunk
from cognee.tests.test_data.mock_data.mock_ocr_results import (
    create_mock_chunk_bbox,
    create_mock_page_dimensions,
)


class TestBoundingBoxModel:
    """Tests for BoundingBox model validation and properties."""

    def test_valid_bbox_creation(self):
        """Test creating a valid BoundingBox with normalized coordinates."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
            confidence=0.95,
        )

        assert bbox.x_min == 0.1
        assert bbox.y_min == 0.2
        assert bbox.x_max == 0.8
        assert bbox.y_max == 0.9
        assert bbox.confidence == 0.95

    def test_bbox_with_pixel_coords(self):
        """Test BoundingBox with both normalized and pixel coordinates."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
            pixel_x_min=100,
            pixel_y_min=200,
            pixel_x_max=800,
            pixel_y_max=900,
            confidence=1.0,
        )

        assert bbox.pixel_x_min == 100
        assert bbox.pixel_y_min == 200
        assert bbox.pixel_x_max == 800
        assert bbox.pixel_y_max == 900

    @pytest.mark.parametrize(
        "x_min,y_min,x_max,y_max",
        [
            (-0.1, 0.2, 0.8, 0.9),  # Negative x_min
            (0.1, -0.2, 0.8, 0.9),  # Negative y_min
            (1.1, 0.2, 0.8, 0.9),  # x_min > 1
            (0.1, 0.2, 1.1, 0.9),  # x_max > 1
            (0.8, 0.2, 0.1, 0.9),  # Inverted x (x_min > x_max)
            (0.1, 0.9, 0.8, 0.2),  # Inverted y (y_min > y_max)
        ],
    )
    def test_invalid_bbox_coords(self, x_min, y_min, x_max, y_max):
        """Test that invalid bbox coordinates raise ValidationError."""
        with pytest.raises((ValidationError, ValueError, AssertionError)):
            BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

    def test_bbox_area_property(self):
        """Test bbox area calculation."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.6,
            y_max=0.7,
        )

        expected_area = (0.6 - 0.1) * (0.7 - 0.2)  # 0.5 * 0.5 = 0.25
        assert abs(bbox.area - expected_area) < 0.001

    def test_bbox_center_property(self):
        """Test bbox center calculation."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.9,
            y_max=0.8,
        )

        center_x, center_y = bbox.center
        assert abs(center_x - 0.5) < 0.001  # (0.1 + 0.9) / 2
        assert abs(center_y - 0.5) < 0.001  # (0.2 + 0.8) / 2

    def test_bbox_default_confidence(self):
        """Test that bbox confidence defaults to 1.0."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
        )

        assert bbox.confidence == 1.0


class TestPageDimensionsModel:
    """Tests for PageDimensions model."""

    def test_page_dimensions_creation(self):
        """Test creating PageDimensions."""
        dims = PageDimensions(width=1000, height=1400, dpi=150)

        assert dims.width == 1000
        assert dims.height == 1400
        assert dims.dpi == 150

    def test_page_dimensions_optional_dpi(self):
        """Test PageDimensions with optional dpi field."""
        dims = PageDimensions(width=800, height=1200)

        assert dims.width == 800
        assert dims.height == 1200
        assert dims.dpi is None


class TestLayoutTypeEnum:
    """Tests for LayoutType enum."""

    def test_layout_type_values(self):
        """Test all LayoutType enum values."""
        assert LayoutType.TEXT == "text"
        assert LayoutType.TITLE == "title"
        assert LayoutType.HEADING == "heading"
        assert LayoutType.PARAGRAPH == "paragraph"
        assert LayoutType.TABLE == "table"
        assert LayoutType.FIGURE == "figure"
        assert LayoutType.CAPTION == "caption"
        assert LayoutType.HEADER == "header"
        assert LayoutType.FOOTER == "footer"
        assert LayoutType.UNKNOWN == "unknown"


class TestLayoutChunkModel:
    """Tests for LayoutChunk model."""

    def test_layout_chunk_creation_minimal(self):
        """Test creating LayoutChunk with minimal fields (backward compatible)."""
        chunk = LayoutChunk(
            text="Sample text content",
            word_count=3,
            chunk_id="test-chunk-1",
        )

        assert chunk.text == "Sample text content"
        assert chunk.word_count == 3
        assert chunk.chunk_id == "test-chunk-1"
        assert chunk.bounding_boxes == []
        assert chunk.page_number is None
        assert chunk.layout_type == LayoutType.TEXT

    def test_layout_chunk_creation_full(self):
        """Test creating LayoutChunk with all layout fields."""
        bbox1 = create_mock_chunk_bbox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.2)
        bbox2 = create_mock_chunk_bbox(x_min=0.1, y_min=0.3, x_max=0.5, y_max=0.4)
        dims = create_mock_page_dimensions()

        chunk = LayoutChunk(
            text="Sample text with layout",
            word_count=4,
            chunk_id="test-chunk-2",
            bounding_boxes=[bbox1, bbox2],
            page_number=1,
            page_dimensions=dims,
            layout_type=LayoutType.PARAGRAPH,
            ocr_confidence=0.95,
            reading_order=1,
            column_index=0,
        )

        assert len(chunk.bounding_boxes) == 2
        assert chunk.page_number == 1
        assert chunk.layout_type == LayoutType.PARAGRAPH
        assert chunk.ocr_confidence == 0.95
        assert chunk.reading_order == 1
        assert chunk.column_index == 0

    def test_primary_bbox_property(self):
        """Test primary_bbox returns largest bbox by area."""
        bbox_small = create_mock_chunk_bbox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.2)  # 0.2 * 0.1 = 0.02
        bbox_large = create_mock_chunk_bbox(x_min=0.1, y_min=0.3, x_max=0.9, y_max=0.8)  # 0.8 * 0.5 = 0.4

        chunk = LayoutChunk(
            text="Test",
            word_count=1,
            bounding_boxes=[bbox_small, bbox_large],
        )

        primary = chunk.primary_bbox
        assert primary == bbox_large
        assert primary.area > bbox_small.area

    def test_primary_bbox_empty(self):
        """Test primary_bbox returns None when no bboxes."""
        chunk = LayoutChunk(
            text="Test",
            word_count=1,
            bounding_boxes=[],
        )

        assert chunk.primary_bbox is None

    def test_has_layout_info_property(self):
        """Test has_layout_info property."""
        # With bbox
        chunk_with_bbox = LayoutChunk(
            text="Test",
            word_count=1,
            bounding_boxes=[create_mock_chunk_bbox()],
        )
        assert chunk_with_bbox.has_layout_info is True

        # With page_number
        chunk_with_page = LayoutChunk(
            text="Test",
            word_count=1,
            page_number=1,
        )
        assert chunk_with_page.has_layout_info is True

        # Without layout info
        chunk_plain = LayoutChunk(
            text="Test",
            word_count=1,
        )
        assert chunk_plain.has_layout_info is False

    def test_get_page_numbers_method(self):
        """Test get_page_numbers returns sorted unique list."""
        chunk = LayoutChunk(
            text="Multi-page chunk",
            word_count=2,
            page_number=1,
            page_numbers=[3, 1, 2, 2],  # Unsorted with duplicates
        )

        page_nums = chunk.get_page_numbers()
        assert page_nums == [1, 2, 3]  # Sorted and unique

    def test_get_page_numbers_single_page(self):
        """Test get_page_numbers with only page_number field."""
        chunk = LayoutChunk(
            text="Single page",
            word_count=2,
            page_number=5,
        )

        page_nums = chunk.get_page_numbers()
        assert page_nums == [5]

    def test_get_page_numbers_empty(self):
        """Test get_page_numbers returns empty list when no pages."""
        chunk = LayoutChunk(
            text="No pages",
            word_count=2,
        )

        page_nums = chunk.get_page_numbers()
        assert page_nums == []

    def test_backward_compatibility_with_document_chunk(self):
        """Test LayoutChunk is compatible with DocumentChunk."""
        chunk = LayoutChunk(
            text="Test text",
            word_count=2,
            chunk_id="test-id",
        )

        # Should be instance of both
        assert isinstance(chunk, LayoutChunk)
        assert isinstance(chunk, DocumentChunk)

    def test_multi_page_chunk(self):
        """Test chunk spanning multiple pages."""
        chunk = LayoutChunk(
            text="Content spanning pages",
            word_count=3,
            page_numbers=[1, 2, 3],
            bounding_boxes=[
                create_mock_chunk_bbox(x_min=0.1, y_min=0.9, x_max=0.9, y_max=1.0),  # Bottom of page 1
                create_mock_chunk_bbox(x_min=0.1, y_min=0.0, x_max=0.9, y_max=0.1),  # Top of page 2
            ],
        )

        assert len(chunk.page_numbers) == 3
        assert len(chunk.bounding_boxes) == 2

    def test_layout_chunk_json_serialization(self):
        """Test LayoutChunk can be serialized to JSON."""
        bbox = create_mock_chunk_bbox()
        chunk = LayoutChunk(
            text="Test",
            word_count=1,
            bounding_boxes=[bbox],
            page_number=1,
            layout_type=LayoutType.TEXT,
        )

        # Should serialize without errors
        json_data = chunk.model_dump()
        assert json_data["text"] == "Test"
        assert json_data["page_number"] == 1
        assert json_data["layout_type"] == "text"
        assert len(json_data["bounding_boxes"]) == 1
