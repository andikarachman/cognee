"""Unit tests for SearchResult models with layout metadata."""

import pytest
from uuid import UUID, uuid4
from cognee.modules.search.types.SearchResult import (
    BoundingBox,
    LayoutMetadata,
    ChunkMetadata,
    SearchResult,
    DocumentReference,
)


class TestBoundingBoxModel:
    """Tests for BoundingBox model in SearchResult."""

    def test_bbox_creation(self):
        """Test creating a BoundingBox."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)

        assert bbox.x_min == 0.1
        assert bbox.y_min == 0.2
        assert bbox.x_max == 0.8
        assert bbox.y_max == 0.9

    def test_bbox_json_serialization(self):
        """Test BoundingBox JSON serialization."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        json_data = bbox.model_dump()

        assert json_data == {
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.8,
            "y_max": 0.9,
            "confidence": None,
        }


class TestLayoutMetadataModel:
    """Tests for LayoutMetadata model."""

    def test_layout_metadata_full(self):
        """Test creating LayoutMetadata with all fields."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        layout = LayoutMetadata(
            page_number=1,
            bbox=bbox,
            layout_type="text",
            confidence=0.95,
            page_width=1000,
            page_height=1400,
        )

        assert layout.page_number == 1
        assert layout.bbox == bbox
        assert layout.layout_type == "text"
        assert layout.confidence == 0.95
        assert layout.page_width == 1000
        assert layout.page_height == 1400

    def test_layout_metadata_optional_fields(self):
        """Test LayoutMetadata with optional fields as None."""
        layout = LayoutMetadata()

        assert layout.page_number is None
        assert layout.bbox is None
        assert layout.layout_type is None
        assert layout.confidence is None
        assert layout.page_width is None
        assert layout.page_height is None

    def test_layout_metadata_partial(self):
        """Test LayoutMetadata with only some fields."""
        layout = LayoutMetadata(
            page_number=2,
            layout_type="table",
        )

        assert layout.page_number == 2
        assert layout.layout_type == "table"
        assert layout.bbox is None
        assert layout.confidence is None

    def test_layout_metadata_json_serialization(self):
        """Test LayoutMetadata JSON serialization."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        layout = LayoutMetadata(
            page_number=1,
            bbox=bbox,
            layout_type="text",
            confidence=0.95,
        )

        json_data = layout.model_dump()
        assert json_data["page_number"] == 1
        assert json_data["bbox"] == {
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.8,
            "y_max": 0.9,
            "confidence": None,
        }
        assert json_data["layout_type"] == "text"
        assert json_data["confidence"] == 0.95


class TestChunkMetadataModel:
    """Tests for ChunkMetadata model."""

    def test_chunk_metadata_minimal(self):
        """Test ChunkMetadata with minimal required fields."""
        chunk = ChunkMetadata(text="Sample chunk text")

        assert chunk.text == "Sample chunk text"
        assert chunk.chunk_id is None
        assert chunk.chunk_index is None
        assert chunk.score is None
        assert chunk.layout is None

    def test_chunk_metadata_full(self):
        """Test ChunkMetadata with all fields."""
        chunk_id = uuid4()
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        layout = LayoutMetadata(
            page_number=1,
            bbox=bbox,
            layout_type="text",
            confidence=0.95,
        )

        chunk = ChunkMetadata(
            text="Full chunk metadata",
            chunk_id=chunk_id,
            chunk_index=0,
            score=0.87,
            layout=layout,
        )

        assert chunk.text == "Full chunk metadata"
        assert chunk.chunk_id == chunk_id
        assert chunk.chunk_index == 0
        assert chunk.score == 0.87
        assert chunk.layout == layout
        assert chunk.layout.page_number == 1

    def test_chunk_metadata_without_layout(self):
        """Test ChunkMetadata without layout (backward compatible)."""
        chunk = ChunkMetadata(
            text="No layout metadata",
            chunk_id=uuid4(),
            score=0.92,
        )

        assert chunk.text == "No layout metadata"
        assert chunk.layout is None

    def test_chunk_metadata_json_serialization(self):
        """Test ChunkMetadata JSON serialization."""
        chunk_id = uuid4()
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        layout = LayoutMetadata(page_number=1, bbox=bbox)

        chunk = ChunkMetadata(
            text="Test chunk",
            chunk_id=chunk_id,
            chunk_index=5,
            score=0.88,
            layout=layout,
        )

        json_data = chunk.model_dump()
        assert json_data["text"] == "Test chunk"
        assert json_data["chunk_id"] == chunk_id
        assert json_data["chunk_index"] == 5
        assert json_data["score"] == 0.88
        assert json_data["layout"]["page_number"] == 1
        assert json_data["layout"]["bbox"] is not None


class TestSearchResultModel:
    """Tests for SearchResult model with chunks."""

    def test_search_result_minimal(self):
        """Test SearchResult with minimal fields (backward compatible)."""
        result = SearchResult(
            search_result="Legacy result format",
            dataset_id=uuid4(),
            dataset_name=None,
        )

        assert result.search_result == "Legacy result format"
        assert result.dataset_id is not None
        assert result.chunks is None
        assert result.document is None

    def test_search_result_with_chunks(self):
        """Test SearchResult with chunks array."""
        dataset_id = uuid4()
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        layout = LayoutMetadata(page_number=1, bbox=bbox, layout_type="text")

        chunk1 = ChunkMetadata(
            text="First chunk",
            chunk_id=uuid4(),
            score=0.95,
            layout=layout,
        )
        chunk2 = ChunkMetadata(
            text="Second chunk",
            chunk_id=uuid4(),
            score=0.88,
        )

        result = SearchResult(
            search_result="Query results",
            dataset_id=dataset_id,
            dataset_name="test_dataset",
            chunks=[chunk1, chunk2],
        )

        assert result.dataset_id == dataset_id
        assert result.dataset_name == "test_dataset"
        assert len(result.chunks) == 2
        assert result.chunks[0].text == "First chunk"
        assert result.chunks[0].layout is not None
        assert result.chunks[1].layout is None  # Second chunk has no layout

    def test_search_result_with_document_reference(self):
        """Test SearchResult with document reference."""
        doc_ref = DocumentReference(
            id=uuid4(),
            name="test_document.pdf",
        )

        result = SearchResult(
            search_result="Results",
            dataset_id=uuid4(),
            dataset_name=None,
            document=doc_ref,
        )

        assert result.document is not None
        assert result.document.name == "test_document.pdf"

    def test_search_result_json_serialization(self):
        """Test SearchResult JSON serialization."""
        dataset_id = uuid4()
        chunk_id = uuid4()
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
        layout = LayoutMetadata(page_number=1, bbox=bbox)

        chunk = ChunkMetadata(
            text="Test",
            chunk_id=chunk_id,
            score=0.9,
            layout=layout,
        )

        result = SearchResult(
            search_result="Test results",
            dataset_id=dataset_id,
            dataset_name="test_ds",
            chunks=[chunk],
        )

        json_data = result.model_dump()
        assert json_data["dataset_id"] == dataset_id
        assert json_data["dataset_name"] == "test_ds"
        assert len(json_data["chunks"]) == 1
        assert json_data["chunks"][0]["text"] == "Test"
        assert json_data["chunks"][0]["layout"]["page_number"] == 1

    def test_backward_compatibility_structure(self):
        """Test that SearchResult maintains backward compatibility."""
        # Old-style result (just search_result field)
        old_result = SearchResult(
            search_result="Legacy format",
            dataset_id=uuid4(),
            dataset_name=None,
        )

        # Should have all required fields
        assert hasattr(old_result, "search_result")
        assert hasattr(old_result, "dataset_id")

        # Optional new fields should be None
        assert old_result.chunks is None
        assert old_result.document is None
