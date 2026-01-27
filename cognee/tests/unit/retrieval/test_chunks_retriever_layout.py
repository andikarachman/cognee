"""Unit tests for ChunksRetriever layout metadata extraction."""

import pytest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.search.types.SearchResult import (
    ChunkMetadata,
    LayoutMetadata,
    BoundingBox,
)


class TestLayoutInfoDetection:
    """Tests for detecting layout information in payloads."""

    def test_has_layout_info_with_bbox(self):
        """Test detection of layout info when bbox present."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {
            "text": "Sample text",
            "bounding_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}],
            "page_number": 1,
        }

        assert retriever._has_layout_info(payload) is True

    def test_has_layout_info_with_page_number_only(self):
        """Test detection of layout info with only page number."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {
            "text": "Sample text",
            "page_number": 1,
        }

        assert retriever._has_layout_info(payload) is True

    def test_has_layout_info_negative(self):
        """Test detection returns False when no layout info."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {
            "text": "Sample text",
        }

        assert retriever._has_layout_info(payload) is False

    def test_has_layout_info_with_empty_bbox(self):
        """Test detection with empty bbox list."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {
            "text": "Sample text",
            "bounding_boxes": [],
        }

        assert retriever._has_layout_info(payload) is False


class TestBboxExtraction:
    """Tests for extracting bounding box from payload."""

    def test_extract_bbox_from_dict(self):
        """Test extracting bbox from dictionary format."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {
            "bounding_boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}
            ]
        }

        bbox = retriever._extract_bbox(payload)

        assert bbox is not None
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.2
        assert bbox.x2 == 0.8
        assert bbox.y2 == 0.9

    def test_extract_bbox_multiple_boxes(self):
        """Test extracting primary bbox when multiple bboxes."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {
            "bounding_boxes": [
                {"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.2},  # Small
                {"x_min": 0.1, "y_min": 0.3, "x_max": 0.9, "y_max": 0.8},  # Large
                {"x_min": 0.1, "y_min": 0.85, "x_max": 0.5, "y_max": 0.9},  # Medium
            ]
        }

        bbox = retriever._extract_bbox(payload)

        # Should return the largest bbox
        assert bbox is not None
        assert bbox.x1 == 0.1
        assert bbox.y1 == 0.3
        assert bbox.x2 == 0.9
        assert bbox.y2 == 0.8

    def test_extract_bbox_empty_list(self):
        """Test extracting bbox from empty list returns None."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {"bounding_boxes": []}

        bbox = retriever._extract_bbox(payload)

        assert bbox is None

    def test_extract_bbox_missing_field(self):
        """Test extracting bbox when field is missing."""
        retriever = ChunksRetriever(include_layout=True)

        payload = {"text": "No bbox field"}

        bbox = retriever._extract_bbox(payload)

        assert bbox is None


class TestChunkMetadataExtraction:
    """Tests for extracting chunk metadata with layout."""

    def test_extract_chunk_metadata_with_layout(self):
        """Test extracting ChunkMetadata with layout information."""
        retriever = ChunksRetriever(include_layout=True)

        mock_result = Mock()
        mock_result.payload = {
            "text": "Sample chunk text",
            "id": str(uuid4()),
            "chunk_index": 5,
            "bounding_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}],
            "page_number": 1,
            "layout_type": "text",
            "ocr_confidence": 0.95,
        }
        mock_result.score = 0.87

        chunks = retriever._extract_chunk_metadata([mock_result])

        assert len(chunks) == 1
        chunk = chunks[0]

        assert isinstance(chunk, ChunkMetadata)
        assert chunk.text == "Sample chunk text"
        assert chunk.chunk_index == 5
        assert chunk.score == 0.87
        assert chunk.layout is not None
        assert chunk.layout.page_number == 1
        assert chunk.layout.bbox is not None
        assert chunk.layout.layout_type == "text"
        assert chunk.layout.confidence == 0.95

    def test_extract_chunk_metadata_without_layout(self):
        """Test extracting ChunkMetadata without layout information."""
        retriever = ChunksRetriever(include_layout=True)

        mock_result = Mock()
        mock_result.payload = {
            "text": "Plain chunk text",
            "id": str(uuid4()),
        }
        mock_result.score = 0.75

        chunks = retriever._extract_chunk_metadata([mock_result])

        assert len(chunks) == 1
        chunk = chunks[0]

        assert chunk.text == "Plain chunk text"
        assert chunk.layout is None  # No layout info available

    def test_extract_multiple_chunks(self):
        """Test extracting metadata from multiple results."""
        retriever = ChunksRetriever(include_layout=True)

        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.payload = {
                "text": f"Chunk {i}",
                "id": str(uuid4()),
                "page_number": i + 1,
                "bounding_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}],
            }
            mock_result.score = 0.9 - (i * 0.1)
            mock_results.append(mock_result)

        chunks = retriever._extract_chunk_metadata(mock_results)

        assert len(chunks) == 3
        assert chunks[0].text == "Chunk 0"
        assert chunks[1].text == "Chunk 1"
        assert chunks[2].text == "Chunk 2"
        assert all(c.layout is not None for c in chunks)

    def test_extract_preserves_score_order(self):
        """Test that extraction preserves score ordering."""
        retriever = ChunksRetriever(include_layout=True)

        mock_results = [
            Mock(payload={"text": "Best", "id": str(uuid4())}, score=0.95),
            Mock(payload={"text": "Good", "id": str(uuid4())}, score=0.85),
            Mock(payload={"text": "OK", "id": str(uuid4())}, score=0.75),
        ]

        chunks = retriever._extract_chunk_metadata(mock_results)

        assert chunks[0].score == 0.95
        assert chunks[1].score == 0.85
        assert chunks[2].score == 0.75


class TestGetContextMethod:
    """Tests for get_context method with layout."""

    @pytest.mark.asyncio
    async def test_get_context_with_include_layout_true(self):
        """Test get_context returns structured metadata when include_layout=True."""
        retriever = ChunksRetriever(top_k=5, include_layout=True)

        # Mock vector engine search results
        mock_result = Mock()
        mock_result.payload = {
            "text": "Sample text",
            "id": str(uuid4()),
            "page_number": 1,
            "bounding_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9}],
            "layout_type": "text",
        }
        mock_result.score = 0.9

        # Patch get_vector_engine
        with pytest.mock.patch(
            "cognee.modules.retrieval.chunks_retriever.get_vector_engine"
        ) as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.search.return_value = [mock_result]
            mock_get_engine.return_value = mock_engine

            context = await retriever.get_context("test query")

            # Should return list of ChunkMetadata
            assert isinstance(context, list)
            assert len(context) == 1
            assert isinstance(context[0], ChunkMetadata)
            assert context[0].layout is not None

    @pytest.mark.asyncio
    async def test_get_context_with_include_layout_false(self):
        """Test get_context returns raw payloads when include_layout=False."""
        retriever = ChunksRetriever(top_k=5, include_layout=False)

        # Mock vector engine search results
        mock_result = Mock()
        mock_result.payload = {"text": "Sample text", "id": str(uuid4())}
        mock_result.score = 0.9

        with pytest.mock.patch(
            "cognee.modules.retrieval.chunks_retriever.get_vector_engine"
        ) as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.search.return_value = [mock_result]
            mock_get_engine.return_value = mock_engine

            context = await retriever.get_context("test query")

            # Should return raw payloads (backward compatible)
            assert isinstance(context, list)
            assert len(context) == 1
            # Should be dict, not ChunkMetadata
            assert isinstance(context[0], dict)

    @pytest.mark.asyncio
    async def test_get_context_empty_results(self):
        """Test get_context handles empty search results."""
        retriever = ChunksRetriever(top_k=5, include_layout=True)

        with pytest.mock.patch(
            "cognee.modules.retrieval.chunks_retriever.get_vector_engine"
        ) as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.search.return_value = []
            mock_get_engine.return_value = mock_engine

            context = await retriever.get_context("test query")

            assert context == []


class TestRetrieverConfiguration:
    """Tests for retriever configuration."""

    def test_retriever_initialization_defaults(self):
        """Test retriever initializes with default values."""
        retriever = ChunksRetriever()

        assert retriever.top_k == 5  # Default
        assert retriever.include_layout is False  # Default (backward compatible)

    def test_retriever_initialization_custom(self):
        """Test retriever initializes with custom values."""
        retriever = ChunksRetriever(top_k=10, include_layout=True)

        assert retriever.top_k == 10
        assert retriever.include_layout is True

    def test_retriever_opt_in_layout(self):
        """Test that layout extraction is opt-in."""
        retriever_default = ChunksRetriever()
        retriever_explicit_false = ChunksRetriever(include_layout=False)
        retriever_enabled = ChunksRetriever(include_layout=True)

        assert retriever_default.include_layout is False
        assert retriever_explicit_false.include_layout is False
        assert retriever_enabled.include_layout is True


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    @pytest.mark.asyncio
    async def test_existing_code_still_works(self):
        """Test that existing code without include_layout still works."""
        # Old usage: no include_layout parameter
        retriever = ChunksRetriever(top_k=5)

        mock_result = Mock()
        mock_result.payload = {"text": "Sample", "id": str(uuid4())}
        mock_result.score = 0.9

        with pytest.mock.patch(
            "cognee.modules.retrieval.chunks_retriever.get_vector_engine"
        ) as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.search.return_value = [mock_result]
            mock_get_engine.return_value = mock_engine

            context = await retriever.get_context("query")

            # Should return raw payloads (old behavior)
            assert isinstance(context, list)
            assert isinstance(context[0], dict)
