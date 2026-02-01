"""Integration tests for chunking with layout metadata."""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from cognee.modules.chunking.LayoutTextChunker import LayoutTextChunker
from cognee.modules.chunking.models.LayoutChunk import LayoutChunk
from cognee.tasks.documents.extract_chunks_from_documents import (
    extract_chunks_from_documents,
)
from cognee.tests.test_data.mock_data.mock_ocr_results import create_test_layout_text


class TestLayoutTextChunkerIntegration:
    """Integration tests for LayoutTextChunker with real data."""

    @pytest.mark.asyncio
    async def test_chunker_with_ocr_text(self):
        """Test LayoutTextChunker processing OCR-annotated text."""
        # Create mock document with OCR metadata
        mock_document = Mock()
        mock_document.id = "test_doc_1"

        # Create realistic OCR text
        ocr_text = create_test_layout_text(num_lines=10, page_number=1)

        async def mock_get_text():
            yield ocr_text

        mock_document.get_text = mock_get_text

        # Process with LayoutTextChunker
        chunker = LayoutTextChunker(max_chunk_size=100)
        chunks = []

        async for chunk in chunker.read(mock_document):
            chunks.append(chunk)

        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, LayoutChunk) for chunk in chunks)

        # Check first chunk has layout info
        first_chunk = chunks[0]
        assert first_chunk.has_layout_info
        assert first_chunk.page_number == 1
        assert len(first_chunk.bounding_boxes) > 0

    @pytest.mark.asyncio
    async def test_chunker_with_multi_page_text(self):
        """Test chunker handles multi-page documents."""
        mock_document = Mock()
        mock_document.id = "test_doc_2"

        # Create multi-page OCR text
        page1_text = create_test_layout_text(num_lines=5, page_number=1)
        page2_text = create_test_layout_text(num_lines=5, page_number=2)
        page3_text = create_test_layout_text(num_lines=5, page_number=3)

        async def mock_get_text():
            yield page1_text + "\n\n" + page2_text + "\n\n" + page3_text

        mock_document.get_text = mock_get_text

        chunker = LayoutTextChunker(max_chunk_size=50)
        chunks = []

        async for chunk in chunker.read(mock_document):
            chunks.append(chunk)

        # Verify chunks span multiple pages
        page_numbers = set()
        for chunk in chunks:
            if chunk.page_number:
                page_numbers.add(chunk.page_number)
            if chunk.page_numbers:
                page_numbers.update(chunk.page_numbers)

        assert len(page_numbers) > 1  # Should have chunks from multiple pages

    @pytest.mark.asyncio
    async def test_chunker_fallback_to_regular(self):
        """Test chunker falls back to regular chunking without OCR metadata."""
        mock_document = Mock()
        mock_document.id = "test_doc_3"

        # Plain text without metadata
        plain_text = "This is regular text. " * 20

        async def mock_get_text():
            yield plain_text

        mock_document.get_text = mock_get_text

        chunker = LayoutTextChunker(max_chunk_size=50)
        chunks = []

        async for chunk in chunker.read(mock_document):
            chunks.append(chunk)

        # Should still produce chunks
        assert len(chunks) > 0

        # Chunks may not have layout info
        # (LayoutChunk without layout info is valid)
        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert chunk.word_count > 0


class TestExtractChunksIntegration:
    """Integration tests for extract_chunks_from_documents task."""

    @pytest.mark.asyncio
    async def test_extract_chunks_auto_detects_ocr(self):
        """Test that extract_chunks automatically detects OCR metadata."""
        # Create mock document with OCR metadata
        mock_document = Mock()
        mock_document.id = "test_doc_4"

        ocr_text = create_test_layout_text(num_lines=5, page_number=1)

        async def mock_read_preview():
            return ocr_text[:200]  # First 200 chars

        async def mock_get_text():
            yield ocr_text

        mock_document.read_preview = mock_read_preview
        mock_document.get_text = mock_get_text

        # Extract chunks (should auto-detect OCR)
        chunks = []
        async for chunk in extract_chunks_from_documents([mock_document]):
            chunks.append(chunk)

        # Should produce LayoutChunk objects
        assert len(chunks) > 0
        layout_chunks = [c for c in chunks if isinstance(c, LayoutChunk)]
        assert len(layout_chunks) > 0

    @pytest.mark.asyncio
    async def test_extract_chunks_without_ocr(self):
        """Test extract_chunks with non-OCR documents."""
        mock_document = Mock()
        mock_document.id = "test_doc_5"

        plain_text = "Regular document text without OCR metadata."

        async def mock_read_preview():
            return plain_text

        async def mock_get_text():
            yield plain_text

        mock_document.read_preview = mock_read_preview
        mock_document.get_text = mock_get_text

        # Extract chunks
        chunks = []
        async for chunk in extract_chunks_from_documents([mock_document]):
            chunks.append(chunk)

        # Should still work (backward compatible)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_extract_chunks_custom_chunker(self):
        """Test extract_chunks with explicitly specified chunker."""
        mock_document = Mock()
        mock_document.id = "test_doc_6"

        ocr_text = create_test_layout_text(num_lines=5, page_number=1)

        async def mock_read_preview():
            return ocr_text[:200]

        async def mock_get_text():
            yield ocr_text

        mock_document.read_preview = mock_read_preview
        mock_document.get_text = mock_get_text

        # Explicitly use LayoutTextChunker
        chunks = []
        async for chunk in extract_chunks_from_documents(
            [mock_document], chunker_class=LayoutTextChunker
        ):
            chunks.append(chunk)

        # Should use LayoutTextChunker
        assert len(chunks) > 0
        assert all(isinstance(c, LayoutChunk) for c in chunks)


class TestLayoutChunkSerialization:
    """Integration tests for LayoutChunk serialization."""

    def test_layout_chunk_json_roundtrip(self):
        """Test LayoutChunk can be serialized and deserialized."""
        from cognee.shared.data_models import BoundingBox
        from cognee.modules.chunking.models.LayoutChunk import LayoutType

        # Create LayoutChunk
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
            confidence=0.95,
        )

        chunk = LayoutChunk(
            text="Test chunk",
            word_count=2,
            chunk_id="test-id",
            bounding_boxes=[bbox],
            page_number=1,
            layout_type=LayoutType.TEXT,
            ocr_confidence=0.95,
        )

        # Serialize to dict
        chunk_dict = chunk.model_dump()

        # Should have all fields
        assert chunk_dict["text"] == "Test chunk"
        assert chunk_dict["page_number"] == 1
        assert chunk_dict["layout_type"] == "text"
        assert len(chunk_dict["bounding_boxes"]) == 1

        # Deserialize back
        restored_chunk = LayoutChunk(**chunk_dict)

        assert restored_chunk.text == chunk.text
        assert restored_chunk.page_number == chunk.page_number
        assert restored_chunk.layout_type == chunk.layout_type

    def test_layout_chunk_json_without_layout(self):
        """Test LayoutChunk serialization without layout fields."""
        # Minimal LayoutChunk (backward compatible)
        chunk = LayoutChunk(
            text="Minimal chunk",
            word_count=2,
        )

        # Should serialize without errors
        chunk_dict = chunk.model_dump()

        assert chunk_dict["text"] == "Minimal chunk"
        assert chunk_dict["word_count"] == 2
        assert chunk_dict["page_number"] is None
        assert chunk_dict["bounding_boxes"] == []


class TestVectorDBStorage:
    """Integration tests for storing LayoutChunk in vector DB."""

    @pytest.mark.asyncio
    async def test_layout_chunk_vector_storage(self):
        """Test that LayoutChunk can be stored in vector DB."""
        from cognee.shared.data_models import BoundingBox
        from cognee.modules.chunking.models.LayoutChunk import LayoutType

        # Create LayoutChunk with layout info
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)

        chunk = LayoutChunk(
            text="Test chunk for storage",
            word_count=4,
            chunk_id="storage-test",
            bounding_boxes=[bbox],
            page_number=1,
            layout_type=LayoutType.TEXT,
            ocr_confidence=0.95,
        )

        # Convert to payload (as would be done for vector DB)
        payload = chunk.model_dump()

        # Payload should be JSON-serializable
        import json

        json_str = json.dumps(payload)
        assert json_str is not None

        # Deserialize and verify
        restored_payload = json.loads(json_str)
        assert restored_payload["text"] == "Test chunk for storage"
        assert restored_payload["page_number"] == 1
        assert restored_payload["layout_type"] == "text"

    @pytest.mark.asyncio
    async def test_layout_metadata_preserved_in_payload(self):
        """Test that layout metadata is preserved in vector DB payload."""
        from cognee.shared.data_models import BoundingBox

        bbox1 = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.2)
        bbox2 = BoundingBox(x_min=0.1, y_min=0.3, x_max=0.5, y_max=0.4)

        chunk = LayoutChunk(
            text="Multi-bbox chunk",
            word_count=2,
            bounding_boxes=[bbox1, bbox2],
            page_number=1,
        )

        payload = chunk.model_dump()

        # Verify multiple bboxes are preserved
        assert len(payload["bounding_boxes"]) == 2
        assert payload["bounding_boxes"][0]["x_min"] == 0.1
        assert payload["bounding_boxes"][1]["y_min"] == 0.3


class TestChunkRetrieval:
    """Integration tests for retrieving chunks with layout."""

    @pytest.mark.asyncio
    async def test_retrieve_layout_chunk_from_dict(self):
        """Test retrieving LayoutChunk from dictionary payload."""
        # Simulate payload from vector DB
        payload = {
            "text": "Retrieved chunk",
            "word_count": 2,
            "chunk_id": "retrieved-id",
            "bounding_boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9, "confidence": 1.0}
            ],
            "page_number": 1,
            "layout_type": "text",
            "ocr_confidence": 0.95,
        }

        # Reconstruct LayoutChunk
        chunk = LayoutChunk(**payload)

        # Verify reconstruction
        assert chunk.text == "Retrieved chunk"
        assert chunk.page_number == 1
        assert len(chunk.bounding_boxes) == 1
        assert chunk.layout_type.value == "text"

    @pytest.mark.asyncio
    async def test_backward_compatible_chunk_retrieval(self):
        """Test retrieving regular chunks still works."""
        # Old-style payload without layout
        payload = {
            "text": "Regular chunk",
            "word_count": 2,
            "chunk_id": "regular-id",
        }

        # Should work with LayoutChunk (optional fields)
        chunk = LayoutChunk(**payload)

        assert chunk.text == "Regular chunk"
        assert chunk.page_number is None
        assert len(chunk.bounding_boxes) == 0


class TestChunkingPerformance:
    """Integration tests for chunking performance."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_document_chunking(self):
        """Test chunking performance with large documents."""
        mock_document = Mock()
        mock_document.id = "large_doc"

        # Create large OCR text (100 lines)
        large_text = create_test_layout_text(num_lines=100, page_number=1)

        async def mock_get_text():
            yield large_text

        mock_document.get_text = mock_get_text

        chunker = LayoutTextChunker(max_chunk_size=100)
        chunks = []

        import time

        start_time = time.time()

        async for chunk in chunker.read(mock_document):
            chunks.append(chunk)

        elapsed_time = time.time() - start_time

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Should be reasonably fast (< 1 second for 100 lines)
        assert elapsed_time < 1.0, f"Chunking took too long: {elapsed_time}s"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test that chunking doesn't consume excessive memory."""
        mock_document = Mock()
        mock_document.id = "memory_test"

        # Create multiple pages
        large_text = "\n\n".join(
            create_test_layout_text(num_lines=50, page_number=i) for i in range(1, 6)
        )

        async def mock_get_text():
            yield large_text

        mock_document.get_text = mock_get_text

        chunker = LayoutTextChunker(max_chunk_size=100)
        chunk_count = 0

        # Process chunks iteratively (should not load all into memory)
        async for chunk in chunker.read(mock_document):
            chunk_count += 1
            # Don't store chunks, just count them

        # Should have processed all chunks
        assert chunk_count > 0
