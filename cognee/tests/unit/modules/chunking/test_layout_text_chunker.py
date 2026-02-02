"""Unit tests for LayoutTextChunker with block format support."""

import json
import pytest
from uuid import uuid4

from cognee.modules.chunking.LayoutTextChunker import LayoutTextChunker
from cognee.modules.chunking.models.LayoutChunk import LayoutChunk, LayoutType
from cognee.modules.data.processing.document_types import Document
from cognee.shared.data_models import BoundingBox


@pytest.fixture
def make_text_generator():
    """Factory for async text generators."""

    def _factory(*texts):
        async def gen():
            for text in texts:
                yield text

        return gen

    return _factory


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        name="test_document",
        raw_data_location="/test/path",
        external_metadata=None,
        mime_type="text/plain",
    )


async def collect_chunks(chunker):
    """Consume async generator and return list of chunks."""
    chunks = []
    async for chunk in chunker.read():
        chunks.append(chunk)
    return chunks


class TestFormatDetection:
    """Tests for detecting different formats (JSON, block, flat, plain)."""

    def test_detect_format_json(self, sample_document, make_text_generator):
        """Test detection of JSON format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        json_text = json.dumps({
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {"loader": "ocr_image_loader"},
            "document": {"total_pages": 1},
            "pages": [],
            "plain_text": "Hello world",
        })
        assert chunker._detect_format(json_text) == "json"

    def test_detect_format_legacy_block(self, sample_document, make_text_generator):
        """Test detection of legacy block format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = """[LAYOUT:title, bbox=(0.10,0.05,0.90,0.10), confidence=0.95]
Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), confidence=0.98]
[/LAYOUT]"""
        assert chunker._detect_format(text) == "legacy_block"

    def test_detect_format_legacy_flat(self, sample_document, make_text_generator):
        """Test detection of legacy flat format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = "Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), type=title, confidence=0.98]"
        assert chunker._detect_format(text) == "legacy_flat"

    def test_detect_format_plain(self, sample_document, make_text_generator):
        """Test detection of plain text."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = "Just plain text without any metadata"
        assert chunker._detect_format(text) == "plain"

    def test_has_block_format_true(self, sample_document, make_text_generator):
        """Test detection of block format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = """[LAYOUT:title, bbox=(0.10,0.05,0.90,0.10), confidence=0.95]
Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), confidence=0.98]
[/LAYOUT]"""
        assert chunker._has_block_format(text) is True

    def test_has_block_format_false(self, sample_document, make_text_generator):
        """Test detection when block format is not present."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = "Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), type=title, confidence=0.98]"
        assert chunker._has_block_format(text) is False

    def test_has_ocr_metadata_true_json(self, sample_document, make_text_generator):
        """Test detection of OCR metadata in JSON format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        json_text = json.dumps({"cognee_ocr_format": "1.0", "plain_text": "Test"})
        assert chunker._has_ocr_metadata(json_text) is True

    def test_has_ocr_metadata_true_legacy(self, sample_document, make_text_generator):
        """Test detection of OCR metadata in legacy format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = "Text [page=1, bbox=(0.1,0.2,0.8,0.9), type=text, confidence=0.95]"
        assert chunker._has_ocr_metadata(text) is True

    def test_has_ocr_metadata_false(self, sample_document, make_text_generator):
        """Test detection when OCR metadata is not present."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        text = "Just plain text without any metadata"
        assert chunker._has_ocr_metadata(text) is False


class TestJSONFormatParsing:
    """Tests for parsing JSON format OCR output."""

    def test_parse_json_format_block(self, sample_document, make_text_generator):
        """Test parsing JSON block format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {"loader": "ocr_image_loader"},
            "document": {"total_pages": 1},
            "pages": [{
                "page_number": 1,
                "width": 1000,
                "height": 1400,
                "layout_blocks": [{
                    "layout_type": "title",
                    "bbox": {"x_min": 0.1, "y_min": 0.05, "x_max": 0.9, "y_max": 0.1},
                    "confidence": 0.95,
                    "elements": [{
                        "text": "Hello World",
                        "bbox": {"x_min": 0.12, "y_min": 0.06, "x_max": 0.88, "y_max": 0.09},
                        "confidence": 0.98,
                        "layout_type": "title",
                    }],
                }],
            }],
            "plain_text": "Hello World",
        }
        json_text = json.dumps(json_data)
        plain_text, blocks = chunker._parse_json_format(json_text)

        assert plain_text == "Hello World"
        assert len(blocks) == 1
        layout_type, bbox, confidence, elements = blocks[0]
        assert layout_type == "title"
        assert bbox.x_min == pytest.approx(0.1)
        assert confidence == pytest.approx(0.95)
        assert len(elements) == 1
        assert elements[0]["text"] == "Hello World"

    def test_parse_json_format_flat(self, sample_document, make_text_generator):
        """Test parsing JSON flat format."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "flat",
            "source": {"loader": "ocr_image_loader"},
            "document": {"total_pages": 1},
            "pages": [{
                "page_number": 1,
                "width": 1000,
                "height": 1400,
                "elements": [
                    {
                        "text": "Line one",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.15},
                        "confidence": 0.95,
                        "layout_type": "text",
                    },
                    {
                        "text": "Line two",
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.9, "y_max": 0.25},
                        "confidence": 0.93,
                        "layout_type": "text",
                    },
                ],
            }],
            "plain_text": "Line one Line two",
        }
        json_text = json.dumps(json_data)
        plain_text, blocks = chunker._parse_json_format(json_text)

        assert plain_text == "Line one Line two"
        assert len(blocks) == 1  # One pseudo-block per page
        layout_type, bbox, confidence, elements = blocks[0]
        assert layout_type == "text"
        assert bbox is None  # Flat format has no layout bbox
        assert len(elements) == 2

    def test_parse_json_format_multipage(self, sample_document, make_text_generator):
        """Test parsing JSON with multiple pages."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {"loader": "ocr_pdf_loader"},
            "document": {"total_pages": 2},
            "pages": [
                {
                    "page_number": 1,
                    "layout_blocks": [{
                        "layout_type": "text",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.5},
                        "confidence": 0.92,
                        "elements": [{"text": "Page 1 content", "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.2}, "confidence": 0.95, "layout_type": "text"}],
                    }],
                },
                {
                    "page_number": 2,
                    "layout_blocks": [{
                        "layout_type": "text",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.5},
                        "confidence": 0.90,
                        "elements": [{"text": "Page 2 content", "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.2}, "confidence": 0.93, "layout_type": "text"}],
                    }],
                },
            ],
            "plain_text": "Page 1 content\n\nPage 2 content",
        }
        json_text = json.dumps(json_data)
        plain_text, blocks = chunker._parse_json_format(json_text)

        assert len(blocks) == 2
        assert blocks[0][3][0]["text"] == "Page 1 content"
        assert blocks[0][3][0]["page_number"] == 1
        assert blocks[1][3][0]["text"] == "Page 2 content"
        assert blocks[1][3][0]["page_number"] == 2

    def test_parse_json_format_invalid_json(self, sample_document, make_text_generator):
        """Test handling of invalid JSON."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        invalid_json = "not valid json {"
        plain_text, blocks = chunker._parse_json_format(invalid_json)

        assert plain_text == ""
        assert len(blocks) == 0


class TestBlockFormatParsing:
    """Tests for parsing block format."""

    def test_parse_layout_block_open(self, sample_document, make_text_generator):
        """Test parsing layout block opening tag."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        line = "[LAYOUT:title, bbox=(0.10,0.05,0.90,0.15), confidence=0.95]"
        result = chunker._parse_layout_block_open(line)

        assert result is not None
        layout_type, bbox, confidence = result
        assert layout_type == "title"
        assert bbox.x_min == pytest.approx(0.10)
        assert bbox.y_min == pytest.approx(0.05)
        assert bbox.x_max == pytest.approx(0.90)
        assert bbox.y_max == pytest.approx(0.15)
        assert confidence == pytest.approx(0.95)

    def test_parse_layout_block_open_no_confidence(
        self, sample_document, make_text_generator
    ):
        """Test parsing layout block opening tag without confidence."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        line = "[LAYOUT:text, bbox=(0.10,0.20,0.90,0.50)]"
        result = chunker._parse_layout_block_open(line)

        assert result is not None
        layout_type, bbox, confidence = result
        assert layout_type == "text"
        assert confidence is None

    def test_parse_layout_block_open_invalid(self, sample_document, make_text_generator):
        """Test parsing invalid opening tag returns None."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        line = "Not a layout tag"
        result = chunker._parse_layout_block_open(line)
        assert result is None

    def test_parse_block_element(self, sample_document, make_text_generator):
        """Test parsing text element within a block."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        line = "Sample text content [page=1, bbox=(0.12,0.16,0.85,0.22), confidence=0.95]"
        result = chunker._parse_block_element(line)

        assert result is not None
        assert result["text"] == "Sample text content"
        assert result["page_number"] == 1
        assert result["bbox"].x_min == pytest.approx(0.12)
        assert result["confidence"] == pytest.approx(0.95)

    def test_parse_block_format_single_block(self, sample_document, make_text_generator):
        """Test parsing a single layout block."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        lines = [
            "[LAYOUT:title, bbox=(0.10,0.05,0.90,0.10), confidence=0.95]",
            "Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), confidence=0.98]",
            "[/LAYOUT]",
        ]
        blocks = chunker._parse_block_format(lines)

        assert len(blocks) == 1
        layout_type, bbox, confidence, elements = blocks[0]
        assert layout_type == "title"
        assert confidence == pytest.approx(0.95)
        assert len(elements) == 1
        assert elements[0]["text"] == "Title Text"

    def test_parse_block_format_multiple_blocks(
        self, sample_document, make_text_generator
    ):
        """Test parsing multiple layout blocks."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        lines = [
            "[LAYOUT:title, bbox=(0.10,0.05,0.90,0.10), confidence=0.95]",
            "Title [page=1, bbox=(0.12,0.06,0.88,0.09), confidence=0.98]",
            "[/LAYOUT]",
            "",
            "[LAYOUT:text, bbox=(0.10,0.15,0.90,0.40), confidence=0.92]",
            "First line [page=1, bbox=(0.12,0.16,0.85,0.22), confidence=0.95]",
            "Second line [page=1, bbox=(0.12,0.24,0.80,0.30), confidence=0.93]",
            "[/LAYOUT]",
        ]
        blocks = chunker._parse_block_format(lines)

        assert len(blocks) == 2
        # First block (title)
        assert blocks[0][0] == "title"
        assert len(blocks[0][3]) == 1
        # Second block (text)
        assert blocks[1][0] == "text"
        assert len(blocks[1][3]) == 2
        assert blocks[1][3][0]["text"] == "First line"
        assert blocks[1][3][1]["text"] == "Second line"

    def test_parse_block_format_elements_get_layout_type(
        self, sample_document, make_text_generator
    ):
        """Test that parsed elements receive layout_type from block header."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        lines = [
            "[LAYOUT:heading, bbox=(0.10,0.05,0.90,0.10), confidence=0.95]",
            "Chapter 1 [page=1, bbox=(0.12,0.06,0.88,0.09), confidence=0.98]",
            "[/LAYOUT]",
        ]
        blocks = chunker._parse_block_format(lines)

        assert len(blocks) == 1
        elements = blocks[0][3]
        assert elements[0]["layout_type"] == "heading"


class TestFlatFormatParsing:
    """Tests for parsing flat format (backward compatibility)."""

    def test_parse_layout_element(self, sample_document, make_text_generator):
        """Test parsing flat format element."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        line = "Sample text [page=1, bbox=(0.12,0.16,0.85,0.22), type=text, confidence=0.95]"
        result = chunker._parse_layout_element(line)

        assert result is not None
        assert result["text"] == "Sample text"
        assert result["page_number"] == 1
        assert result["layout_type"] == "text"
        assert result["confidence"] == pytest.approx(0.95)

    def test_parse_layout_element_no_confidence(
        self, sample_document, make_text_generator
    ):
        """Test parsing flat format without confidence."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        line = "Sample text [page=1, bbox=(0.12,0.16,0.85,0.22), type=title]"
        result = chunker._parse_layout_element(line)

        assert result is not None
        assert result["text"] == "Sample text"
        assert result["layout_type"] == "title"
        assert result["confidence"] is None


class TestLayoutChunkCreation:
    """Tests for creating LayoutChunks."""

    def test_create_layout_chunk_with_layout_bbox(
        self, sample_document, make_text_generator
    ):
        """Test creating LayoutChunk with layout_bbox and layout_confidence."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        elements = [
            {
                "text": "Test content",
                "page_number": 1,
                "bbox": BoundingBox(x_min=0.12, y_min=0.16, x_max=0.85, y_max=0.22),
                "confidence": 0.95,
                "layout_type": "text",
            }
        ]
        layout_bbox = BoundingBox(x_min=0.10, y_min=0.15, x_max=0.90, y_max=0.40)

        chunk = chunker._create_layout_chunk(
            elements,
            chunk_index=0,
            layout_bbox=layout_bbox,
            layout_confidence=0.92,
            layout_type_override="text",
        )

        assert isinstance(chunk, LayoutChunk)
        assert chunk.layout_bbox is not None
        assert chunk.layout_bbox.x_min == pytest.approx(0.10)
        assert chunk.layout_confidence == pytest.approx(0.92)
        assert chunk.layout_type == LayoutType.TEXT

    def test_create_layout_chunk_without_layout_bbox(
        self, sample_document, make_text_generator
    ):
        """Test creating LayoutChunk without layout_bbox (flat format)."""
        chunker = LayoutTextChunker(
            sample_document,
            make_text_generator(""),
            max_chunk_size=512,
        )
        elements = [
            {
                "text": "Test content",
                "page_number": 1,
                "bbox": BoundingBox(x_min=0.12, y_min=0.16, x_max=0.85, y_max=0.22),
                "confidence": 0.95,
                "layout_type": "title",
            }
        ]

        chunk = chunker._create_layout_chunk(elements, chunk_index=0)

        assert isinstance(chunk, LayoutChunk)
        assert chunk.layout_bbox is None
        assert chunk.layout_confidence is None
        assert chunk.layout_type == LayoutType.TITLE


@pytest.mark.asyncio
class TestLayoutTextChunkerRead:
    """Tests for LayoutTextChunker.read() method."""

    async def test_read_block_format(self, sample_document, make_text_generator):
        """Test reading block format produces LayoutChunks with layout_bbox."""
        text = """[LAYOUT:title, bbox=(0.10,0.05,0.90,0.10), confidence=0.95]
Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), confidence=0.98]
[/LAYOUT]

[LAYOUT:text, bbox=(0.10,0.15,0.90,0.40), confidence=0.92]
First paragraph line [page=1, bbox=(0.12,0.16,0.85,0.22), confidence=0.95]
Second paragraph line [page=1, bbox=(0.12,0.24,0.80,0.30), confidence=0.93]
[/LAYOUT]
"""
        get_text = make_text_generator(text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 2

        # First chunk (title)
        assert isinstance(chunks[0], LayoutChunk)
        assert chunks[0].layout_type == LayoutType.TITLE
        assert chunks[0].layout_bbox is not None
        assert chunks[0].layout_confidence == pytest.approx(0.95)
        assert "Title Text" in chunks[0].text

        # Second chunk (text)
        assert isinstance(chunks[1], LayoutChunk)
        assert chunks[1].layout_type == LayoutType.TEXT
        assert chunks[1].layout_bbox is not None
        assert chunks[1].layout_confidence == pytest.approx(0.92)
        assert "First paragraph line" in chunks[1].text
        assert "Second paragraph line" in chunks[1].text

    async def test_read_flat_format(self, sample_document, make_text_generator):
        """Test reading flat format (backward compatibility)."""
        text = """Title Text [page=1, bbox=(0.12,0.06,0.88,0.09), type=title, confidence=0.98]
Paragraph text [page=1, bbox=(0.12,0.16,0.85,0.22), type=text, confidence=0.95]
"""
        get_text = make_text_generator(text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 1  # Both fit in one chunk
        assert isinstance(chunks[0], LayoutChunk)
        # Flat format doesn't have layout_bbox
        assert chunks[0].layout_bbox is None
        assert chunks[0].layout_confidence is None

    async def test_read_no_metadata_fallback(self, sample_document, make_text_generator):
        """Test fallback to regular chunking when no metadata present."""
        text = "Just plain text without any OCR metadata."
        get_text = make_text_generator(text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        # Should produce DocumentChunk, not LayoutChunk
        assert len(chunks) >= 1

    async def test_read_block_format_respects_chunk_size(
        self, sample_document, make_text_generator
    ):
        """Test that block format respects max_chunk_size."""
        # Create a block with many elements that exceed chunk size
        text = """[LAYOUT:text, bbox=(0.10,0.15,0.90,0.80), confidence=0.92]
Line number zero with some words [page=1, bbox=(0.12,0.16,0.85,0.19), confidence=0.95]
Line number one with some words [page=1, bbox=(0.12,0.19,0.85,0.22), confidence=0.95]
Line number two with some words [page=1, bbox=(0.12,0.22,0.85,0.25), confidence=0.95]
Line number three with some words [page=1, bbox=(0.12,0.25,0.85,0.28), confidence=0.95]
Line number four with some words [page=1, bbox=(0.12,0.28,0.85,0.31), confidence=0.95]
Line number five with some words [page=1, bbox=(0.12,0.31,0.85,0.34), confidence=0.95]
Line number six with some words [page=1, bbox=(0.12,0.34,0.85,0.37), confidence=0.95]
Line number seven with some words [page=1, bbox=(0.12,0.37,0.85,0.40), confidence=0.95]
Line number eight with some words [page=1, bbox=(0.12,0.40,0.85,0.43), confidence=0.95]
Line number nine with some words [page=1, bbox=(0.12,0.43,0.85,0.46), confidence=0.95]
Line number ten with some words [page=1, bbox=(0.12,0.46,0.85,0.49), confidence=0.95]
Line number eleven with some words [page=1, bbox=(0.12,0.49,0.85,0.52), confidence=0.95]
Line number twelve with some words [page=1, bbox=(0.12,0.52,0.85,0.55), confidence=0.95]
Line number thirteen with some words [page=1, bbox=(0.12,0.55,0.85,0.58), confidence=0.95]
Line number fourteen with some words [page=1, bbox=(0.12,0.58,0.85,0.61), confidence=0.95]
Line number fifteen with some words [page=1, bbox=(0.12,0.61,0.85,0.64), confidence=0.95]
Line number sixteen with some words [page=1, bbox=(0.12,0.64,0.85,0.67), confidence=0.95]
Line number seventeen with some words [page=1, bbox=(0.12,0.67,0.85,0.70), confidence=0.95]
Line number eighteen with some words [page=1, bbox=(0.12,0.70,0.85,0.73), confidence=0.95]
Line number nineteen with some words [page=1, bbox=(0.12,0.73,0.85,0.76), confidence=0.95]
[/LAYOUT]
"""
        get_text = make_text_generator(text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=50)
        chunks = await collect_chunks(chunker)

        # Should produce multiple chunks due to size limit
        assert len(chunks) > 1
        # All chunks should have the same layout_bbox since from same block
        for chunk in chunks:
            assert isinstance(chunk, LayoutChunk)
            assert chunk.layout_type == LayoutType.TEXT

    async def test_read_json_format_block(self, sample_document, make_text_generator):
        """Test reading JSON block format produces LayoutChunks."""
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {"loader": "ocr_image_loader", "ocr_engine": "paddleocr"},
            "document": {"total_pages": 1, "content_hash": "abc123"},
            "pages": [{
                "page_number": 1,
                "width": 1000,
                "height": 1400,
                "layout_blocks": [
                    {
                        "layout_type": "title",
                        "bbox": {"x_min": 0.1, "y_min": 0.05, "x_max": 0.9, "y_max": 0.1},
                        "confidence": 0.95,
                        "elements": [{
                            "text": "Document Title",
                            "bbox": {"x_min": 0.12, "y_min": 0.06, "x_max": 0.88, "y_max": 0.09},
                            "confidence": 0.98,
                            "layout_type": "title",
                        }],
                    },
                    {
                        "layout_type": "text",
                        "bbox": {"x_min": 0.1, "y_min": 0.15, "x_max": 0.9, "y_max": 0.4},
                        "confidence": 0.92,
                        "elements": [
                            {
                                "text": "First paragraph text",
                                "bbox": {"x_min": 0.12, "y_min": 0.16, "x_max": 0.85, "y_max": 0.22},
                                "confidence": 0.95,
                                "layout_type": "text",
                            },
                            {
                                "text": "Second paragraph text",
                                "bbox": {"x_min": 0.12, "y_min": 0.24, "x_max": 0.80, "y_max": 0.30},
                                "confidence": 0.93,
                                "layout_type": "text",
                            },
                        ],
                    },
                ],
            }],
            "plain_text": "Document Title First paragraph text Second paragraph text",
        }
        json_text = json.dumps(json_data)
        get_text = make_text_generator(json_text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 2

        # First chunk (title)
        assert isinstance(chunks[0], LayoutChunk)
        assert chunks[0].layout_type == LayoutType.TITLE
        assert chunks[0].layout_bbox is not None
        assert chunks[0].layout_confidence == pytest.approx(0.95)
        assert "Document Title" in chunks[0].text

        # Second chunk (text)
        assert isinstance(chunks[1], LayoutChunk)
        assert chunks[1].layout_type == LayoutType.TEXT
        assert chunks[1].layout_bbox is not None
        assert chunks[1].layout_confidence == pytest.approx(0.92)
        assert "First paragraph text" in chunks[1].text

    async def test_read_json_format_flat(self, sample_document, make_text_generator):
        """Test reading JSON flat format produces LayoutChunks."""
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "flat",
            "source": {"loader": "ocr_image_loader", "ocr_engine": "paddleocr"},
            "document": {"total_pages": 1, "content_hash": "abc123"},
            "pages": [{
                "page_number": 1,
                "width": 1000,
                "height": 1400,
                "elements": [
                    {
                        "text": "First line",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.15},
                        "confidence": 0.95,
                        "layout_type": "text",
                    },
                    {
                        "text": "Second line",
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.9, "y_max": 0.25},
                        "confidence": 0.93,
                        "layout_type": "text",
                    },
                ],
            }],
            "plain_text": "First line Second line",
        }
        json_text = json.dumps(json_data)
        get_text = make_text_generator(json_text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 1
        assert isinstance(chunks[0], LayoutChunk)
        # Flat format doesn't have layout_bbox
        assert chunks[0].layout_bbox is None
        assert "First line" in chunks[0].text
        assert "Second line" in chunks[0].text

    async def test_read_json_format_multipage(self, sample_document, make_text_generator):
        """Test reading JSON with multiple pages."""
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {"loader": "ocr_pdf_loader"},
            "document": {"total_pages": 2},
            "pages": [
                {
                    "page_number": 1,
                    "layout_blocks": [{
                        "layout_type": "text",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.5},
                        "confidence": 0.92,
                        "elements": [{"text": "Page one content", "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.2}, "confidence": 0.95, "layout_type": "text"}],
                    }],
                },
                {
                    "page_number": 2,
                    "layout_blocks": [{
                        "layout_type": "text",
                        "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.5},
                        "confidence": 0.90,
                        "elements": [{"text": "Page two content", "bbox": {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.2}, "confidence": 0.93, "layout_type": "text"}],
                    }],
                },
            ],
            "plain_text": "Page one content\n\nPage two content",
        }
        json_text = json.dumps(json_data)
        get_text = make_text_generator(json_text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        assert len(chunks) == 2
        assert chunks[0].page_number == 1
        assert "Page one content" in chunks[0].text
        assert chunks[1].page_number == 2
        assert "Page two content" in chunks[1].text

    async def test_read_json_format_empty_blocks_falls_back(
        self, sample_document, make_text_generator
    ):
        """Test that empty blocks with plain_text falls back to regular chunking."""
        json_data = {
            "cognee_ocr_format": "1.0",
            "format_type": "block",
            "source": {"loader": "ocr_image_loader"},
            "document": {"total_pages": 1},
            "pages": [],
            "plain_text": "This is the plain text content.",
        }
        json_text = json.dumps(json_data)
        get_text = make_text_generator(json_text)
        chunker = LayoutTextChunker(sample_document, get_text, max_chunk_size=512)
        chunks = await collect_chunks(chunker)

        # Should produce at least one chunk from plain_text
        assert len(chunks) >= 1
        assert "plain text content" in chunks[0].text
