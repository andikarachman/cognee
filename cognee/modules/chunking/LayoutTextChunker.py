"""Layout-aware text chunker that parses OCR metadata."""

import re
from uuid import NAMESPACE_OID, uuid5
from typing import List, Dict, Any, Optional
from cognee.shared.logging_utils import get_logger
from cognee.tasks.chunks import chunk_by_paragraph
from cognee.modules.chunking.Chunker import Chunker
from cognee.shared.data_models import BoundingBox
from .models.DocumentChunk import DocumentChunk
from .models.LayoutChunk import LayoutChunk, LayoutType

logger = get_logger()


class LayoutTextChunker(Chunker):
    """
    Chunker that parses OCR/layout metadata from text.

    Expects text format:
        Text content [page=N, bbox=(x,y,x,y), type=TYPE, confidence=C]

    Falls back to regular DocumentChunk if no metadata found.
    """

    # Regex pattern to match layout metadata
    LAYOUT_PATTERN = re.compile(
        r"^(.*?)\s*\[page=(\d+),\s*bbox=\(([\d.]+),([\d.]+),([\d.]+),([\d.]+)\),"
        r"\s*type=(\w+)(?:,\s*confidence=([\d.]+))?\]$"
    )

    def _has_ocr_metadata(self, text: str) -> bool:
        """
        Check if text contains OCR metadata markers.

        Args:
            text: Text to check

        Returns:
            True if text has OCR metadata
        """
        return "[page=" in text and "bbox=" in text

    def _parse_layout_element(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line with layout metadata.

        Args:
            line: Line of text with metadata

        Returns:
            Dictionary with parsed data or None if no match
        """
        match = self.LAYOUT_PATTERN.match(line.strip())
        if not match:
            return None

        text, page, x_min, y_min, x_max, y_max, layout_type, confidence = match.groups()

        return {
            "text": text.strip(),
            "page_number": int(page),
            "bbox": BoundingBox(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
                confidence=float(confidence) if confidence else 1.0,
            ),
            "layout_type": layout_type,
            "confidence": float(confidence) if confidence else None,
        }

    def _group_elements_by_chunks(
        self, elements: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group layout elements into chunks based on token size.

        Args:
            elements: List of parsed layout elements

        Returns:
            List of element groups (chunks)
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for element in elements:
            # Estimate token count (rough approximation)
            text = element["text"]
            element_size = len(text.split())

            if current_size + element_size <= self.max_chunk_size and current_chunk:
                current_chunk.append(element)
                current_size += element_size
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [element]
                current_size = element_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_layout_chunk(
        self, elements: List[Dict[str, Any]], chunk_index: int
    ) -> LayoutChunk:
        """
        Create a LayoutChunk from a group of elements.

        Args:
            elements: List of layout elements
            chunk_index: Index of this chunk

        Returns:
            LayoutChunk with aggregated metadata
        """
        # Combine text
        combined_text = " ".join(elem["text"] for elem in elements)

        # Collect all bounding boxes
        bboxes = [elem["bbox"] for elem in elements]

        # Get page numbers
        page_numbers = sorted(set(elem["page_number"] for elem in elements))
        primary_page = page_numbers[0] if page_numbers else None

        # Determine dominant layout type (most common)
        layout_types = [elem["layout_type"] for elem in elements]
        dominant_type = max(set(layout_types), key=layout_types.count)

        # Calculate average confidence if available
        confidences = [
            elem["confidence"] for elem in elements if elem["confidence"] is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Estimate chunk size (word count)
        chunk_size = len(combined_text.split())

        try:
            return LayoutChunk(
                id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{chunk_index}"),
                text=combined_text,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=chunk_index,
                cut_type="layout",
                contains=[],
                metadata={"index_fields": ["text"]},
                bounding_boxes=bboxes,
                page_number=primary_page,
                page_numbers=page_numbers if len(page_numbers) > 1 else None,
                layout_type=LayoutType(dominant_type)
                if dominant_type in LayoutType.__members__
                else LayoutType.UNKNOWN,
                ocr_confidence=avg_confidence,
            )
        except Exception as e:
            logger.error(f"Failed to create LayoutChunk: {e}")
            # Fallback to regular DocumentChunk
            return DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{chunk_index}"),
                text=combined_text,
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=chunk_index,
                cut_type="layout",
                contains=[],
                metadata={"index_fields": ["text"]},
            )

    async def read(self):
        """
        Read and chunk text with layout awareness.

        Yields:
            LayoutChunk or DocumentChunk objects
        """
        # Check if document has layout metadata
        text_preview = ""
        async for content_text in self.get_text():
            text_preview = content_text[:1000]  # Sample first 1000 chars
            break

        has_layout = self._has_ocr_metadata(text_preview)

        if not has_layout:
            # No layout metadata, use regular paragraph chunking
            logger.info("No layout metadata detected, using regular chunking")
            async for chunk in self._fallback_to_regular_chunking():
                yield chunk
            return

        # Parse layout metadata
        logger.info("Layout metadata detected, using layout-aware chunking")

        elements = []
        async for content_text in self.get_text():
            lines = content_text.split("\n")
            for line in lines:
                if not line.strip() or line.strip().startswith("Page "):
                    continue  # Skip empty lines and page headers

                parsed = self._parse_layout_element(line)
                if parsed:
                    elements.append(parsed)

        if not elements:
            logger.warning("No valid layout elements parsed, falling back to regular chunking")
            async for chunk in self._fallback_to_regular_chunking():
                yield chunk
            return

        # Group elements into chunks
        element_groups = self._group_elements_by_chunks(elements)

        # Create LayoutChunk for each group
        for chunk_index, group in enumerate(element_groups):
            yield self._create_layout_chunk(group, chunk_index)

    async def _fallback_to_regular_chunking(self):
        """Fallback to regular paragraph-based chunking."""
        paragraph_chunks = []
        chunk_index = 0

        async for content_text in self.get_text():
            for chunk_data in chunk_by_paragraph(
                content_text,
                self.max_chunk_size,
                batch_paragraphs=True,
            ):
                if self.chunk_size + chunk_data["chunk_size"] <= self.max_chunk_size:
                    paragraph_chunks.append(chunk_data)
                    self.chunk_size += chunk_data["chunk_size"]
                else:
                    if len(paragraph_chunks) == 0:
                        yield DocumentChunk(
                            id=chunk_data["chunk_id"],
                            text=chunk_data["text"],
                            chunk_size=chunk_data["chunk_size"],
                            is_part_of=self.document,
                            chunk_index=chunk_index,
                            cut_type=chunk_data["cut_type"],
                            contains=[],
                            metadata={"index_fields": ["text"]},
                        )
                        paragraph_chunks = []
                        self.chunk_size = 0
                    else:
                        chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                        yield DocumentChunk(
                            id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{chunk_index}"),
                            text=chunk_text,
                            chunk_size=self.chunk_size,
                            is_part_of=self.document,
                            chunk_index=chunk_index,
                            cut_type=paragraph_chunks[-1]["cut_type"],
                            contains=[],
                            metadata={"index_fields": ["text"]},
                        )
                        paragraph_chunks = [chunk_data]
                        self.chunk_size = chunk_data["chunk_size"]

                    chunk_index += 1

        if len(paragraph_chunks) > 0:
            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{chunk_index}"),
                text=" ".join(chunk["text"] for chunk in paragraph_chunks),
                chunk_size=self.chunk_size,
                is_part_of=self.document,
                chunk_index=chunk_index,
                cut_type=paragraph_chunks[-1]["cut_type"],
                contains=[],
                metadata={"index_fields": ["text"]},
            )
