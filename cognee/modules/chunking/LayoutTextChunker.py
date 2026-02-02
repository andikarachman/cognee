"""Layout-aware text chunker that parses OCR metadata."""

import json
import re
from uuid import NAMESPACE_OID, uuid5
from typing import List, Dict, Any, Optional, Tuple
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

    Supports three formats:
    1. JSON format (new, preferred):
        {"cognee_ocr_format": "1.0", "format_type": "block"|"flat", ...}

    2. Block format (legacy):
        [LAYOUT:type, bbox=(x,y,x,y), confidence=C]
        Text content [page=N, bbox=(x,y,x,y), confidence=C]
        [/LAYOUT]

    3. Flat format (legacy, backward compat):
        Text content [page=N, bbox=(x,y,x,y), type=TYPE, confidence=C]

    Falls back to regular DocumentChunk if no metadata found.
    """

    # Regex pattern for layout block opening tag (legacy)
    LAYOUT_BLOCK_OPEN_PATTERN = re.compile(
        r"^\[LAYOUT:(\w+),\s*bbox=\(([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)"
        r"(?:,\s*confidence=([\d.]+))?\]$"
    )

    # Regex pattern for layout block closing tag (legacy)
    LAYOUT_BLOCK_CLOSE_PATTERN = re.compile(r"^\[/LAYOUT\]$")

    # Regex pattern for text element within layout block (legacy, no type field)
    BLOCK_ELEMENT_PATTERN = re.compile(
        r"^(.*?)\s*\[page=(\d+),\s*bbox=\(([\d.]+),([\d.]+),([\d.]+),([\d.]+)\),"
        r"\s*confidence=([\d.]+)\]$"
    )

    # Regex pattern to match flat layout metadata (legacy)
    LAYOUT_PATTERN = re.compile(
        r"^(.*?)\s*\[page=(\d+),\s*bbox=\(([\d.]+),([\d.]+),([\d.]+),([\d.]+)\),"
        r"\s*type=(\w+)(?:,\s*confidence=([\d.]+))?\]$"
    )

    def _detect_format(self, text: str) -> str:
        """
        Detect the format of the input text.

        Args:
            text: Text to analyze

        Returns:
            Format string: "json", "legacy_block", "legacy_flat", or "plain"
        """
        stripped = text.strip()

        # Check for JSON format
        if stripped.startswith("{") and '"cognee_ocr_format"' in text[:500]:
            return "json"

        # Check for legacy block format
        if "[LAYOUT:" in text and "[/LAYOUT]" in text:
            return "legacy_block"

        # Check for legacy flat format
        if "[page=" in text and "bbox=" in text:
            return "legacy_flat"

        return "plain"

    def _has_ocr_metadata(self, text: str) -> bool:
        """
        Check if text contains any form of OCR metadata.

        Args:
            text: Text to check

        Returns:
            True if text has OCR metadata
        """
        detected_format = self._detect_format(text)
        return detected_format != "plain"

    def _has_block_format(self, text: str) -> bool:
        """
        Check if text uses the legacy block format with [LAYOUT:...][/LAYOUT].

        Args:
            text: Text to check

        Returns:
            True if text uses block format
        """
        return "[LAYOUT:" in text and "[/LAYOUT]" in text

    def _parse_json_format(
        self, text: str
    ) -> Tuple[str, List[Tuple[str, BoundingBox, Optional[float], List[Dict[str, Any]]]]]:
        """
        Parse JSON format OCR output.

        Args:
            text: JSON string

        Returns:
            Tuple of (plain_text, list of (layout_type, bbox, confidence, elements))
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON OCR format: {e}")
            return "", []

        plain_text = data.get("plain_text", "")
        format_type = data.get("format_type", "flat")
        pages = data.get("pages", [])

        blocks = []

        for page in pages:
            page_number = page.get("page_number", 1)

            if format_type == "block" and page.get("layout_blocks"):
                # Parse block format
                for layout_block in page["layout_blocks"]:
                    layout_type = layout_block.get("layout_type", "text")
                    bbox_data = layout_block.get("bbox", {})
                    layout_confidence = layout_block.get("confidence")

                    layout_bbox = BoundingBox(
                        x_min=bbox_data.get("x_min", 0),
                        y_min=bbox_data.get("y_min", 0),
                        x_max=bbox_data.get("x_max", 1),
                        y_max=bbox_data.get("y_max", 1),
                    )

                    elements = []
                    for elem in layout_block.get("elements", []):
                        elem_bbox_data = elem.get("bbox", {})
                        elem_bbox = BoundingBox(
                            x_min=elem_bbox_data.get("x_min", 0),
                            y_min=elem_bbox_data.get("y_min", 0),
                            x_max=elem_bbox_data.get("x_max", 1),
                            y_max=elem_bbox_data.get("y_max", 1),
                            confidence=elem.get("confidence", 1.0),
                        )

                        elements.append(
                            {
                                "text": elem.get("text", ""),
                                "page_number": page_number,
                                "bbox": elem_bbox,
                                "confidence": elem.get("confidence"),
                                "layout_type": elem.get("layout_type", layout_type),
                            }
                        )

                    if elements:
                        blocks.append((layout_type, layout_bbox, layout_confidence, elements))

            elif page.get("elements"):
                # Parse flat format - group all elements as one block per page
                elements = []
                for elem in page["elements"]:
                    elem_bbox_data = elem.get("bbox", {})
                    elem_bbox = BoundingBox(
                        x_min=elem_bbox_data.get("x_min", 0),
                        y_min=elem_bbox_data.get("y_min", 0),
                        x_max=elem_bbox_data.get("x_max", 1),
                        y_max=elem_bbox_data.get("y_max", 1),
                        confidence=elem.get("confidence", 1.0),
                    )

                    elements.append(
                        {
                            "text": elem.get("text", ""),
                            "page_number": page_number,
                            "bbox": elem_bbox,
                            "confidence": elem.get("confidence"),
                            "layout_type": elem.get("layout_type", "text"),
                        }
                    )

                if elements:
                    # Create a pseudo-block covering all elements
                    blocks.append(("text", None, None, elements))

        return plain_text, blocks

    def _parse_layout_block_open(
        self, line: str
    ) -> Optional[Tuple[str, BoundingBox, Optional[float]]]:
        """
        Parse a layout block opening tag (legacy format).

        Args:
            line: Line of text

        Returns:
            Tuple of (layout_type, bbox, confidence) or None if no match
        """
        match = self.LAYOUT_BLOCK_OPEN_PATTERN.match(line.strip())
        if not match:
            return None

        layout_type, x_min, y_min, x_max, y_max, confidence = match.groups()

        bbox = BoundingBox(
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
        )

        return (
            layout_type,
            bbox,
            float(confidence) if confidence else None,
        )

    def _parse_block_element(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a text element line within a layout block (legacy format).

        Args:
            line: Line of text with metadata

        Returns:
            Dictionary with parsed data or None if no match
        """
        match = self.BLOCK_ELEMENT_PATTERN.match(line.strip())
        if not match:
            return None

        text, page, x_min, y_min, x_max, y_max, confidence = match.groups()

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
            "confidence": float(confidence) if confidence else None,
        }

    def _parse_layout_element(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line with layout metadata (legacy flat format).

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
        self,
        elements: List[Dict[str, Any]],
        chunk_index: int,
        layout_bbox: Optional[BoundingBox] = None,
        layout_confidence: Optional[float] = None,
        layout_type_override: Optional[str] = None,
    ) -> LayoutChunk:
        """
        Create a LayoutChunk from a group of elements.

        Args:
            elements: List of layout elements
            chunk_index: Index of this chunk
            layout_bbox: Optional bbox of the containing layout block
            layout_confidence: Optional confidence of the layout detection
            layout_type_override: Optional layout type from block header

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

        # Determine layout type
        if layout_type_override:
            dominant_type = layout_type_override
        else:
            # Determine dominant layout type (most common) from flat format
            layout_types = [elem.get("layout_type", "text") for elem in elements]
            dominant_type = max(set(layout_types), key=layout_types.count)

        # Calculate average confidence if available
        confidences = [
            elem["confidence"] for elem in elements if elem.get("confidence") is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # Estimate chunk size (word count)
        chunk_size = len(combined_text.split())

        # Map layout type string to LayoutType enum
        valid_layout_values = {lt.value for lt in LayoutType}
        if dominant_type in valid_layout_values:
            layout_type_enum = LayoutType(dominant_type)
        else:
            layout_type_enum = LayoutType.UNKNOWN

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
                layout_type=layout_type_enum,
                ocr_confidence=avg_confidence,
                layout_bbox=layout_bbox,
                layout_confidence=layout_confidence,
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

    def _parse_block_format(
        self, lines: List[str]
    ) -> List[Tuple[str, BoundingBox, Optional[float], List[Dict[str, Any]]]]:
        """
        Parse text in legacy block format with [LAYOUT:...][/LAYOUT] blocks.

        Args:
            lines: List of text lines

        Returns:
            List of tuples (layout_type, layout_bbox, layout_confidence, elements)
        """
        blocks = []
        current_block = None
        current_elements = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("Page "):
                continue

            # Check for layout block opening
            block_open = self._parse_layout_block_open(stripped)
            if block_open:
                # Start new block
                if current_block is not None and current_elements:
                    # Save previous block
                    blocks.append((*current_block, current_elements))
                current_block = block_open
                current_elements = []
                continue

            # Check for layout block closing
            if self.LAYOUT_BLOCK_CLOSE_PATTERN.match(stripped):
                if current_block is not None:
                    blocks.append((*current_block, current_elements))
                current_block = None
                current_elements = []
                continue

            # If inside a block, parse element
            if current_block is not None:
                parsed = self._parse_block_element(stripped)
                if parsed:
                    # Add layout_type from block header for consistency
                    parsed["layout_type"] = current_block[0]
                    current_elements.append(parsed)

        # Handle unclosed block
        if current_block is not None and current_elements:
            blocks.append((*current_block, current_elements))

        return blocks

    async def read(self):
        """
        Read and chunk text with layout awareness.

        Yields:
            LayoutChunk or DocumentChunk objects
        """
        # Collect all text first for format detection
        full_text = ""
        async for content_text in self.get_text():
            full_text += content_text

        text_preview = full_text[:2000]  # Sample first 2000 chars

        # Detect format
        detected_format = self._detect_format(text_preview)

        if detected_format == "plain":
            # No layout metadata, use regular paragraph chunking
            logger.info("No layout metadata detected, using regular chunking")
            async for chunk in self._fallback_to_regular_chunking():
                yield chunk
            return

        if detected_format == "json":
            # Parse JSON format
            logger.info("JSON format detected, using structured OCR parsing")
            plain_text, blocks = self._parse_json_format(full_text)

            if not blocks:
                logger.warning("No valid layout blocks in JSON, falling back to regular chunking")
                # Use plain_text from JSON for regular chunking
                if plain_text:
                    # Temporarily override get_text to use plain_text
                    async for chunk in self._chunk_plain_text(plain_text):
                        yield chunk
                else:
                    async for chunk in self._fallback_to_regular_chunking():
                        yield chunk
                return

            chunk_index = 0
            for layout_type, layout_bbox, layout_confidence, elements in blocks:
                if not elements:
                    continue

                # Group elements within this block by chunk size
                element_groups = self._group_elements_by_chunks(elements)

                for group in element_groups:
                    yield self._create_layout_chunk(
                        group,
                        chunk_index,
                        layout_bbox=layout_bbox,
                        layout_confidence=layout_confidence,
                        layout_type_override=layout_type,
                    )
                    chunk_index += 1
            return

        # Handle legacy formats
        lines = full_text.split("\n")

        if detected_format == "legacy_block":
            # Parse legacy block format
            logger.info("Legacy block format detected, using layout-block-aware chunking")
            blocks = self._parse_block_format(lines)

            if not blocks:
                logger.warning("No valid layout blocks parsed, falling back to regular chunking")
                async for chunk in self._fallback_to_regular_chunking():
                    yield chunk
                return

            chunk_index = 0
            for layout_type, layout_bbox, layout_confidence, elements in blocks:
                if not elements:
                    continue

                # Group elements within this block by chunk size
                element_groups = self._group_elements_by_chunks(elements)

                for group in element_groups:
                    yield self._create_layout_chunk(
                        group,
                        chunk_index,
                        layout_bbox=layout_bbox,
                        layout_confidence=layout_confidence,
                        layout_type_override=layout_type,
                    )
                    chunk_index += 1

        else:  # legacy_flat
            # Parse legacy flat format (backward compat)
            logger.info("Legacy flat format detected, using layout-aware chunking")

            elements = []
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

    async def _chunk_plain_text(self, plain_text: str):
        """Chunk plain text extracted from JSON format."""
        paragraph_chunks = []
        chunk_index = 0
        chunk_size = 0

        for chunk_data in chunk_by_paragraph(
            plain_text,
            self.max_chunk_size,
            batch_paragraphs=True,
        ):
            if chunk_size + chunk_data["chunk_size"] <= self.max_chunk_size:
                paragraph_chunks.append(chunk_data)
                chunk_size += chunk_data["chunk_size"]
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
                    chunk_size = 0
                else:
                    chunk_text = " ".join(chunk["text"] for chunk in paragraph_chunks)
                    yield DocumentChunk(
                        id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{chunk_index}"),
                        text=chunk_text,
                        chunk_size=chunk_size,
                        is_part_of=self.document,
                        chunk_index=chunk_index,
                        cut_type=paragraph_chunks[-1]["cut_type"],
                        contains=[],
                        metadata={"index_fields": ["text"]},
                    )
                    paragraph_chunks = [chunk_data]
                    chunk_size = chunk_data["chunk_size"]

                chunk_index += 1

        if len(paragraph_chunks) > 0:
            yield DocumentChunk(
                id=uuid5(NAMESPACE_OID, f"{str(self.document.id)}-{chunk_index}"),
                text=" ".join(chunk["text"] for chunk in paragraph_chunks),
                chunk_size=chunk_size,
                is_part_of=self.document,
                chunk_index=chunk_index,
                cut_type=paragraph_chunks[-1]["cut_type"],
                contains=[],
                metadata={"index_fields": ["text"]},
            )

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
