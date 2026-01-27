"""PDF loader using pdfplumber for digital PDFs with layout extraction."""

from typing import List
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata

logger = get_logger(__name__)


class PdfPlumberLoader(LoaderInterface):
    """
    PDF loader using pdfplumber for digital PDFs.

    Extracts text with word-level or line-level bounding boxes from PDFs
    with native text. Detects layout elements like tables, headers, footers.
    Much faster than OCR as it uses native PDF parsing.

    Output format: Text with inline bbox metadata for layout-aware chunking.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    @property
    def loader_name(self) -> str:
        return "pdfplumber_loader"

    def can_handle(self, extension: str, mime_type: str) -> bool:
        """Check if file can be handled by this loader."""
        if extension in self.supported_extensions and mime_type in self.supported_mime_types:
            return True
        return False

    def _normalize_bbox(self, bbox: tuple, page_width: float, page_height: float) -> tuple:
        """
        Normalize bbox coordinates to 0-1 range.

        Args:
            bbox: (x0, y0, x1, y1) in PDF coordinates
            page_width: Page width
            page_height: Page height

        Returns:
            Normalized (x_min, y_min, x_max, y_max) in 0-1 range
        """
        x0, y0, x1, y1 = bbox
        return (
            x0 / page_width if page_width > 0 else 0,
            y0 / page_height if page_height > 0 else 0,
            x1 / page_width if page_width > 0 else 1,
            y1 / page_height if page_height > 0 else 1,
        )

    def _detect_layout_type(
        self, bbox: tuple, page_height: float, is_in_table: bool = False
    ) -> str:
        """
        Detect layout type based on position heuristics.

        Args:
            bbox: Bounding box (x0, y0, x1, y1)
            page_height: Page height
            is_in_table: Whether element is part of a table

        Returns:
            Layout type string
        """
        if is_in_table:
            return "table"

        _, y0, _, y1 = bbox
        # Header: top 10% of page
        if y1 < page_height * 0.1:
            return "header"
        # Footer: bottom 10% of page
        elif y0 > page_height * 0.9:
            return "footer"
        else:
            return "text"

    def _extract_words_with_layout(self, page, page_num: int) -> List[str]:
        """
        Extract words with bounding boxes and layout type.

        Args:
            page: pdfplumber page object
            page_num: Page number

        Returns:
            List of formatted text lines with bbox metadata
        """
        formatted_lines = []
        page_width = page.width
        page_height = page.height

        # Extract words with bounding boxes
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        )

        # Find tables on this page
        tables = page.find_tables()
        table_bboxes = [table.bbox for table in tables] if tables else []

        # Group words into lines based on y-coordinate
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 3

        for word in words:
            word_y = (word["top"] + word["bottom"]) / 2

            if current_y is None or abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
                if current_y is None:
                    current_y = word_y
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                current_y = word_y

        if current_line:
            lines.append(current_line)

        # Format each line with bbox metadata
        for line_words in lines:
            if not line_words:
                continue

            # Combine words in line
            line_text = " ".join(w["text"] for w in line_words)

            # Calculate line bbox (encompassing all words)
            line_x0 = min(w["x0"] for w in line_words)
            line_y0 = min(w["top"] for w in line_words)
            line_x1 = max(w["x1"] for w in line_words)
            line_y1 = max(w["bottom"] for w in line_words)
            line_bbox = (line_x0, line_y0, line_x1, line_y1)

            # Check if line is in a table
            is_in_table = any(
                self._bbox_intersects(line_bbox, table_bbox) for table_bbox in table_bboxes
            )

            # Detect layout type
            layout_type = self._detect_layout_type(line_bbox, page_height, is_in_table)

            # Normalize bbox
            norm_bbox = self._normalize_bbox(line_bbox, page_width, page_height)

            # Format: "text [page=N, bbox=(x,y,x,y), type=TYPE]"
            formatted_lines.append(
                f"{line_text} [page={page_num}, "
                f"bbox=({norm_bbox[0]:.3f},{norm_bbox[1]:.3f},"
                f"{norm_bbox[2]:.3f},{norm_bbox[3]:.3f}), "
                f"type={layout_type}]"
            )

        return formatted_lines

    def _bbox_intersects(self, bbox1: tuple, bbox2: tuple) -> bool:
        """Check if two bounding boxes intersect."""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)

    async def load(
        self,
        file_path: str,
        extract_word_level: bool = False,
        **kwargs,
    ) -> str:
        """
        Load PDF file and extract text with layout metadata.

        Args:
            file_path: Path to the PDF file
            extract_word_level: Extract word-level bboxes (default: line-level)
            **kwargs: Additional arguments

        Returns:
            Path to stored file with layout-aware text

        Raises:
            ImportError: If pdfplumber is not installed
            Exception: If PDF processing fails
        """
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError(
                "pdfplumber is required for PDF processing. "
                "Install with: pip install cognee[ocr]"
            ) from e

        try:
            with open(file_path, "rb") as file:
                file_metadata = await get_file_metadata(file)
                # Name file based on hash with layout prefix
                storage_file_name = "layout_text_" + file_metadata["content_hash"] + ".txt"

                logger.info(f"Reading PDF with pdfplumber: {file_path}")

                with pdfplumber.open(file_path) as pdf:
                    content_parts = []

                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            content_parts.append(f"Page {page_num}:")

                            # Extract words/lines with layout
                            formatted_lines = self._extract_words_with_layout(page, page_num)

                            if formatted_lines:
                                content_parts.extend(formatted_lines)
                                content_parts.append("")  # Blank line between pages
                            else:
                                logger.warning(f"No text extracted from page {page_num}")

                        except Exception as e:
                            logger.warning(
                                f"Failed to extract layout from page {page_num}: {e}"
                            )
                            continue

                # Combine all content
                full_content = "\n".join(content_parts)

                if not full_content.strip():
                    logger.warning(
                        f"No content extracted from {file_path}, "
                        "may need OCR or different loader"
                    )

                storage_config = get_storage_config()
                data_root_directory = storage_config["data_root_directory"]
                storage = get_file_storage(data_root_directory)

                full_file_path = await storage.store(storage_file_name, full_content)

                logger.info(
                    f"Successfully processed PDF with pdfplumber: "
                    f"{file_path} -> {full_file_path}"
                )
                return full_file_path

        except Exception as e:
            logger.error(f"Failed to process PDF {file_path} with pdfplumber: {e}")
            raise Exception(f"pdfplumber PDF processing failed: {e}") from e
