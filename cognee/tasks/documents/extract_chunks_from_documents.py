from uuid import UUID
from sqlalchemy import select
from typing import AsyncGenerator

from cognee.shared.logging_utils import get_logger
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.data.models import Data
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.chunking.Chunker import Chunker
from cognee.tasks.documents.exceptions import InvalidChunkSizeError, InvalidChunkerError


async def update_document_token_count(document_id: UUID, token_count: int) -> None:
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        document_data_point = (
            await session.execute(select(Data).filter(Data.id == document_id))
        ).scalar_one_or_none()

        if document_data_point:
            document_data_point.token_count = token_count
            await session.merge(document_data_point)
            await session.commit()
        else:
            raise ValueError(f"Document with id {document_id} not found.")


logger = get_logger(__name__)


async def _detect_ocr_metadata(document: Document) -> bool:
    """
    Detect if document contains OCR/layout metadata.

    Supports detection of:
    1. JSON format: {"cognee_ocr_format": "1.0", ...}
    2. Legacy block format: [LAYOUT:...][/LAYOUT]
    3. Legacy flat format: [page=N, bbox=...]

    Args:
        document: Document to check

    Returns:
        True if document has OCR metadata
    """
    try:
        from cognee.infrastructure.files.utils.open_data_file import open_data_file

        # Read first 2000 characters to check for metadata
        async with open_data_file(document.raw_data_location, mode="r", encoding="utf-8") as file:
            text_preview = file.read(2000)

            # Check for JSON format (new structured format)
            stripped = text_preview.strip()
            if stripped.startswith("{") and '"cognee_ocr_format"' in text_preview:
                return True

            # Check for legacy formats
            has_page = "[page=" in text_preview
            has_bbox = "bbox=" in text_preview

            # Also check for legacy block format markers
            has_layout_block = "[LAYOUT:" in text_preview and "[/LAYOUT]" in text_preview

            return (has_page and has_bbox) or has_layout_block

    except Exception as e:
        logger.debug(f"Could not detect OCR metadata: {e}")
        return False


async def extract_chunks_from_documents(
    documents: list[Document],
    max_chunk_size: int,
    chunker: Chunker = TextChunker,
    auto_detect_layout: bool = True,
) -> AsyncGenerator:
    """
    Extracts chunks of data from a list of documents based on the specified chunking parameters.

    Automatically detects OCR/layout metadata and uses LayoutTextChunker when present
    (unless auto_detect_layout=False).

    Supports three OCR metadata formats:
    1. JSON format (new): {"cognee_ocr_format": "1.0", ...}
    2. Legacy block format: [LAYOUT:...][/LAYOUT]
    3. Legacy flat format: [page=N, bbox=...]

    Args:
        documents: List of documents to chunk
        max_chunk_size: Maximum chunk size in tokens
        chunker: Chunker class to use (default: TextChunker)
        auto_detect_layout: Automatically use LayoutTextChunker if OCR metadata detected

    Yields:
        Document chunks

    Notes:
        - The `read` method of the `Document` class must be implemented to support the chunking operation.
        - The `chunker` parameter determines the chunking logic and should align with the document type.
        - If OCR metadata is detected and auto_detect_layout=True, LayoutTextChunker will be used
          regardless of the chunker parameter.
    """
    if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
        raise InvalidChunkSizeError(max_chunk_size)
    if not isinstance(chunker, type):
        raise InvalidChunkerError()
    if not hasattr(chunker, "read"):
        raise InvalidChunkerError()

    for document in documents:
        document_token_count = 0

        # Detect OCR metadata and select appropriate chunker
        selected_chunker = chunker
        if auto_detect_layout:
            has_ocr = await _detect_ocr_metadata(document)
            if has_ocr:
                logger.info(
                    f"OCR metadata detected in document {document.id}, using LayoutTextChunker"
                )
                try:
                    from cognee.modules.chunking.LayoutTextChunker import (
                        LayoutTextChunker,
                    )

                    selected_chunker = LayoutTextChunker
                except ImportError:
                    logger.warning("LayoutTextChunker not available, using default chunker")

        async for document_chunk in document.read(
            max_chunk_size=max_chunk_size, chunker_cls=selected_chunker
        ):
            document_token_count += document_chunk.chunk_size
            document_chunk.belongs_to_set = document.belongs_to_set
            yield document_chunk

        await update_document_token_count(document.id, document_token_count)
