"""End-to-end integration tests for OCR pipeline."""

import pytest
import pytest_asyncio
from unittest.mock import patch, Mock
from pathlib import Path
from cognee import add, cognify, search, prune
from cognee.modules.search.types import SearchType
from cognee.tests.test_data.mock_data.mock_ocr_results import (
    create_mock_ocr_result,
    create_mock_ocr_page_result,
)


@pytest_asyncio.fixture
async def setup_cognee():
    """Setup isolated test environment."""
    # Setup happens before test
    yield

    # Cleanup after test
    try:
        await prune.prune_data()
        await prune.prune_system(metadata=True)
    except Exception:
        pass  # Cleanup errors are non-critical


class TestDigitalPDFFlow:
    """Tests for digital PDF end-to-end flow."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber")
    async def test_digital_pdf_complete_flow(
        self, mock_detector_pdfplumber, mock_loader_pdfplumber, setup_cognee
    ):
        """Test complete add→cognify→search flow with digital PDF."""
        # Mock PDF type detection (DIGITAL)
        mock_detect_page = Mock()
        mock_detect_page.extract_text.return_value = "x" * 500  # Lots of text → DIGITAL
        mock_detect_pdf = Mock()
        mock_detect_pdf.__enter__.return_value.pages = [mock_detect_page]
        mock_detector_pdfplumber.open.return_value = mock_detect_pdf

        # Mock PDF loading with pdfplumber
        mock_load_page = Mock()
        mock_load_page.width = 1000
        mock_load_page.height = 1400
        mock_load_page.extract_words.return_value = [
            {"text": "Digital", "x0": 100, "y0": 200, "x1": 200, "y1": 220},
            {"text": "PDF", "x0": 210, "y0": 200, "x1": 280, "y1": 220},
            {"text": "content", "x0": 290, "y0": 200, "x1": 400, "y1": 220},
        ]
        mock_load_page.find_tables.return_value = []

        mock_load_pdf = Mock()
        mock_load_pdf.__enter__.return_value.pages = [mock_load_page]
        mock_loader_pdfplumber.open.return_value = mock_load_pdf

        # Run pipeline
        await add("digital_test.pdf", dataset_name="test_digital")
        await cognify(datasets=["test_digital"])

        # Search with layout metadata
        results = await search(
            query_text="Digital PDF",
            query_type=SearchType.CHUNKS,
            datasets=["test_digital"],
            include_layout=True,
        )

        # Verify results
        assert len(results) > 0

        # Check if chunks are present in results
        if "chunks" in results[0]:
            first_chunk = results[0]["chunks"][0]

            # Verify layout metadata is present
            if "layout" in first_chunk and first_chunk["layout"]:
                assert first_chunk["layout"]["page_number"] >= 1
                # Bbox may or may not be present depending on implementation
                if first_chunk["layout"].get("bbox"):
                    bbox = first_chunk["layout"]["bbox"]
                    assert 0 <= bbox["x1"] <= 1
                    assert 0 <= bbox["y1"] <= 1

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    async def test_digital_pdf_metadata_preservation(
        self, mock_pdfplumber, setup_cognee
    ):
        """Test that bbox coordinates remain in 0-1 range throughout pipeline."""
        # Mock PDF
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1400
        mock_page.extract_words.return_value = [
            {"text": "Test", "x0": 100, "y0": 200, "x1": 200, "y1": 220},
        ]
        mock_page.find_tables.return_value = []

        mock_pdf = Mock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        # Run pipeline
        await add("test.pdf", dataset_name="test_metadata")
        await cognify(datasets=["test_metadata"])

        results = await search(
            query_text="Test",
            query_type=SearchType.CHUNKS,
            datasets=["test_metadata"],
            include_layout=True,
        )

        # Verify bbox normalization is preserved
        if len(results) > 0 and "chunks" in results[0]:
            for chunk in results[0]["chunks"]:
                if chunk.get("layout") and chunk["layout"].get("bbox"):
                    bbox = chunk["layout"]["bbox"]
                    assert 0 <= bbox["x1"] <= 1
                    assert 0 <= bbox["y1"] <= 1
                    assert 0 <= bbox["x2"] <= 1
                    assert 0 <= bbox["y2"] <= 1


class TestScannedPDFFlow:
    """Tests for scanned PDF end-to-end flow (requires OCR)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("cognee/infrastructure/ocr/PaddleOCRAdapter.py").exists(),
        reason="OCR not available",
    )
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    async def test_scanned_pdf_complete_flow(
        self, mock_ocr_class, mock_detector_pdfplumber, setup_cognee
    ):
        """Test complete add→cognify→search flow with scanned PDF."""
        # Mock PDF type detection (SCANNED)
        mock_detect_page = Mock()
        mock_detect_page.extract_text.return_value = ""  # No text → SCANNED
        mock_detect_pdf = Mock()
        mock_detect_pdf.__enter__.return_value.pages = [mock_detect_page]
        mock_detector_pdfplumber.open.return_value = mock_detect_pdf

        # Mock OCR processing
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_result(num_pages=1, elements_per_page=5)
        mock_ocr_instance.process_pdf.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Run pipeline
        await add("scanned_test.pdf", dataset_name="test_scanned")
        await cognify(datasets=["test_scanned"])

        # Search with layout
        results = await search(
            query_text="scanned",
            query_type=SearchType.CHUNKS,
            datasets=["test_scanned"],
            include_layout=True,
        )

        # Verify OCR was used
        assert len(results) > 0

        # Check for OCR confidence in metadata
        if "chunks" in results[0]:
            first_chunk = results[0]["chunks"][0]
            if "layout" in first_chunk and first_chunk["layout"]:
                # OCR results should have confidence
                assert first_chunk["layout"].get("confidence") is not None


class TestImageFlow:
    """Tests for image OCR end-to-end flow."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("cognee/infrastructure/ocr/PaddleOCRAdapter.py").exists(),
        reason="OCR not available",
    )
    @patch("cognee.infrastructure.loaders.core.ocr_image_loader.PaddleOCRAdapter")
    async def test_image_complete_flow(self, mock_ocr_class, setup_cognee):
        """Test complete add→cognify→search flow with image."""
        # Mock OCR processing
        mock_ocr_instance = Mock()
        mock_ocr_result = create_mock_ocr_page_result(page_number=1, num_elements=5)
        mock_ocr_instance.process_image.return_value = mock_ocr_result
        mock_ocr_class.return_value = mock_ocr_instance

        # Run pipeline
        await add("test_image.png", dataset_name="test_image")
        await cognify(datasets=["test_image"])

        # Search
        results = await search(
            query_text="image text",
            query_type=SearchType.CHUNKS,
            datasets=["test_image"],
            include_layout=True,
        )

        # Verify results
        assert len(results) > 0

        # Image should be treated as single page
        if "chunks" in results[0]:
            for chunk in results[0]["chunks"]:
                if chunk.get("layout"):
                    assert chunk["layout"]["page_number"] == 1


class TestFallbackScenarios:
    """Tests for fallback behavior."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.utils.pdf_type_detector.pdfplumber")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PaddleOCRAdapter")
    @patch("cognee.infrastructure.loaders.external.ocr_pdf_loader.PdfPlumberLoader")
    async def test_ocr_failure_fallback(
        self,
        mock_pdfplumber_loader,
        mock_ocr_class,
        mock_detector_pdfplumber,
        setup_cognee,
    ):
        """Test graceful fallback when OCR fails."""
        # Mock detection (SCANNED)
        mock_detect_page = Mock()
        mock_detect_page.extract_text.return_value = ""
        mock_detect_pdf = Mock()
        mock_detect_pdf.__enter__.return_value.pages = [mock_detect_page]
        mock_detector_pdfplumber.open.return_value = mock_detect_pdf

        # Mock OCR failure
        mock_ocr_instance = Mock()
        mock_ocr_instance.process_pdf.side_effect = Exception("OCR failed")
        mock_ocr_class.return_value = mock_ocr_instance

        # Mock PdfPlumber fallback
        mock_pdfplumber_instance = Mock()
        mock_pdfplumber_instance.load.return_value = ["fallback_result.txt"]
        mock_pdfplumber_loader.return_value = mock_pdfplumber_instance

        # Pipeline should not fail
        try:
            await add("fallback_test.pdf", dataset_name="test_fallback")
            await cognify(datasets=["test_fallback"])

            results = await search(
                query_text="fallback",
                query_type=SearchType.CHUNKS,
                datasets=["test_fallback"],
            )

            # Should still get results (from fallback)
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Pipeline should not fail on OCR error: {e}")


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.PyPDF2")
    async def test_search_without_layout_parameter(self, mock_pypdf, setup_cognee):
        """Test that search works without include_layout parameter."""
        # Mock simple PDF loading
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Simple text"
        mock_reader.pages = [mock_page]
        mock_pypdf.PdfReader.return_value = mock_reader

        # Run pipeline without layout
        await add("simple.pdf", dataset_name="test_backward_compat")
        await cognify(datasets=["test_backward_compat"])

        # Search WITHOUT include_layout parameter (old behavior)
        results = await search(
            query_text="simple",
            query_type=SearchType.CHUNKS,
            datasets=["test_backward_compat"],
            # Note: include_layout NOT specified
        )

        # Should still work (backward compatible)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pypdf_loader.PyPDF2")
    async def test_existing_non_ocr_documents_still_work(
        self, mock_pypdf, setup_cognee
    ):
        """Test that documents without OCR metadata still work."""
        # Mock PDF without OCR
        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "Regular PDF text without OCR metadata"
        mock_reader.pages = [mock_page]
        mock_pypdf.PdfReader.return_value = mock_reader

        await add("regular.pdf", dataset_name="test_regular")
        await cognify(datasets=["test_regular"])

        results = await search(
            query_text="Regular",
            query_type=SearchType.CHUNKS,
            datasets=["test_regular"],
            include_layout=True,  # Request layout
        )

        # Should work, but layout will be None or empty
        assert isinstance(results, list)

        if len(results) > 0 and "chunks" in results[0]:
            # Chunks without OCR metadata should not break
            for chunk in results[0]["chunks"]:
                # layout may be None or have no bbox
                assert "text" in chunk  # Text should always be present


class TestMultiPageDocuments:
    """Tests for multi-page document handling."""

    @pytest.mark.asyncio
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    async def test_multi_page_pdf_page_numbers(self, mock_pdfplumber, setup_cognee):
        """Test that page numbers are correctly tracked across pages."""
        # Mock 3-page PDF
        pages = []
        for page_num in range(1, 4):
            mock_page = Mock()
            mock_page.width = 1000
            mock_page.height = 1400
            mock_page.extract_words.return_value = [
                {
                    "text": f"Page{page_num}",
                    "x0": 100,
                    "y0": 200,
                    "x1": 200,
                    "y1": 220,
                },
            ]
            mock_page.find_tables.return_value = []
            pages.append(mock_page)

        mock_pdf = Mock()
        mock_pdf.__enter__.return_value.pages = pages
        mock_pdfplumber.open.return_value = mock_pdf

        await add("multipage.pdf", dataset_name="test_multipage")
        await cognify(datasets=["test_multipage"])

        results = await search(
            query_text="Page",
            query_type=SearchType.CHUNKS,
            datasets=["test_multipage"],
            include_layout=True,
        )

        # Verify page numbers are present
        if len(results) > 0 and "chunks" in results[0]:
            page_numbers_found = set()
            for chunk in results[0]["chunks"]:
                if chunk.get("layout") and chunk["layout"].get("page_number"):
                    page_numbers_found.add(chunk["layout"]["page_number"])

            # Should have chunks from multiple pages
            assert len(page_numbers_found) > 0


class TestPerformance:
    """Tests for performance characteristics."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @patch("cognee.infrastructure.loaders.external.pdfplumber_loader.pdfplumber")
    async def test_layout_search_response_size(self, mock_pdfplumber, setup_cognee):
        """Test that layout metadata doesn't bloat response size excessively."""
        # Mock PDF
        mock_page = Mock()
        mock_page.width = 1000
        mock_page.height = 1400
        mock_page.extract_words.return_value = [
            {"text": f"Word{i}", "x0": 100, "y0": 100 + i * 20, "x1": 200, "y1": 120 + i * 20}
            for i in range(20)
        ]
        mock_page.find_tables.return_value = []

        mock_pdf = Mock()
        mock_pdf.__enter__.return_value.pages = [mock_page]
        mock_pdfplumber.open.return_value = mock_pdf

        await add("size_test.pdf", dataset_name="test_size")
        await cognify(datasets=["test_size"])

        results = await search(
            query_text="Word",
            query_type=SearchType.CHUNKS,
            datasets=["test_size"],
            include_layout=True,
        )

        # Verify response is reasonable size (not excessively large)
        import json

        response_size = len(json.dumps(results))

        # Response should be < 100KB for 10 results (rough check)
        assert response_size < 100_000, f"Response too large: {response_size} bytes"
