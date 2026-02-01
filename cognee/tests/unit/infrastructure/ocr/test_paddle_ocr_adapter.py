"""Unit tests for PaddleOCRAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, patch as mock_patch
from pathlib import Path
from cognee.shared.data_models import BoundingBox
from cognee.infrastructure.ocr.PaddleOCRAdapter import (
    OCRTextElement,
    OCRPageResult,
    OCRDocumentResult,
    PaddleOCRAdapter,
)


class TestBoundingBoxNormalization:
    """Tests for bounding box normalization to 0-1 range."""

    @pytest.mark.parametrize(
        "page_width,page_height,pixel_coords,expected_normalized",
        [
            # Standard page
            (1000, 1400, (100, 200, 800, 900), (0.1, 0.143, 0.8, 0.643)),
            # Small page
            (600, 800, (60, 80, 540, 720), (0.1, 0.1, 0.9, 0.9)),
            # Large page
            (2400, 3200, (240, 320, 2160, 2880), (0.1, 0.1, 0.9, 0.9)),
            # Edge case: full page
            (1000, 1000, (0, 0, 1000, 1000), (0.0, 0.0, 1.0, 1.0)),
        ],
    )
    def test_normalize_bbox(self, page_width, page_height, pixel_coords, expected_normalized):
        """Test bbox normalization with various page dimensions."""
        adapter = PaddleOCRAdapter()
        x0, y0, x1, y1 = pixel_coords

        # PaddleOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        bbox_coords = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        bbox = adapter._normalize_bbox(
            bbox_coords=bbox_coords,
            page_width=page_width,
            page_height=page_height,
        )

        exp_x_min, exp_y_min, exp_x_max, exp_y_max = expected_normalized
        assert abs(bbox.x_min - exp_x_min) < 0.01
        assert abs(bbox.y_min - exp_y_min) < 0.01
        assert abs(bbox.x_max - exp_x_max) < 0.01
        assert abs(bbox.y_max - exp_y_max) < 0.01

    def test_normalized_bbox_bounds(self):
        """Test that normalized coordinates are always in 0-1 range."""
        adapter = PaddleOCRAdapter()

        # PaddleOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        bbox_coords = [[50, 100], [950, 100], [950, 1300], [50, 1300]]

        bbox = adapter._normalize_bbox(
            bbox_coords=bbox_coords,
            page_width=1000,
            page_height=1400,
        )

        assert 0 <= bbox.x_min <= 1
        assert 0 <= bbox.y_min <= 1
        assert 0 <= bbox.x_max <= 1
        assert 0 <= bbox.y_max <= 1
        assert bbox.x_min < bbox.x_max
        assert bbox.y_min < bbox.y_max

    def test_bbox_preserves_pixel_coords(self):
        """Test that normalized bbox preserves original pixel coordinates."""
        adapter = PaddleOCRAdapter()

        # PaddleOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        bbox_coords = [[100, 200], [800, 200], [800, 900], [100, 900]]

        bbox = adapter._normalize_bbox(
            bbox_coords=bbox_coords,
            page_width=1000,
            page_height=1400,
        )

        assert bbox.pixel_x_min == 100
        assert bbox.pixel_y_min == 200
        assert bbox.pixel_x_max == 800
        assert bbox.pixel_y_max == 900


class TestImageProcessing:
    """Tests for image OCR processing."""

    @pytest.mark.asyncio
    @patch("PIL.Image")
    async def test_process_image_success(self, mock_image_class):
        """Test successful image OCR processing."""
        # Mock image
        mock_image = MagicMock()
        mock_image.size = (1000, 1400)
        mock_image_class.open.return_value.__enter__.return_value = mock_image

        # Mock OCR results
        mock_ocr_result = [
            [
                [[100, 200], [800, 200], [800, 280], [100, 280]],  # Bbox corners
                ("Sample text", 0.95),  # (text, confidence)
            ],
            [
                [[100, 300], [800, 300], [800, 380], [100, 380]],
                ("More text", 0.88),
            ],
        ]

        adapter = PaddleOCRAdapter(min_confidence=0.8)
        adapter._ocr_engine = Mock()
        adapter._ocr_engine.ocr.return_value = [mock_ocr_result]

        result = await adapter.process_image("test.png", page_number=1)

        assert isinstance(result, OCRPageResult)
        assert result.page_number == 1
        assert result.page_width == 1000
        assert result.page_height == 1400
        assert len(result.elements) == 2
        assert result.elements[0].text == "Sample text"
        assert result.elements[0].confidence == 0.95
        assert result.elements[1].text == "More text"
        assert result.elements[1].confidence == 0.88

    @pytest.mark.asyncio
    @patch("PIL.Image")
    async def test_process_image_confidence_filtering(self, mock_image_class):
        """Test that low-confidence results are filtered out."""
        mock_image = MagicMock()
        mock_image.size = (1000, 1000)
        mock_image_class.open.return_value.__enter__.return_value = mock_image

        # Mix of high and low confidence results
        mock_ocr_result = [
            [
                [[100, 100], [200, 100], [200, 150], [100, 150]],
                ("Good text", 0.95),  # Above threshold
            ],
            [
                [[100, 200], [200, 200], [200, 250], [100, 250]],
                ("Bad text", 0.50),  # Below threshold
            ],
            [
                [[100, 300], [200, 300], [200, 350], [100, 350]],
                ("OK text", 0.80),  # Exactly at threshold
            ],
        ]

        adapter = PaddleOCRAdapter(min_confidence=0.8)
        adapter._ocr_engine = Mock()
        adapter._ocr_engine.ocr.return_value = [mock_ocr_result]

        result = await adapter.process_image("test.png", page_number=1)

        # Should only have 2 elements (>= 0.8 confidence)
        assert len(result.elements) == 2
        assert result.elements[0].text == "Good text"
        assert result.elements[1].text == "OK text"

    @pytest.mark.asyncio
    @patch("PIL.Image")
    async def test_process_image_empty_result(self, mock_image_class):
        """Test handling of empty OCR results."""
        mock_image = MagicMock()
        mock_image.size = (1000, 1000)
        mock_image_class.open.return_value.__enter__.return_value = mock_image

        adapter = PaddleOCRAdapter()
        adapter._ocr_engine = Mock()
        adapter._ocr_engine.ocr.return_value = [[]]  # Empty result

        result = await adapter.process_image("test.png", page_number=1)

        assert isinstance(result, OCRPageResult)
        assert len(result.elements) == 0

    @pytest.mark.asyncio
    @patch("PIL.Image")
    async def test_process_image_error_handling(self, mock_image_class):
        """Test error handling when image processing fails."""
        mock_image_class.open.side_effect = Exception("Image load failed")

        adapter = PaddleOCRAdapter()

        with pytest.raises(Exception):
            await adapter.process_image("invalid.png", page_number=1)


class TestPDFProcessing:
    """Tests for PDF OCR processing."""

    @pytest.mark.asyncio
    @patch("PIL.Image")
    @patch("pdf2image.convert_from_path")
    async def test_process_pdf_success(self, mock_convert, mock_image_class):
        """Test successful PDF OCR processing."""
        # Mock PDF pages as images
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()

        mock_convert.return_value = [mock_page1, mock_page2]

        # Mock Image.open to return proper image with size for the temp files
        mock_image = MagicMock()
        mock_image.size = (1000, 1400)
        mock_image_class.open.return_value.__enter__.return_value = mock_image

        # Mock OCR results for each page
        mock_ocr_result_page1 = [
            [
                [[100, 200], [800, 200], [800, 280], [100, 280]],
                ("Page 1 text", 0.95),
            ],
        ]
        mock_ocr_result_page2 = [
            [
                [[100, 200], [800, 200], [800, 280], [100, 280]],
                ("Page 2 text", 0.90),
            ],
        ]

        adapter = PaddleOCRAdapter()
        adapter._ocr_engine = Mock()
        adapter._ocr_engine.ocr.side_effect = [[mock_ocr_result_page1], [mock_ocr_result_page2]]

        result = await adapter.process_pdf("test.pdf")

        assert isinstance(result, OCRDocumentResult)
        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[0].elements[0].text == "Page 1 text"
        assert result.pages[1].page_number == 2
        assert result.pages[1].elements[0].text == "Page 2 text"

    @pytest.mark.asyncio
    @patch("pdf2image.convert_from_path")
    async def test_process_pdf_empty(self, mock_convert):
        """Test handling of empty PDF."""
        mock_convert.return_value = []

        adapter = PaddleOCRAdapter()

        result = await adapter.process_pdf("empty.pdf")

        assert isinstance(result, OCRDocumentResult)
        assert result.total_pages == 0
        assert len(result.pages) == 0

    @pytest.mark.asyncio
    @patch("pdf2image.convert_from_path")
    async def test_process_pdf_error_handling(self, mock_convert):
        """Test error handling when PDF conversion fails."""
        mock_convert.side_effect = Exception("PDF conversion failed")

        adapter = PaddleOCRAdapter()

        with pytest.raises(Exception):
            await adapter.process_pdf("invalid.pdf")


class TestLazyInitialization:
    """Tests for lazy initialization of OCR engine."""

    def test_ocr_engine_not_initialized_on_creation(self):
        """Test that OCR engine is not initialized on adapter creation."""
        adapter = PaddleOCRAdapter()

        assert adapter._ocr_engine is None

    @pytest.mark.asyncio
    @patch("paddleocr.PaddleOCR")
    @patch("paddleocr.__version__", "3.0.0")
    async def test_ocr_engine_initialized_on_first_use(self, mock_paddle_ocr):
        """Test that OCR engine is initialized on first _get_ocr_engine call."""
        mock_engine = Mock()
        mock_paddle_ocr.return_value = mock_engine

        adapter = PaddleOCRAdapter(lang="en", use_gpu=False)

        engine = adapter._get_ocr_engine()

        assert engine == mock_engine
        assert adapter._ocr_engine == mock_engine
        # PaddleOCR 3.x uses device parameter instead of use_gpu
        mock_paddle_ocr.assert_called_once_with(
            lang="en",
            device="cpu",
        )

    @pytest.mark.asyncio
    @patch("paddleocr.PaddleOCR")
    async def test_ocr_engine_reused_on_subsequent_calls(self, mock_paddle_ocr):
        """Test that OCR engine is reused and not re-initialized."""
        mock_engine = Mock()
        mock_paddle_ocr.return_value = mock_engine

        adapter = PaddleOCRAdapter()

        engine1 = adapter._get_ocr_engine()
        engine2 = adapter._get_ocr_engine()

        assert engine1 == engine2
        mock_paddle_ocr.assert_called_once()  # Only called once

    @pytest.mark.asyncio
    async def test_import_error_handling(self):
        """Test handling when PaddleOCR is not installed."""
        import sys

        # Temporarily remove paddleocr from sys.modules to simulate it not being installed
        paddleocr_module = sys.modules.get("paddleocr")
        if "paddleocr" in sys.modules:
            del sys.modules["paddleocr"]

        try:
            with patch.dict("sys.modules", {"paddleocr": None}):
                adapter = PaddleOCRAdapter()

                with pytest.raises(ImportError, match="PaddleOCR"):
                    adapter._get_ocr_engine()
        finally:
            # Restore paddleocr module if it was present
            if paddleocr_module is not None:
                sys.modules["paddleocr"] = paddleocr_module


class TestOCRDataModels:
    """Tests for OCR data model structures."""

    def test_bounding_box_creation(self):
        """Test BoundingBox creation and properties."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
            pixel_x_min=100,
            pixel_y_min=200,
            pixel_x_max=800,
            pixel_y_max=900,
            confidence=0.95,
        )

        assert bbox.x_min == 0.1
        assert bbox.pixel_x_min == 100
        assert bbox.confidence == 0.95
        assert bbox.area == pytest.approx((0.8 - 0.1) * (0.9 - 0.2))

    def test_ocr_text_element_creation(self):
        """Test OCRTextElement creation."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
            pixel_x_min=100,
            pixel_y_min=200,
            pixel_x_max=800,
            pixel_y_max=900,
        )

        element = OCRTextElement(
            text="Sample text",
            bbox=bbox,
            confidence=0.95,
            page_number=1,
        )

        assert element.text == "Sample text"
        assert element.bbox == bbox
        assert element.confidence == 0.95
        assert element.page_number == 1

    def test_ocr_page_result_creation(self):
        """Test OCRPageResult creation."""
        bbox = BoundingBox(
            x_min=0.1,
            y_min=0.2,
            x_max=0.8,
            y_max=0.9,
            pixel_x_min=100,
            pixel_y_min=200,
            pixel_x_max=800,
            pixel_y_max=900,
        )
        element = OCRTextElement(
            text="Test",
            bbox=bbox,
            confidence=0.9,
            page_number=1,
        )

        page_result = OCRPageResult(
            page_number=1,
            elements=[element],
            page_width=1000,
            page_height=1400,
        )

        assert page_result.page_number == 1
        assert len(page_result.elements) == 1
        assert page_result.page_width == 1000
        assert page_result.page_height == 1400

    def test_ocr_document_result_creation(self):
        """Test OCRDocumentResult creation."""
        page1 = OCRPageResult(
            page_number=1,
            elements=[],
            page_width=1000,
            page_height=1400,
        )
        page2 = OCRPageResult(
            page_number=2,
            elements=[],
            page_width=1000,
            page_height=1400,
        )

        doc_result = OCRDocumentResult(
            pages=[page1, page2],
            total_pages=2,
        )

        assert doc_result.total_pages == 2
        assert len(doc_result.pages) == 2
        assert doc_result.pages[0].page_number == 1
        assert doc_result.pages[1].page_number == 2


class TestLayoutTypeMapping:
    """Tests for PPStructureV3 layout type mapping."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("text", "text"),
            ("title", "title"),
            ("paragraph title", "heading"),
            ("paragraph", "paragraph"),
            ("table", "table"),
            ("table caption", "caption"),
            ("figure", "figure"),
            ("figure caption", "caption"),
            ("image", "figure"),
            ("header", "header"),
            ("footer", "footer"),
            ("page number", "footer"),
            ("formula", "code"),
            ("formula number", "code"),
            ("reference", "text"),
            ("footnote", "text"),
            ("chart", "figure"),
            ("algorithm", "code"),
            ("seal", "figure"),
            ("list", "list"),
            ("unknown_label", "text"),  # Default fallback
        ],
    )
    def test_layout_label_mapping(self, label, expected):
        """Test that PPStructureV3 labels map correctly to layout types."""
        adapter = PaddleOCRAdapter()
        assert adapter._map_layout_label_to_type(label) == expected

    def test_case_insensitive_mapping(self):
        """Test that label mapping is case-insensitive."""
        adapter = PaddleOCRAdapter()
        assert adapter._map_layout_label_to_type("TEXT") == "text"
        assert adapter._map_layout_label_to_type("Title") == "title"
        assert adapter._map_layout_label_to_type("TABLE") == "table"


class TestBBoxOverlap:
    """Tests for bounding box overlap calculation."""

    def test_full_overlap(self):
        """Test IoU calculation for fully overlapping bboxes."""
        adapter = PaddleOCRAdapter()
        overlap = adapter._calculate_bbox_overlap(0, 0, 1, 1, 0, 0, 1, 1)
        assert overlap == 1.0

    def test_no_overlap(self):
        """Test IoU calculation for non-overlapping bboxes."""
        adapter = PaddleOCRAdapter()
        overlap = adapter._calculate_bbox_overlap(0, 0, 0.5, 0.5, 0.6, 0.6, 1, 1)
        assert overlap == 0.0

    def test_partial_overlap(self):
        """Test IoU calculation for partially overlapping bboxes."""
        adapter = PaddleOCRAdapter()
        overlap = adapter._calculate_bbox_overlap(0, 0, 0.6, 0.6, 0.4, 0.4, 1, 1)
        assert 0 < overlap < 1

    def test_touching_bboxes(self):
        """Test IoU for bboxes that touch at edges (no overlap)."""
        adapter = PaddleOCRAdapter()
        overlap = adapter._calculate_bbox_overlap(0, 0, 0.5, 0.5, 0.5, 0, 1, 0.5)
        assert overlap == 0.0

    def test_contained_bbox(self):
        """Test IoU when one bbox is completely inside another."""
        adapter = PaddleOCRAdapter()
        overlap = adapter._calculate_bbox_overlap(0.2, 0.2, 0.8, 0.8, 0, 0, 1, 1)
        # Intersection = 0.6*0.6 = 0.36
        # Union = 0.36 + (1.0 - 0.36) = 1.0
        # IoU = 0.36 / 1.0 = 0.36
        assert overlap == pytest.approx(0.36)


class TestPPStructureV3Init:
    """Tests for PPStructureV3 initialization."""

    def test_structure_engine_lazy_init(self):
        """Test that PPStructureV3 engine is not initialized until first use."""
        adapter = PaddleOCRAdapter(use_structure=True)
        assert adapter._structure_engine is None
        assert adapter.use_structure is True

    def test_structure_config_passed(self):
        """Test that structure_config is stored correctly."""
        config = {'use_table_recognition': True, 'use_formula_recognition': False}
        adapter = PaddleOCRAdapter(use_structure=True, structure_config=config)
        assert adapter.structure_config == config

    def test_default_structure_config(self):
        """Test that structure_config defaults to empty dict."""
        adapter = PaddleOCRAdapter(use_structure=True)
        assert adapter.structure_config == {}


class TestDualEngineRouting:
    """Tests for routing between PaddleOCR and PPStructureV3."""

    @pytest.mark.asyncio
    async def test_routes_to_paddleocr_by_default(self):
        """Test that process_image routes to PaddleOCR when use_structure=False."""
        adapter = PaddleOCRAdapter(use_structure=False)

        # Mock the PaddleOCR processing method
        with patch.object(
            adapter, '_process_image_with_paddleocr',
            new=AsyncMock(return_value=OCRPageResult(
                page_number=1,
                elements=[],
                page_width=1000,
                page_height=1000,
            ))
        ) as mock_paddleocr_process:
            result = await adapter.process_image("test.jpg", page_number=1)

            mock_paddleocr_process.assert_called_once_with("test.jpg", 1)
            assert result.page_number == 1

    @pytest.mark.asyncio
    async def test_routes_to_structure_when_enabled(self):
        """Test that process_image routes to PPStructureV3 when use_structure=True."""
        adapter = PaddleOCRAdapter(use_structure=True)

        # Mock the PPStructureV3 processing method
        with patch.object(
            adapter, '_process_image_with_structure',
            new=AsyncMock(return_value=OCRPageResult(
                page_number=1,
                elements=[],
                page_width=1000,
                page_height=1000,
                layout_info={'layout_boxes': []},
            ))
        ) as mock_structure_process:
            result = await adapter.process_image("test.jpg", page_number=1)

            mock_structure_process.assert_called_once_with("test.jpg", 1)
            assert result.page_number == 1
            assert result.layout_info is not None


class TestOCRToLayoutMatching:
    """Tests for matching OCR results to layout regions."""

    def test_find_layout_type_for_bbox_single_match(self):
        """Test finding layout type when bbox matches a single region."""
        adapter = PaddleOCRAdapter()

        ocr_bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.3)
        layout_boxes = [
            {'coordinate': [100, 100, 500, 300], 'label': 'title'},
        ]

        layout_type = adapter._find_layout_type_for_bbox(
            ocr_bbox, layout_boxes, 1000, 1000
        )

        assert layout_type == 'title'

    def test_find_layout_type_for_bbox_best_overlap(self):
        """Test finding layout type when bbox overlaps multiple regions."""
        adapter = PaddleOCRAdapter()

        ocr_bbox = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.6)
        layout_boxes = [
            {'coordinate': [0, 0, 500, 500], 'label': 'text'},  # Partial overlap
            {'coordinate': [100, 100, 800, 800], 'label': 'table'},  # Better overlap
        ]

        layout_type = adapter._find_layout_type_for_bbox(
            ocr_bbox, layout_boxes, 1000, 1000
        )

        assert layout_type == 'table'

    def test_find_layout_type_for_bbox_no_match(self):
        """Test default layout type when no regions match."""
        adapter = PaddleOCRAdapter()

        ocr_bbox = BoundingBox(x_min=0.9, y_min=0.9, x_max=1.0, y_max=1.0)
        layout_boxes = [
            {'coordinate': [0, 0, 500, 500], 'label': 'title'},
        ]

        layout_type = adapter._find_layout_type_for_bbox(
            ocr_bbox, layout_boxes, 1000, 1000
        )

        assert layout_type == 'text'  # Default fallback

    def test_match_ocr_to_layout(self):
        """Test matching OCR results to layout classifications."""
        adapter = PaddleOCRAdapter(min_confidence=0.5)

        rec_texts = ['Title Text', 'Body paragraph']
        rec_scores = [0.95, 0.88]
        rec_polys = [
            [[100, 100], [500, 100], [500, 200], [100, 200]],
            [[100, 300], [800, 300], [800, 500], [100, 500]],
        ]
        layout_boxes = [
            {'coordinate': [50, 50, 550, 250], 'label': 'title'},
            {'coordinate': [50, 250, 850, 550], 'label': 'paragraph'},
        ]

        elements = adapter._match_ocr_to_layout(
            rec_texts, rec_scores, rec_polys,
            layout_boxes, 1000, 1000, page_number=1
        )

        assert len(elements) == 2
        assert elements[0].text == 'Title Text'
        assert elements[0].layout_type == 'title'
        assert elements[1].text == 'Body paragraph'
        assert elements[1].layout_type == 'paragraph'

    def test_match_ocr_to_layout_confidence_filter(self):
        """Test that low confidence results are filtered out."""
        adapter = PaddleOCRAdapter(min_confidence=0.8)

        rec_texts = ['High confidence', 'Low confidence']
        rec_scores = [0.95, 0.6]  # Second one below threshold
        rec_polys = [
            [[100, 100], [500, 100], [500, 200], [100, 200]],
            [[100, 300], [500, 300], [500, 400], [100, 400]],
        ]
        layout_boxes = []

        elements = adapter._match_ocr_to_layout(
            rec_texts, rec_scores, rec_polys,
            layout_boxes, 1000, 1000, page_number=1
        )

        assert len(elements) == 1
        assert elements[0].text == 'High confidence'


class TestOCRTextElementWithLayout:
    """Tests for OCRTextElement with layout_type field."""

    def test_ocr_text_element_default_layout_type(self):
        """Test that layout_type defaults to 'text' for backward compatibility."""
        bbox = BoundingBox(
            x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9,
            pixel_x_min=100, pixel_y_min=200,
            pixel_x_max=800, pixel_y_max=900,
        )

        element = OCRTextElement(
            text="Sample text",
            bbox=bbox,
            confidence=0.95,
            page_number=1,
        )

        assert element.layout_type == "text"

    def test_ocr_text_element_custom_layout_type(self):
        """Test OCRTextElement with custom layout_type."""
        bbox = BoundingBox(
            x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9,
            pixel_x_min=100, pixel_y_min=200,
            pixel_x_max=800, pixel_y_max=900,
        )

        element = OCRTextElement(
            text="Table header",
            bbox=bbox,
            confidence=0.92,
            page_number=1,
            layout_type="table",
        )

        assert element.layout_type == "table"

    def test_ocr_page_result_with_layout_info(self):
        """Test OCRPageResult with optional layout_info."""
        bbox = BoundingBox(
            x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9,
            pixel_x_min=100, pixel_y_min=200,
            pixel_x_max=800, pixel_y_max=900,
        )
        element = OCRTextElement(
            text="Test",
            bbox=bbox,
            confidence=0.9,
            page_number=1,
            layout_type="title",
        )

        layout_info = {
            'layout_boxes': [
                {'coordinate': [100, 200, 800, 900], 'label': 'title'}
            ]
        }

        page_result = OCRPageResult(
            page_number=1,
            elements=[element],
            page_width=1000,
            page_height=1400,
            layout_info=layout_info,
        )

        assert page_result.layout_info is not None
        assert 'layout_boxes' in page_result.layout_info
        assert page_result.elements[0].layout_type == "title"
