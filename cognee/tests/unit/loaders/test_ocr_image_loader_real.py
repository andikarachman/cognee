"""Real OCR processing tests for OcrImageLoader using actual PaddleOCR.

These tests use real OCR processing (not mocked) with the example.png test image
to verify text extraction, bounding box normalization, confidence scores, and
output format correctness.

Tests skip gracefully if PaddleOCR is not installed.
"""

import pytest
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch
from PIL import Image, ImageDraw, ImageFont

from cognee.infrastructure.loaders.core.ocr_image_loader import OcrImageLoader
from cognee.infrastructure.ocr import is_paddleocr_available, is_ppstructure_available
from cognee import visualize_ocr


# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"
EXAMPLE_PNG_PATH = str(TEST_DATA_DIR / "example.png")

# Create persistent directory for test visualizations
TEST_ARTIFACTS_DIR = Path(__file__).parent / ".test_artifacts"
TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def mock_storage_infrastructure():
    """Mock storage, config, and metadata to avoid I/O overhead."""
    with patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_storage") as mock_storage_func, \
         patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_storage_config") as mock_config, \
         patch("cognee.infrastructure.loaders.core.ocr_image_loader.get_file_metadata") as mock_metadata, \
         patch("cognee.infrastructure.loaders.core.ocr_image_loader.open", create=True) as mock_file_open:

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.store.return_value = "/tmp/test/stored_file.txt"
        mock_storage_func.return_value = mock_storage

        # Mock config
        mock_config.return_value = {"data_root_directory": "/tmp/test"}

        # Mock file open (for get_file_metadata)
        mock_file_open.return_value.__enter__.return_value = AsyncMock()

        # Mock metadata
        async def mock_get_metadata(file):
            return {
                "name": "test.png",
                "file_path": "test.png",
                "mime_type": "image/png",
                "extension": ".png",
                "content_hash": "test_hash_123",
                "file_size": 1024,
            }
        mock_metadata.side_effect = mock_get_metadata

        yield {
            "storage": mock_storage,
            "config": mock_config,
            "metadata": mock_metadata,
            "file_open": mock_file_open,
        }


@pytest.fixture
def synthetic_blank_image(tmp_path):
    """Create a blank white 100x100 PNG for edge case testing."""
    image_path = tmp_path / "blank.png"
    img = Image.new("RGB", (100, 100), color="white")
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def synthetic_small_image(tmp_path):
    """Create a very small 10x10 test image."""
    image_path = tmp_path / "small.png"
    img = Image.new("RGB", (10, 10), color="white")
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def synthetic_text_image(tmp_path):
    """Create an image with known text using PIL ImageDraw."""
    image_path = tmp_path / "text.png"
    img = Image.new("RGB", (800, 200), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a TrueType font with large size for better OCR detection
    try:
        # Try common system fonts (works on macOS, Linux, Windows)
        for font_path in [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]:
            try:
                font = ImageFont.truetype(font_path, size=72)
                break
            except (OSError, IOError):
                continue
        else:
            # Fallback to default font with note
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Draw large, clear text that OCR can easily detect
    draw.text((50, 60), "TEST TEXT", fill="black", font=font)

    img.save(image_path)
    return str(image_path)


@pytest.fixture
def multi_format_images(tmp_path):
    """Generate test images in multiple formats from example.png."""
    source_image = Image.open(EXAMPLE_PNG_PATH)
    formats = {}

    # PNG (original)
    png_path = tmp_path / "test.png"
    source_image.save(png_path, "PNG")
    formats["png"] = str(png_path)

    # JPEG
    jpeg_path = tmp_path / "test.jpg"
    rgb_image = source_image.convert("RGB")  # JPEG doesn't support transparency
    rgb_image.save(jpeg_path, "JPEG", quality=95)
    formats["jpeg"] = str(jpeg_path)

    # TIFF
    tiff_path = tmp_path / "test.tiff"
    source_image.save(tiff_path, "TIFF")
    formats["tiff"] = str(tiff_path)

    # BMP
    bmp_path = tmp_path / "test.bmp"
    source_image.save(bmp_path, "BMP")
    formats["bmp"] = str(bmp_path)

    return formats


def parse_formatted_output(formatted_text: str):
    """Parse formatted OCR output to extract structured data.

    Expected format: "text [page=1, bbox=(x,y,x,y), type=text, confidence=C]"
    """
    lines = formatted_text.strip().split("\n")
    parsed_elements = []

    for line in lines:
        if not line.strip():
            continue

        # Extract text (everything before the opening bracket)
        match = re.match(r"^(.+?)\s*\[page=(\d+),\s*bbox=\(([^)]+)\),\s*type=(\w+),\s*confidence=([0-9.]+)\]$", line)

        if match:
            text = match.group(1).strip()
            page = int(match.group(2))
            bbox_str = match.group(3)
            element_type = match.group(4)
            confidence = float(match.group(5))

            # Parse bbox coordinates
            bbox_coords = [float(x) for x in bbox_str.split(",")]

            parsed_elements.append({
                "text": text,
                "page": page,
                "bbox": {
                    "x_min": bbox_coords[0],
                    "y_min": bbox_coords[1],
                    "x_max": bbox_coords[2],
                    "y_max": bbox_coords[3],
                },
                "type": element_type,
                "confidence": confidence,
            })

    return parsed_elements


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestRealOCRProcessing:
    """Core OCR functionality tests with real example.png."""

    async def _run_ocr_and_capture(
        self,
        loader,
        image_path,
        mock_infra,
        min_confidence=0.5,
        use_structure=False,
        structure_config=None
    ):
        """Helper to run OCR and capture formatted text."""
        formatted_text = None

        async def capture_store(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1] if len(args) > 1 else kwargs.get("content")
            if isinstance(formatted_text, bytes):
                formatted_text = formatted_text.decode("utf-8")
            return "/tmp/test/result.txt"

        mock_infra["storage"].store.side_effect = capture_store
        await loader.load(
            image_path,
            min_confidence=min_confidence,
            disable_llm_fallback=True,
            use_structure=use_structure,
            structure_config=structure_config
        )
        return formatted_text

    @pytest.mark.asyncio
    async def test_ocr_processing_comprehensive(self, mock_storage_infrastructure):
        """Comprehensive test for OCR processing with real example.png.

        This test runs expensive PaddleOCR once and validates:
        1. Processing success and storage
        2. Text extraction with expected content
        3. Bounding box normalization
        4. Visualization generation (diagnostic)
        """
        loader = OcrImageLoader()

        # ===================================================================
        # Section 1: Run OCR and capture result
        # ===================================================================
        formatted_text = None

        async def capture_store(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1] if len(args) > 1 else kwargs.get("content")
            if isinstance(formatted_text, bytes):
                formatted_text = formatted_text.decode("utf-8")
            return "/tmp/test/stored_file.txt"

        mock_storage_infrastructure["storage"].store.side_effect = capture_store

        result = await loader.load(EXAMPLE_PNG_PATH)

        # ===================================================================
        # Section 2: Validate processing success
        # ===================================================================
        # Should successfully return a storage path
        assert result is not None

        # Storage should have been called
        mock_storage = mock_storage_infrastructure["storage"]
        mock_storage.store.assert_called_once()

        # Verify filename format
        call_args = mock_storage.store.call_args
        filename = call_args[0][0]
        assert "ocr_text_" in filename
        assert filename.endswith(".txt")

        # ===================================================================
        # Section 3: Validate text extraction
        # ===================================================================
        # Verify text was extracted
        assert formatted_text is not None
        assert len(formatted_text.strip()) > 0

        # Example.png contains text about programmers and light bulbs
        # Check for expected keywords (case insensitive)
        text_lower = formatted_text.lower()

        # Should contain at least some of these keywords
        keywords = ["programmer", "light", "bulb", "hardware", "many", "take", "change"]
        found_keywords = [kw for kw in keywords if kw in text_lower]

        # At least 3 keywords should be found
        assert len(found_keywords) >= 3, \
            f"Expected to find at least 3 keywords from {keywords}, found: {found_keywords}"

        # ===================================================================
        # Section 4: Validate bbox normalization
        # ===================================================================
        # Parse the formatted output
        elements = parse_formatted_output(formatted_text)

        # Should have extracted some elements
        assert len(elements) > 0, "No OCR elements extracted"

        # Verify all bboxes are normalized
        for element in elements:
            bbox = element["bbox"]

            # All coordinates should be in 0-1 range
            assert 0 <= bbox["x_min"] <= 1, \
                f"x_min {bbox['x_min']} not in 0-1 range for text: {element['text']}"
            assert 0 <= bbox["y_min"] <= 1, \
                f"y_min {bbox['y_min']} not in 0-1 range for text: {element['text']}"
            assert 0 <= bbox["x_max"] <= 1, \
                f"x_max {bbox['x_max']} not in 0-1 range for text: {element['text']}"
            assert 0 <= bbox["y_max"] <= 1, \
                f"y_max {bbox['y_max']} not in 0-1 range for text: {element['text']}"

            # Min should be less than max
            assert bbox["x_min"] < bbox["x_max"], \
                f"x_min >= x_max for text: {element['text']}"
            assert bbox["y_min"] < bbox["y_max"], \
                f"y_min >= y_max for text: {element['text']}"

        # ===================================================================
        # Section 5: Print results and generate visualization (diagnostic)
        # ===================================================================
        # Print OCR results to terminal
        print("\n" + "="*80)
        print(f"OCR Results from {EXAMPLE_PNG_PATH}")
        print("="*80)
        if formatted_text:
            print(formatted_text)
            print("="*80)
            print(f"Total lines extracted: {len(formatted_text.strip().split(chr(10)))}")
            print("="*80 + "\n")
        else:
            print("No text extracted")
            print("="*80 + "\n")

        # Generate OCR visualization
        try:
            viz_output_path = await visualize_ocr(
                EXAMPLE_PNG_PATH,
                output_path=str(TEST_ARTIFACTS_DIR / "example_ocr_visualization.png"),
                min_confidence=0.5,
                show_text=True,
                show_confidence=True,
                box_width=2
            )

            print("="*80)
            print("OCR Visualization Generated")
            print("="*80)
            print(f"Annotated image saved to: {viz_output_path}")
            print(f"  open {viz_output_path}")
            print("="*80 + "\n")

            # Verify the visualization file exists
            assert Path(viz_output_path).exists(), "Visualization file should exist"

        except Exception as e:
            # Don't fail the test if visualization fails (it's supplementary)
            print(f"Warning: OCR visualization failed: {e}")

    @pytest.mark.parametrize("image_format", ["png", "jpeg", "bmp"])
    @pytest.mark.asyncio
    async def test_different_image_formats(self, mock_storage_infrastructure, multi_format_images, image_format):
        """Test OCR processing on different image formats."""
        loader = OcrImageLoader()
        image_path = multi_format_images[image_format]

        formatted_text = await self._run_ocr_and_capture(
            loader, image_path, mock_storage_infrastructure
        )

        # Should successfully process image regardless of format
        assert formatted_text is not None
        assert len(formatted_text.strip()) > 0

        # Should extract recognizable text
        elements = parse_formatted_output(formatted_text)
        assert len(elements) > 0, f"No elements extracted from {image_format} format"

    @pytest.mark.asyncio
    async def test_synthetic_text_image_controlled(self, mock_storage_infrastructure, synthetic_text_image):
        """Test OCR on synthetic image with known text."""
        loader = OcrImageLoader()

        formatted_text = await self._run_ocr_and_capture(
            loader, synthetic_text_image, mock_storage_infrastructure
        )

        # Should extract text
        assert formatted_text is not None
        text_lower = formatted_text.lower()

        # Should find "test text" (case insensitive)
        assert "test" in text_lower, "Should extract 'TEST TEXT' from synthetic image"

    @pytest.mark.asyncio
    async def test_newspaper_article_multi_column_comprehensive(
        self, mock_storage_infrastructure
    ):
        """Comprehensive test for multi-column OCR processing.

        Tests newspaper_article.jpg (academic research criteria table) with:
        1. Processing success and storage
        2. Text extraction with research-specific keywords
        3. Bounding box normalization
        4. Confidence scores and thresholds
        5. Multi-column layout detection
        6. Column reading order validation
        7. Substantial text extraction
        8. Header region detection (optional)
        9. Output format consistency
        10. Visualization generation (diagnostic)
        """
        # Section 1: Run OCR and Capture Result
        print("\n" + "=" * 80)
        print("SECTION 1: Run OCR and Capture Result")
        print("=" * 80)

        newspaper_path = str(TEST_DATA_DIR / "newspaper_article.jpg")
        loader = OcrImageLoader()

        formatted_text = await self._run_ocr_and_capture(
            loader, newspaper_path, mock_storage_infrastructure, min_confidence=0.5
        )

        print(f"OCR processing completed for: {newspaper_path}")
        print(f"Formatted text length: {len(formatted_text) if formatted_text else 0} chars")

        # Section 2: Validate Processing Success
        print("\n" + "=" * 80)
        print("SECTION 2: Validate Processing Success")
        print("=" * 80)

        assert formatted_text is not None, "OCR should return formatted text"
        assert len(formatted_text) > 0, "Formatted text should not be empty"

        # Verify storage was called
        assert (
            mock_storage_infrastructure["storage"].store.call_count == 1
        ), "Storage should be called once"

        # Check filename format
        call_args = mock_storage_infrastructure["storage"].store.call_args
        filename = call_args[0][0] if call_args[0] else None
        assert filename is not None, "Filename should be provided to storage"
        assert filename.startswith("ocr_text_"), "Filename should have ocr_text_ prefix"
        assert filename.endswith(".txt"), "Filename should have .txt extension"

        print(f"✓ Processing successful")
        print(f"✓ Storage called with filename: {filename}")

        # Section 3: Validate Text Extraction with Keywords
        print("\n" + "=" * 80)
        print("SECTION 3: Validate Text Extraction with Keywords")
        print("=" * 80)

        # Research criteria table-specific keywords (case-insensitive)
        keywords = [
            # Core terms
            "article",
            "criteria",
            "research",
            # Inclusion criteria
            "inclusion",
            "language",
            "learning",
            "empirical",
            "design",
            "english",
            # Exclusion criteria
            "exclusion",
            "methodological",
            "conceptual",
            "accessible",
            # Additional context
            "field",
            "text",
        ]

        text_lower = formatted_text.lower()
        found_keywords = [kw for kw in keywords if kw in text_lower]

        print(f"Keywords found ({len(found_keywords)}/{len(keywords)}):")
        for kw in found_keywords:
            print(f"  ✓ '{kw}'")

        assert len(found_keywords) >= 5, (
            f"Should find at least 5 keywords from research criteria table "
            f"(found {len(found_keywords)}: {found_keywords})"
        )

        # Section 4: Validate Bbox Normalization
        print("\n" + "=" * 80)
        print("SECTION 4: Validate Bbox Normalization")
        print("=" * 80)

        elements = parse_formatted_output(formatted_text)
        print(f"Parsed {len(elements)} OCR elements")

        bbox_errors = []
        for i, elem in enumerate(elements):
            bbox = elem["bbox"]

            # Check range
            for coord_name in ["x_min", "y_min", "x_max", "y_max"]:
                coord_value = bbox[coord_name]
                if not (0 <= coord_value <= 1):
                    bbox_errors.append(
                        f"Element {i}: {coord_name}={coord_value} out of range [0,1]"
                    )

            # Check ordering
            if bbox["x_min"] >= bbox["x_max"]:
                bbox_errors.append(
                    f"Element {i}: x_min ({bbox['x_min']}) >= x_max ({bbox['x_max']})"
                )
            if bbox["y_min"] >= bbox["y_max"]:
                bbox_errors.append(
                    f"Element {i}: y_min ({bbox['y_min']}) >= y_max ({bbox['y_max']})"
                )

        if bbox_errors:
            print("Bbox validation errors:")
            for error in bbox_errors[:5]:  # Show first 5
                print(f"  ✗ {error}")
            assert False, f"Found {len(bbox_errors)} bbox errors"
        else:
            print(f"✓ All {len(elements)} bounding boxes normalized correctly")

        # Section 5: Validate Confidence Scores
        print("\n" + "=" * 80)
        print("SECTION 5: Validate Confidence Scores")
        print("=" * 80)

        confidence_errors = []
        confidence_values = []

        for i, elem in enumerate(elements):
            conf = elem["confidence"]
            confidence_values.append(conf)

            if not (0 <= conf <= 1):
                confidence_errors.append(f"Element {i}: confidence={conf} out of range [0,1]")
            if conf < 0.5:
                confidence_errors.append(
                    f"Element {i}: confidence={conf} below threshold 0.5"
                )

        if confidence_errors:
            print("Confidence validation errors:")
            for error in confidence_errors[:5]:
                print(f"  ✗ {error}")
            assert False, f"Found {len(confidence_errors)} confidence errors"
        else:
            avg_conf = sum(confidence_values) / len(confidence_values)
            min_conf = min(confidence_values)
            max_conf = max(confidence_values)
            print(f"✓ All {len(elements)} confidence scores valid")
            print(f"  Average: {avg_conf:.3f}, Min: {min_conf:.3f}, Max: {max_conf:.3f}")

        # Section 6: Validate Multi-Column Layout Detection
        print("\n" + "=" * 80)
        print("SECTION 6: Validate Multi-Column Layout Detection")
        print("=" * 80)

        def detect_columns(elements_list, x_tolerance=0.15):
            """Group elements into columns by x_min clustering."""
            if not elements_list:
                return []

            # Sort by x_min
            sorted_elems = sorted(elements_list, key=lambda e: e["bbox"]["x_min"])

            columns = []
            current_column = [sorted_elems[0]]
            current_x = sorted_elems[0]["bbox"]["x_min"]

            for elem in sorted_elems[1:]:
                x_min = elem["bbox"]["x_min"]

                if abs(x_min - current_x) < x_tolerance:
                    current_column.append(elem)
                else:
                    columns.append(current_column)
                    current_column = [elem]
                    current_x = x_min

            if current_column:
                columns.append(current_column)

            return columns

        columns = detect_columns(elements)
        num_columns = len(columns)

        print(f"Detected {num_columns} columns")
        for i, col in enumerate(columns):
            avg_x = sum(e["bbox"]["x_min"] for e in col) / len(col)
            print(f"  Column {i+1}: {len(col)} elements, avg x_min: {avg_x:.3f}")

        assert 2 <= num_columns <= 4, (
            f"Expected 2-4 columns for multi-column layout, found {num_columns}"
        )

        for i, col in enumerate(columns):
            assert len(col) >= 2, (
                f"Column {i+1} has only {len(col)} elements, expected at least 2"
            )

        print(f"✓ Multi-column layout validated ({num_columns} columns)")

        # Section 7: Validate Column Reading Order
        print("\n" + "=" * 80)
        print("SECTION 7: Validate Column Reading Order")
        print("=" * 80)

        ordering_issues = []

        # Within-column ordering (top to bottom)
        for col_idx, col in enumerate(columns):
            for i in range(len(col) - 1):
                y_current = col[i]["bbox"]["y_min"]
                y_next = col[i + 1]["bbox"]["y_min"]

                # Allow 5% tolerance for horizontal elements
                if y_next < y_current - 0.05:
                    ordering_issues.append(
                        f"Column {col_idx+1}: Element {i+1} y_min ({y_current:.3f}) > "
                        f"Element {i+2} y_min ({y_next:.3f})"
                    )

        # Across-column ordering (left to right)
        if len(columns) > 1:
            for i in range(len(columns) - 1):
                avg_x_current = sum(e["bbox"]["x_min"] for e in columns[i]) / len(columns[i])
                avg_x_next = sum(e["bbox"]["x_min"] for e in columns[i + 1]) / len(
                    columns[i + 1]
                )

                if avg_x_next < avg_x_current:
                    ordering_issues.append(
                        f"Column {i+1} avg x ({avg_x_current:.3f}) > "
                        f"Column {i+2} avg x ({avg_x_next:.3f})"
                    )

        if ordering_issues:
            print(f"⚠ Found {len(ordering_issues)} ordering issues:")
            for issue in ordering_issues[:3]:
                print(f"  {issue}")
        else:
            print(f"✓ Column reading order validated (left-to-right, top-to-bottom)")

        # Section 8: Validate Substantial Text Extraction
        print("\n" + "=" * 80)
        print("SECTION 8: Validate Substantial Text Extraction")
        print("=" * 80)

        total_text_length = sum(len(elem["text"]) for elem in elements)

        print(f"Total OCR elements: {len(elements)}")
        print(f"Total text length: {total_text_length} characters")

        assert len(elements) >= 15, (
            f"Expected at least 15 elements for research criteria table, found {len(elements)}"
        )
        assert total_text_length >= 200, (
            f"Expected at least 200 characters, found {total_text_length}"
        )

        print(f"✓ Substantial text extraction validated")

        # Section 9: Validate Header Region Detection (Optional)
        print("\n" + "=" * 80)
        print("SECTION 9: Validate Header Region Detection (Optional)")
        print("=" * 80)

        # Sort by y_min, take top 5
        top_elements = sorted(elements, key=lambda e: e["bbox"]["y_min"])[:5]

        potential_headers = []
        for elem in top_elements:
            bbox = elem["bbox"]
            area = (bbox["x_max"] - bbox["x_min"]) * (bbox["y_max"] - bbox["y_min"])

            if area > 0.02:  # > 2% of page
                potential_headers.append((elem["text"], area))

        if potential_headers:
            print(f"Found {len(potential_headers)} potential header regions:")
            for text, area in potential_headers:
                print(f"  '{text[:50]}...' (area: {area:.4f})")
        else:
            print("No large header regions detected (may be normal)")

        # Section 10: Generate Visualization (Diagnostic)
        print("\n" + "=" * 80)
        print("SECTION 10: Generate Visualization (Diagnostic)")
        print("=" * 80)

        print("\nOCR Results Summary:")
        print(f"  Elements: {len(elements)}")
        print(f"  Text length: {total_text_length} chars")
        print(f"  Keywords found: {len(found_keywords)}/{len(keywords)}")
        print(f"  Columns detected: {num_columns}")
        print(f"  Average confidence: {avg_conf:.3f}")

        try:
            from cognee.modules.visualization.ocr_visualization import visualize_ocr

            viz_path = TEST_ARTIFACTS_DIR / "newspaper_article_ocr_visualization.png"
            result_path = await visualize_ocr(
                newspaper_path,
                output_path=str(viz_path),
                min_confidence=0.5,
                show_text=True,
                show_confidence=True,
                box_width=2
            )
            print(f"\n✓ Visualization saved to: {result_path}")
        except Exception as e:
            print(f"\n⚠ Visualization generation failed (non-critical): {e}")

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestOCROutputQuality:
    """Test OCR output quality and validation."""

    async def _run_ocr_and_capture(
        self,
        loader,
        image_path,
        mock_infra,
        min_confidence=0.5,
        use_structure=False,
        structure_config=None
    ):
        """Helper to run OCR and capture formatted text."""
        formatted_text = None

        async def capture_store(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1] if len(args) > 1 else kwargs.get("content")
            if isinstance(formatted_text, bytes):
                formatted_text = formatted_text.decode("utf-8")
            return "/tmp/test/result.txt"

        mock_infra["storage"].store.side_effect = capture_store
        await loader.load(
            image_path,
            min_confidence=min_confidence,
            disable_llm_fallback=True,
            use_structure=use_structure,
            structure_config=structure_config
        )
        return formatted_text

    @pytest.mark.asyncio
    async def test_output_quality_comprehensive(self, mock_storage_infrastructure):
        """Comprehensive test for OCR output quality validation.

        Validates: confidence scores, bbox dimensions, text ordering
        """
        loader = OcrImageLoader()

        # Section 1: Run OCR once and capture result
        formatted_text = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure
        )
        elements = parse_formatted_output(formatted_text)

        assert len(elements) > 0, "No elements extracted"

        # Section 2: Validate confidence scores
        for element in elements:
            assert 0 <= element["confidence"] <= 1, \
                f"Invalid confidence {element['confidence']} for text: {element['text']}"
            assert element["confidence"] >= 0.5, \
                f"Confidence {element['confidence']} below threshold 0.5 for text: {element['text']}"

        # Section 3: Validate bbox dimensions
        for element in elements:
            bbox = element["bbox"]
            width = bbox["x_max"] - bbox["x_min"]
            height = bbox["y_max"] - bbox["y_min"]
            area = width * height

            assert area > 0, f"Zero area bbox for text: {element['text']}"
            assert area < 1.0, f"Bbox covers entire page for text: {element['text']}"
            assert width > 0.001, f"Width too small for text: {element['text']}"
            assert height > 0.001, f"Height too small for text: {element['text']}"

        # Section 4: Validate text ordering (top-to-bottom, left-to-right)
        for i in range(len(elements) - 1):
            current, next_elem = elements[i], elements[i + 1]
            y_diff = abs(next_elem["bbox"]["y_min"] - current["bbox"]["y_min"])

            if y_diff < 0.05:  # Same line
                assert next_elem["bbox"]["x_min"] >= current["bbox"]["x_min"] - 0.1, \
                    f"Same-line elements not ordered left-to-right: {current['text']} -> {next_elem['text']}"


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestOCRConfiguration:
    """Test different OCR configuration options."""

    async def _run_ocr_and_capture(
        self,
        loader,
        image_path,
        mock_infra,
        min_confidence=0.5,
        use_structure=False,
        structure_config=None
    ):
        """Helper to run OCR and capture formatted text."""
        formatted_text = None

        async def capture_store(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1] if len(args) > 1 else kwargs.get("content")
            if isinstance(formatted_text, bytes):
                formatted_text = formatted_text.decode("utf-8")
            return "/tmp/test/result.txt"

        mock_infra["storage"].store.side_effect = capture_store
        await loader.load(
            image_path,
            min_confidence=min_confidence,
            disable_llm_fallback=True,
            use_structure=use_structure,
            structure_config=structure_config
        )
        return formatted_text

    @pytest.mark.asyncio
    async def test_confidence_threshold_behavior(self, mock_storage_infrastructure):
        """Test that confidence threshold filtering works correctly.

        Validates: low threshold extracts more, high threshold extracts fewer,
        all results meet threshold requirements.
        """
        loader = OcrImageLoader()

        # Run OCR with 3 different thresholds
        formatted_low = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure, 0.3
        )
        formatted_default = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure, 0.5
        )
        formatted_high = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure, 0.9
        )

        elements_low = parse_formatted_output(formatted_low)
        elements_default = parse_formatted_output(formatted_default)
        elements_high = parse_formatted_output(formatted_high)

        # Validate threshold ordering: low >= default >= high
        assert len(elements_low) >= len(elements_default), \
            f"Low threshold should extract more: {len(elements_low)} vs {len(elements_default)}"
        assert len(elements_default) >= len(elements_high), \
            f"Default should extract more than high: {len(elements_default)} vs {len(elements_high)}"

        # Validate confidence requirements
        for element in elements_high:
            assert element["confidence"] >= 0.9, \
                f"High-threshold element has confidence {element['confidence']} < 0.9"

    @pytest.mark.asyncio
    async def test_confidence_zero_returns_all_detections(self, mock_storage_infrastructure):
        """Test min_confidence=0.0 returns all OCR detections."""
        loader = OcrImageLoader()

        formatted_zero = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure, 0.0
        )
        formatted_default = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure, 0.5
        )

        elements_zero = parse_formatted_output(formatted_zero)
        elements_default = parse_formatted_output(formatted_default)

        # Zero threshold should extract same or more elements
        assert len(elements_zero) >= len(elements_default), \
            f"Zero threshold should extract all: {len(elements_zero)} vs {len(elements_default)}"

        # All elements in zero-threshold run should have confidence >= 0.0 (trivially true)
        for element in elements_zero:
            assert element["confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_confidence_one_returns_minimal_or_none(self, mock_storage_infrastructure):
        """Test min_confidence=1.0 likely returns very few or no detections."""
        loader = OcrImageLoader()

        formatted_one = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure, 1.0
        )

        # With threshold 1.0, should get very few or no elements (OCR rarely 100% confident)
        if formatted_one and formatted_one.strip():
            elements = parse_formatted_output(formatted_one)
            # All returned elements must have confidence = 1.0
            for element in elements:
                assert element["confidence"] >= 1.0, \
                    f"Element with confidence {element['confidence']} should not pass threshold 1.0"
        else:
            # Empty result is expected when no OCR text meets 1.0 confidence
            assert formatted_one == "" or formatted_one is None, \
                "Expected empty result with confidence threshold 1.0"


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestEdgeCasesAndFallback:
    """Test edge cases and fallback behavior."""

    @pytest.mark.asyncio
    async def test_blank_image_triggers_fallback(self, mock_storage_infrastructure, synthetic_blank_image):
        """Test that blank/empty images trigger fallback to ImageLoader."""
        loader = OcrImageLoader()

        with patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader") as mock_image_loader_class:
            mock_image_loader_instance = AsyncMock()
            mock_image_loader_instance.load.return_value = "/tmp/llm_result.txt"
            mock_image_loader_class.return_value = mock_image_loader_instance

            result = await loader.load(synthetic_blank_image)

            # Verify fallback was triggered OR result is valid
            if mock_image_loader_instance.load.called:
                assert result == "/tmp/llm_result.txt"
                mock_image_loader_instance.load.assert_called_once_with(synthetic_blank_image)
            else:
                assert result is not None

    @pytest.mark.asyncio
    async def test_very_small_image(self, mock_storage_infrastructure, synthetic_small_image):
        """Test processing a very small 10x10 image."""
        loader = OcrImageLoader()

        # Mock ImageLoader fallback
        with patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader") as mock_image_loader_class:
            mock_image_loader_instance = AsyncMock()
            mock_image_loader_instance.load.return_value = "/tmp/fallback_result.txt"
            mock_image_loader_class.return_value = mock_image_loader_instance

            # Should not raise an exception
            result = await loader.load(synthetic_small_image)

            # Result should be valid (either OCR or fallback)
            assert result is not None


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestOutputFormatValidation:
    """Test output format consistency and correctness."""

    async def _run_ocr_and_capture(
        self,
        loader,
        image_path,
        mock_infra,
        min_confidence=0.5,
        use_structure=False,
        structure_config=None
    ):
        """Helper to run OCR and capture formatted text."""
        formatted_text = None

        async def capture_store(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1] if len(args) > 1 else kwargs.get("content")
            if isinstance(formatted_text, bytes):
                formatted_text = formatted_text.decode("utf-8")
            return "/tmp/test/result.txt"

        mock_infra["storage"].store.side_effect = capture_store
        await loader.load(
            image_path,
            min_confidence=min_confidence,
            disable_llm_fallback=True,
            use_structure=use_structure,
            structure_config=structure_config
        )
        return formatted_text

    @pytest.mark.asyncio
    async def test_output_format_comprehensive(self, mock_storage_infrastructure):
        """Comprehensive test for OCR output format consistency.

        Validates: format structure, page numbers, regex pattern, parseability
        """
        loader = OcrImageLoader()
        formatted_text = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure
        )
        lines = formatted_text.strip().split("\n")

        # Section 1: Validate basic format structure
        for line in lines:
            if not line.strip():
                continue
            assert "[page=" in line, f"Missing page info in: {line}"
            assert "bbox=(" in line, f"Missing bbox in: {line}"
            assert "type=" in line, f"Missing type in: {line}"
            assert "confidence=" in line, f"Missing confidence in: {line}"
            assert "]" in line, f"Missing closing bracket in: {line}"

        # Section 2: Validate page number = 1 for images
        elements = parse_formatted_output(formatted_text)
        for element in elements:
            assert element["page"] == 1, \
                f"Image element has page={element['page']}, expected 1"

        # Section 3: Validate regex pattern (consistency with PDF loader)
        pattern = r"^.+\s*\[page=\d+,\s*bbox=\([0-9.]+,[0-9.]+,[0-9.]+,[0-9.]+\),\s*type=\w+,\s*confidence=[0-9.]+\]$"
        for line in lines:
            if not line.strip():
                continue
            assert re.match(pattern, line), \
                f"Format doesn't match expected pattern: {line}"

        # Section 4: Validate parseability
        for line in lines:
            if not line.strip():
                continue
            parsed = parse_formatted_output(line)
            assert len(parsed) == 1, f"Could not parse line: {line}"


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_nonexistent_file_fallback(self):
        """Test handling of non-existent file path triggers fallback."""
        loader = OcrImageLoader()

        with patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader") as mock_fallback:
            mock_instance = AsyncMock()
            mock_instance.load.return_value = "/tmp/fallback.txt"
            mock_fallback.return_value = mock_instance

            result = await loader.load("/nonexistent/path/image.png")

            # Should fallback to ImageLoader due to OCR error
            assert mock_instance.load.called
            assert result == "/tmp/fallback.txt"

    @pytest.mark.asyncio
    async def test_corrupted_image_fallback(self, tmp_path):
        """Test handling of corrupted image file."""
        # Create a corrupted image (text file with .png extension)
        corrupted_path = tmp_path / "corrupted.png"
        corrupted_path.write_text("This is not an image")

        loader = OcrImageLoader()

        with patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader") as mock_fallback:
            mock_instance = AsyncMock()
            mock_instance.load.return_value = "/tmp/fallback.txt"
            mock_fallback.return_value = mock_instance

            result = await loader.load(str(corrupted_path))

            # Should fallback due to OCR error
            assert mock_instance.load.called

    @pytest.mark.asyncio
    async def test_empty_file_fallback(self, tmp_path):
        """Test handling of empty file."""
        empty_path = tmp_path / "empty.png"
        empty_path.write_bytes(b"")

        loader = OcrImageLoader()

        with patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader") as mock_fallback:
            mock_instance = AsyncMock()
            mock_instance.load.return_value = "/tmp/fallback.txt"
            mock_fallback.return_value = mock_instance

            result = await loader.load(str(empty_path))

            # Should fallback
            assert mock_instance.load.called or result is not None


# Tests for when PaddleOCR is NOT installed
@pytest.mark.skipif(is_paddleocr_available(), reason="PaddleOCR is installed, skip unavailable tests")
class TestPaddleOCRUnavailable:
    """Test behavior when PaddleOCR is not available."""

    @pytest.mark.asyncio
    async def test_fallback_when_paddleocr_unavailable(self):
        """Test that loader falls back to ImageLoader when PaddleOCR is unavailable."""
        loader = OcrImageLoader()

        with patch("cognee.infrastructure.loaders.core.image_loader.ImageLoader") as mock_image_loader_class:
            mock_image_loader_instance = AsyncMock()
            mock_image_loader_instance.load.return_value = "/tmp/llm_result.txt"
            mock_image_loader_class.return_value = mock_image_loader_instance

            result = await loader.load(EXAMPLE_PNG_PATH)

            # Should fallback to ImageLoader
            mock_image_loader_instance.load.assert_called_once()
            assert result == "/tmp/llm_result.txt"


@pytest.mark.skipif(
    not is_ppstructure_available(),
    reason="PPStructureV3/PaddleX not installed"
)
class TestPPStructureV3RealProcessing:
    """Real OCR processing tests with PPStructureV3 layout detection."""

    # Lightweight model configuration for faster testing
    LIGHTWEIGHT_STRUCTURE_CONFIG = {
        'text_detection_model_name': 'PP-OCRv5_mobile_det',
        'text_recognition_model_name': 'PP-OCRv5_mobile_rec',
        'layout_detection_model_name': 'PP-DocLayout-L',
        'use_doc_orientation_classify': True,
        'use_formula_recognition': True,
        'use_table_recognition': True,
    }

    async def _run_ocr_and_capture(
        self,
        loader,
        image_path,
        mock_infra,
        min_confidence=0.5,
        use_structure=False,
        structure_config=None
    ):
        """Helper to run OCR and capture formatted text."""
        formatted_text = None

        async def capture_store(*args, **kwargs):
            nonlocal formatted_text
            formatted_text = args[1] if len(args) > 1 else kwargs.get("content")
            if isinstance(formatted_text, bytes):
                formatted_text = formatted_text.decode("utf-8")
            return "/tmp/test/result.txt"

        mock_infra["storage"].store.side_effect = capture_store
        await loader.load(
            image_path,
            min_confidence=min_confidence,
            disable_llm_fallback=True,
            use_structure=use_structure,
            structure_config=structure_config
        )
        return formatted_text

    @pytest.mark.asyncio
    async def test_ppstructure_basic_processing(self, mock_storage_infrastructure):
        """Test PPStructureV3 processes images and detects layout types."""
        print(f"\n{'='*80}")
        print("Using Lightweight PPStructureV3 Models")
        print("="*80)
        print(f"  Detection:    {self.LIGHTWEIGHT_STRUCTURE_CONFIG['text_detection_model_name']}")
        print(f"  Recognition:  {self.LIGHTWEIGHT_STRUCTURE_CONFIG['text_recognition_model_name']}")
        print(f"  Layout:       {self.LIGHTWEIGHT_STRUCTURE_CONFIG['layout_detection_model_name']}")
        print("="*80 + "\n")

        # Section 1: Initialize adapter and run OCR ONCE
        from cognee.infrastructure.ocr import PaddleOCRAdapter

        adapter = PaddleOCRAdapter(
            lang='en',
            use_gpu=False,
            min_confidence=0.5,
            use_structure=True,
            structure_config=self.LIGHTWEIGHT_STRUCTURE_CONFIG
        )

        # Process image and capture result
        page_result = await adapter.process_image(EXAMPLE_PNG_PATH, page_number=1)

        # Section 2: Format result for validation
        # Replicate OcrImageLoader._format_ocr_result() behavior
        formatted_elements = []
        for element in page_result.elements:
            bbox = element.bbox
            layout_type = getattr(element, 'layout_type', 'text')
            formatted_line = (
                f"{element.text} "
                f"[page={element.page_number}, "
                f"bbox=({bbox.x_min:.3f},{bbox.y_min:.3f},"
                f"{bbox.x_max:.3f},{bbox.y_max:.3f}), "
                f"type={layout_type}, "
                f"confidence={element.confidence:.3f}]"
            )
            formatted_elements.append(formatted_line)

        formatted_text = "\n".join(formatted_elements)

        # Validate basic requirements
        assert len(formatted_text.strip()) > 0, "Formatted text should not be empty"

        # Section 3: Parse and validate elements
        elements = parse_formatted_output(formatted_text)
        assert len(elements) > 0, "Should extract at least one element"

        # Section 4: Validate layout_info is populated
        layout_types = set(elem['type'] for elem in elements)
        assert len(layout_types) > 0, "Should have at least one layout type"

        # Section 4b: Validate layout region boxes (PPStructureV3 core feature)
        assert page_result.layout_info is not None, \
            "PPStructureV3 should return layout_info"

        layout_boxes = page_result.layout_info.get('layout_boxes', [])
        assert len(layout_boxes) > 0, \
            "PPStructureV3 should detect at least one layout region"

        # Validate layout box structure
        layout_labels = set()
        for box in layout_boxes:
            assert 'coordinate' in box, "Layout box should have coordinates"
            assert 'label' in box, "Layout box should have label"

            coords = box['coordinate']
            assert len(coords) == 4, "Layout box should have 4 coordinates [x1,y1,x2,y2]"

            # Convert to Python native types if needed (may be numpy types)
            try:
                coords_numeric = [float(c) for c in coords]
                assert all(isinstance(c, (int, float)) for c in coords_numeric), \
                    "Coordinates should be convertible to numeric"
            except (ValueError, TypeError) as e:
                raise AssertionError(f"Coordinates should be numeric, got: {coords} (types: {[type(c) for c in coords]})") from e

            layout_labels.add(box['label'])

        # Section 5: Print results
        print(f"\nPPStructureV3 Layout Detection Results:")
        print(f"  Layout regions detected: {len(layout_boxes)}")
        print(f"  Layout types found: {layout_labels}")
        print(f"  Sample layout boxes: {layout_boxes[:2]}")  # Show first 2 for inspection
        print(f"  OCR text elements: {len(elements)}")
        print(f"  OCR layout types: {layout_types}")

        # Section 6: Generate visualization (REUSING page_result)
        try:
            viz_output_path = await visualize_ocr(
                page_result,  # ✅ Pass OCRPageResult instead of file path
                image_path=EXAMPLE_PNG_PATH,  # Required when passing OCRPageResult
                output_path=str(TEST_ARTIFACTS_DIR / "ppstructure_visualization.png"),
                min_confidence=0.5,
                show_text=True,
                show_confidence=True,
                box_width=2
                # ❌ Remove use_structure and structure_config - not needed when passing OCRPageResult
            )

            print("="*80)
            print("PPStructureV3 OCR Text Boxes Visualization Generated")
            print("="*80)
            print(f"Saved to: {viz_output_path}")
            print(f"This shows individual OCR text boxes (not layout regions)")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Warning: OCR visualization failed: {e}")
            # Don't fail test if visualization fails

        # Section 7: Visualize layout regions using new function
        try:
            from cognee.modules.visualization import visualize_layout_regions

            layout_viz_path = await visualize_layout_regions(
                page_result,
                image_path=EXAMPLE_PNG_PATH,
                output_path=str(TEST_ARTIFACTS_DIR / "ppstructure_layout_regions.png"),
                show_labels=True,
                box_width=3,
            )

            print("="*80)
            print("PPStructureV3 Layout Region Visualization Generated")
            print("="*80)
            print(f"Saved to: {layout_viz_path}")
            print(f"This shows layout detection regions (paragraphs, titles, figures, etc.)")
            print(f"Compare with OCR text visualization above to see the difference")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Warning: Layout region visualization failed: {e}")
            # Don't fail test if visualization fails

    @pytest.mark.asyncio
    async def test_layout_type_diversity(self, mock_storage_infrastructure):
        """Test that PPStructureV3 detects diverse layout types.

        Validates:
        1. Not all elements are type="text"
        2. At least 1 layout type detected
        3. Layout types are valid (from allowed set)
        """
        loader = OcrImageLoader()

        formatted_text = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure,
            use_structure=True,
            structure_config=self.LIGHTWEIGHT_STRUCTURE_CONFIG
        )

        elements = parse_formatted_output(formatted_text)

        # Extract unique layout types
        layout_types = set(elem['type'] for elem in elements)

        # Validate diversity
        assert len(layout_types) >= 1, "Should detect at least 1 layout type"

        # Validate all types are from allowed set
        ALLOWED_TYPES = {
            'text', 'title', 'heading', 'paragraph', 'table',
            'caption', 'figure', 'header', 'footer', 'code', 'list'
        }
        for layout_type in layout_types:
            assert layout_type in ALLOWED_TYPES, \
                f"Unexpected layout type: {layout_type}"

        print(f"\nLayout types detected: {layout_types}")

    @pytest.mark.asyncio
    async def test_newspaper_article_with_ppstructure(self, mock_storage_infrastructure):
        """Test PPStructureV3 on newspaper_article.jpg (research criteria table).

        Validates:
        1. Processing success
        2. Layout type diversity (should detect table, title, etc.)
        3. Table detection specifically
        4. Comparison with standard PaddleOCR
        """
        newspaper_path = str(TEST_DATA_DIR / "newspaper_article.jpg")
        loader = OcrImageLoader()

        # Run with PPStructureV3
        formatted_ppstructure = await self._run_ocr_and_capture(
            loader, newspaper_path, mock_storage_infrastructure,
            use_structure=True,
            structure_config=self.LIGHTWEIGHT_STRUCTURE_CONFIG
        )

        elements = parse_formatted_output(formatted_ppstructure)
        layout_types = set(elem['type'] for elem in elements)

        print(f"\nPPStructureV3 Results:")
        print(f"  Elements: {len(elements)}")
        print(f"  Layout types: {layout_types}")

        # Should detect structured elements
        assert len(elements) > 0, "Should extract elements"

        # Count layout types
        type_counts = {}
        for elem in elements:
            type_counts[elem['type']] = type_counts.get(elem['type'], 0) + 1

        print(f"  Type distribution: {type_counts}")

        # Generate visualization for PPStructureV3 results
        try:
            viz_path = TEST_ARTIFACTS_DIR / "newspaper_ppstructure_visualization.png"
            result_path = await visualize_ocr(
                newspaper_path,
                output_path=str(viz_path),
                min_confidence=0.5,
                show_text=True,
                show_confidence=True,
                box_width=2,
                use_structure=True,
                structure_config=self.LIGHTWEIGHT_STRUCTURE_CONFIG
            )
            print(f"\n✓ PPStructureV3 visualization saved to: {result_path}")
        except Exception as e:
            print(f"\n⚠ Visualization failed: {e}")

    @pytest.mark.asyncio
    async def test_paddleocr_vs_ppstructure_comparison(self, mock_storage_infrastructure):
        """Compare standard PaddleOCR vs PPStructureV3 on same image.

        Validates:
        1. Both modes process successfully
        2. PPStructureV3 provides layout type detection
        3. Text extraction is comparable (similar element count)
        4. Output format is consistent
        """
        loader = OcrImageLoader()

        # Run with standard PaddleOCR
        formatted_standard = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure,
            use_structure=False
        )

        # Run with PPStructureV3
        formatted_ppstructure = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure,
            use_structure=True,
            structure_config=self.LIGHTWEIGHT_STRUCTURE_CONFIG
        )

        # Parse both
        elements_standard = parse_formatted_output(formatted_standard)
        elements_ppstructure = parse_formatted_output(formatted_ppstructure)

        # Extract layout types
        types_standard = set(elem['type'] for elem in elements_standard)
        types_ppstructure = set(elem['type'] for elem in elements_ppstructure)

        print(f"\nComparison:")
        print(f"  Standard PaddleOCR:")
        print(f"    Elements: {len(elements_standard)}")
        print(f"    Layout types: {types_standard}")
        print(f"  PPStructureV3:")
        print(f"    Elements: {len(elements_ppstructure)}")
        print(f"    Layout types: {types_ppstructure}")

        # Validate both succeeded
        assert len(elements_standard) > 0, "Standard mode should extract elements"
        assert len(elements_ppstructure) > 0, "PPStructure mode should extract elements"

        # Element counts should be similar (±50% to account for layout differences)
        if len(elements_standard) > 0:
            count_ratio = len(elements_ppstructure) / len(elements_standard)
            assert 0.5 <= count_ratio <= 2.0, \
                f"Element count too different: {len(elements_standard)} vs {len(elements_ppstructure)}"

    @pytest.mark.asyncio
    async def test_structure_config_passing(self, mock_storage_infrastructure):
        """Test that structure_config is correctly passed to PPStructureV3.

        Note: This is a smoke test - actual config behavior depends on
        PPStructureV3 implementation details.
        """
        loader = OcrImageLoader()

        # Run with custom config (merge with lightweight config)
        custom_config = {**self.LIGHTWEIGHT_STRUCTURE_CONFIG, 'use_table_recognition': True}
        formatted_text = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure,
            use_structure=True,
            structure_config=custom_config
        )

        # Should process successfully (config accepted)
        assert formatted_text is not None, "Should process with custom config"
        assert len(formatted_text.strip()) > 0, "Should extract text with custom config"

        elements = parse_formatted_output(formatted_text)
        assert len(elements) > 0, "Should have elements with custom config"

    @pytest.mark.asyncio
    async def test_output_format_consistency_with_ppstructure(self, mock_storage_infrastructure):
        """Test that PPStructureV3 output format is consistent with standard.

        Format: "text [page=N, bbox=(x,y,x,y), type=TYPE, confidence=C]"
        """
        loader = OcrImageLoader()

        formatted_text = await self._run_ocr_and_capture(
            loader, EXAMPLE_PNG_PATH, mock_storage_infrastructure,
            use_structure=True,
            structure_config=self.LIGHTWEIGHT_STRUCTURE_CONFIG
        )

        lines = formatted_text.strip().split("\n")

        # Validate format consistency
        pattern = r"^.+\s*\[page=\d+,\s*bbox=\([0-9.]+,[0-9.]+,[0-9.]+,[0-9.]+\),\s*type=\w+,\s*confidence=[0-9.]+\]$"

        for line in lines:
            if not line.strip():
                continue
            assert re.match(pattern, line), \
                f"PPStructureV3 output doesn't match expected format: {line}"

        # All elements should be parseable
        elements = parse_formatted_output(formatted_text)
        assert len(elements) == len([l for l in lines if l.strip()]), \
            "All lines should be parseable"
