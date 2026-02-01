"""Diagnostic test for OCR bounding box coordinate investigation."""

import pytest
from pathlib import Path
from PIL import Image
from cognee.infrastructure.ocr import PaddleOCRAdapter

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "test_data"
EXAMPLE_PNG_PATH = str(TEST_DATA_DIR / "example.png")


@pytest.mark.asyncio
async def test_diagnose_bbox_coordinates():
    """Diagnostic test to print OCR coordinates and image info."""

    # Get image dimensions
    with Image.open(EXAMPLE_PNG_PATH) as img:
        img_width, img_height = img.size

    print("\n" + "="*80)
    print(f"Image: {EXAMPLE_PNG_PATH}")
    print(f"Dimensions: {img_width} x {img_height}")
    print("="*80)

    # Run OCR
    adapter = PaddleOCRAdapter(min_confidence=0.5)
    ocr_result = await adapter.process_image(EXAMPLE_PNG_PATH)

    print(f"\nTotal elements found: {len(ocr_result.elements)}")

    # Print each element's coordinates
    for idx, element in enumerate(ocr_result.elements):
        bbox = element.bbox
        print(f"\nElement {idx + 1}: '{element.text[:50]}'")
        print(f"  Confidence: {element.confidence:.3f}")
        print(f"  Normalized coords: ({bbox.x_min:.4f}, {bbox.y_min:.4f}) to ({bbox.x_max:.4f}, {bbox.y_max:.4f})")
        print(f"  Pixel coords: ({bbox.pixel_x_min}, {bbox.pixel_y_min}) to ({bbox.pixel_x_max}, {bbox.pixel_y_max})")

        # Check if pixel coords match normalized coords
        expected_px_min = int(bbox.x_min * img_width)
        expected_py_min = int(bbox.y_min * img_height)
        expected_px_max = int(bbox.x_max * img_width)
        expected_py_max = int(bbox.y_max * img_height)

        print(f"  Expected pixel coords: ({expected_px_min}, {expected_py_min}) to ({expected_px_max}, {expected_py_max})")

        # Check for mismatches
        if (bbox.pixel_x_min != expected_px_min or
            bbox.pixel_y_min != expected_py_min or
            bbox.pixel_x_max != expected_px_max or
            bbox.pixel_y_max != expected_py_max):
            print(f"  ⚠️  MISMATCH DETECTED!")
            print(f"     X_min diff: {bbox.pixel_x_min - expected_px_min}")
            print(f"     Y_min diff: {bbox.pixel_y_min - expected_py_min}")
            print(f"     X_max diff: {bbox.pixel_x_max - expected_px_max}")
            print(f"     Y_max diff: {bbox.pixel_y_max - expected_py_max}")

        # Check bounds
        if (bbox.pixel_x_min < 0 or bbox.pixel_x_max > img_width or
            bbox.pixel_y_min < 0 or bbox.pixel_y_max > img_height):
            print(f"  ❌ OUT OF BOUNDS!")

        # Check box dimensions
        box_width = bbox.pixel_x_max - bbox.pixel_x_min
        box_height = bbox.pixel_y_max - bbox.pixel_y_min
        print(f"  Box dimensions: {box_width} x {box_height} pixels")

    print("\n" + "="*80 + "\n")

    # Basic assertions
    assert len(ocr_result.elements) > 0, "Should detect at least one text element"
    assert ocr_result.page_width == img_width, "Page width should match image width"
    assert ocr_result.page_height == img_height, "Page height should match image height"


@pytest.mark.asyncio
async def test_visualize_with_coordinate_debug():
    """Create visualization with coordinate debugging overlays."""
    from cognee.modules.visualization.ocr_visualization import visualize_ocr_page
    from PIL import ImageDraw, ImageFont
    import tempfile

    # Run OCR
    adapter = PaddleOCRAdapter(min_confidence=0.5)
    ocr_result = await adapter.process_image(EXAMPLE_PNG_PATH)

    # Create enhanced visualization with coordinate labels
    with Image.open(EXAMPLE_PNG_PATH) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Try to use a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        # Draw each bounding box with coordinate labels
        for idx, element in enumerate(ocr_result.elements):
            bbox = element.bbox

            # Draw the bounding box in red
            box_coords = [
                bbox.pixel_x_min,
                bbox.pixel_y_min,
                bbox.pixel_x_max,
                bbox.pixel_y_max
            ]
            draw.rectangle(box_coords, outline="red", width=2)

            # Draw crosshairs at corners
            crosshair_size = 5
            # Top-left corner
            draw.line(
                [(bbox.pixel_x_min - crosshair_size, bbox.pixel_y_min),
                 (bbox.pixel_x_min + crosshair_size, bbox.pixel_y_min)],
                fill="blue", width=1
            )
            draw.line(
                [(bbox.pixel_x_min, bbox.pixel_y_min - crosshair_size),
                 (bbox.pixel_x_min, bbox.pixel_y_min + crosshair_size)],
                fill="blue", width=1
            )

            # Bottom-right corner
            draw.line(
                [(bbox.pixel_x_max - crosshair_size, bbox.pixel_y_max),
                 (bbox.pixel_x_max + crosshair_size, bbox.pixel_y_max)],
                fill="blue", width=1
            )
            draw.line(
                [(bbox.pixel_x_max, bbox.pixel_y_max - crosshair_size),
                 (bbox.pixel_x_max, bbox.pixel_y_max + crosshair_size)],
                fill="blue", width=1
            )

            # Add coordinate label above the box
            coord_label = f"({bbox.pixel_x_min},{bbox.pixel_y_min})-({bbox.pixel_x_max},{bbox.pixel_y_max})"
            label_y = max(0, bbox.pixel_y_min - 15)
            draw.text((bbox.pixel_x_min, label_y), coord_label, fill="green", font=font)

        # Save debug visualization
        output_path = Path(tempfile.gettempdir()) / "ocr_bbox_debug.png"
        img.save(output_path)
        print(f"\n📊 Debug visualization saved to: {output_path}")
        print(f"   Open this file to visually inspect bounding box alignment\n")

    assert output_path.exists(), "Debug visualization should be created"
