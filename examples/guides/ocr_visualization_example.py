"""OCR Visualization Examples

This script demonstrates how to use the visualize_ocr() function to create
annotated images showing OCR results with bounding boxes, text labels, and
confidence scores.

Requirements:
    pip install cognee[ocr]

Color Coding:
    - Green boxes: High confidence (>0.8)
    - Orange boxes: Medium confidence (0.5-0.8)
    - Red boxes: Low confidence (<0.5)
"""

import asyncio
from pathlib import Path
from cognee import visualize_ocr
from cognee.infrastructure.ocr import PaddleOCRAdapter


async def example_1_simple_visualization():
    """Example 1: Visualize from file path (automatic OCR)."""
    print("Example 1: Simple visualization")
    print("=" * 50)

    # This will automatically run OCR and create an annotated image
    # Output will be saved to ~/ocr_visualization_example.png
    result_path = await visualize_ocr("example.png")

    print(f"✓ Visualization saved to: {result_path}")
    print()


async def example_2_custom_settings():
    """Example 2: Custom settings."""
    print("Example 2: Custom settings")
    print("=" * 50)

    result_path = await visualize_ocr(
        "example.png",
        output_path="annotated_example.png",  # Custom output path
        min_confidence=0.6,                    # Filter low confidence results
        show_text=True,                        # Show extracted text
        show_confidence=True,                  # Show confidence scores
        box_width=3                            # Thicker bounding boxes
    )

    print(f"✓ Saved to: {result_path}")
    print(f"  - Only showing text with confidence >= 0.6")
    print(f"  - Text labels and confidence scores visible")
    print(f"  - Thicker bounding boxes (3px)")
    print()


async def example_3_reuse_ocr_results():
    """Example 3: Reuse existing OCR results."""
    print("Example 3: Reuse OCR results")
    print("=" * 50)

    # First, run OCR separately (useful if you want to inspect results first)
    adapter = PaddleOCRAdapter(min_confidence=0.5)
    ocr_result = await adapter.process_image("example.png")

    print(f"OCR found {len(ocr_result.elements)} text elements")

    # Now visualize without re-running OCR
    result_path = await visualize_ocr(
        ocr_result,
        image_path="example.png",  # Must provide original image path
        output_path="output.png"
    )

    print(f"✓ Saved to: {result_path}")
    print()


async def example_4_multipage_pdf():
    """Example 4: Multi-page PDF."""
    print("Example 4: Multi-page PDF")
    print("=" * 50)

    # For PDFs, visualize_ocr creates one annotated image per page
    pdf_paths = await visualize_ocr(
        "document.pdf",
        output_path="./ocr_results/",  # Directory for output images
        min_confidence=0.7
    )

    print(f"✓ Generated {len(pdf_paths)} annotated pages")
    for i, path in enumerate(pdf_paths, 1):
        print(f"  Page {i}: {path}")
    print()


async def example_5_create_test_image():
    """Example 5: Create a test image and visualize it."""
    print("Example 5: Create test image and visualize")
    print("=" * 50)

    from PIL import Image, ImageDraw, ImageFont

    # Create a sample image with text
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)

    try:
        # Try to use a larger font if available
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Add some text
    draw.text((50, 50), "OCR Visualization Demo", fill='black', font=font_large)
    draw.text((50, 120), "This text should be detected", fill='black', font=font_small)
    draw.text((50, 160), "With high confidence (clear text)", fill='blue', font=font_small)
    draw.text((50, 250), "Multiple lines of text", fill='green', font=font_small)
    draw.text((50, 290), "Will be annotated with boxes", fill='red', font=font_small)

    # Save test image
    test_image_path = "test_ocr_demo.png"
    img.save(test_image_path)
    print(f"✓ Created test image: {test_image_path}")

    # Visualize it
    result_path = await visualize_ocr(
        test_image_path,
        output_path="test_ocr_demo_annotated.png",
        min_confidence=0.5,
        show_text=True,
        show_confidence=True
    )

    print(f"✓ Annotated image saved to: {result_path}")
    print(f"  Open both images to compare:")
    print(f"  - Original: {test_image_path}")
    print(f"  - Annotated: {result_path}")
    print()


async def example_6_batch_processing():
    """Example 6: Batch process multiple images."""
    print("Example 6: Batch processing")
    print("=" * 50)

    image_files = [
        "image1.png",
        "image2.png",
        "scan.jpg",
    ]

    results = []
    for image_file in image_files:
        if Path(image_file).exists():
            result = await visualize_ocr(
                image_file,
                min_confidence=0.6
            )
            results.append(result)
            print(f"✓ Processed: {image_file} -> {result}")

    print(f"\nTotal processed: {len(results)} images")
    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("OCR VISUALIZATION EXAMPLES")
    print("=" * 60 + "\n")

    # Run examples that don't require specific files
    await example_5_create_test_image()

    # Uncomment to run other examples (requires files to exist)
    # await example_1_simple_visualization()
    # await example_2_custom_settings()
    # await example_3_reuse_ocr_results()
    # await example_4_multipage_pdf()
    # await example_6_batch_processing()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
