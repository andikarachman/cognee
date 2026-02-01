"""OCR result visualization with bounding boxes and text labels."""

from typing import Union, Optional, List, Tuple
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

from cognee.infrastructure.ocr import (
    OCRPageResult,
    OCRDocumentResult,
    PaddleOCRAdapter,
    is_paddleocr_available
)

# Confidence-based colors
HIGH_CONFIDENCE = (0, 255, 0)      # Green (>0.8)
MEDIUM_CONFIDENCE = (255, 165, 0)  # Orange (0.5-0.8)
LOW_CONFIDENCE = (255, 0, 0)       # Red (<0.5)


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Map confidence score to color (green/orange/red)."""
    if confidence > 0.8:
        return HIGH_CONFIDENCE
    elif confidence >= 0.5:
        return MEDIUM_CONFIDENCE
    else:
        return LOW_CONFIDENCE


def truncate_text(text: str, max_length: int = 30) -> str:
    """Truncate text for display on image."""
    return text[:max_length] + "..." if len(text) > max_length else text


def generate_output_path(source_path: str, page_num: Optional[int] = None) -> str:
    """Generate output path in home directory."""
    # Pattern from cognee_network_visualization.py:688-690
    home_dir = os.path.expanduser("~")
    base_name = Path(source_path).stem

    if page_num is not None:
        filename = f"ocr_visualization_{base_name}_page_{page_num}.png"
    else:
        filename = f"ocr_visualization_{base_name}.png"

    return os.path.join(home_dir, filename)


async def visualize_ocr_page(
    image_path: str,
    ocr_result: OCRPageResult,
    output_path: Optional[str] = None,
    min_confidence: float = 0.0,
    show_text: bool = True,
    show_confidence: bool = True,
    box_width: int = 2
) -> str:
    """
    Draw bounding boxes and text on image to visualize OCR results.

    Args:
        image_path: Path to source image
        ocr_result: OCR result containing text elements and bboxes
        output_path: Where to save annotated image (auto-generated if None)
        min_confidence: Only show elements above this confidence
        show_text: Draw text labels on boxes
        show_confidence: Show confidence scores
        box_width: Width of bounding box lines

    Returns:
        Absolute path to saved annotated image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to load default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw each text element
    for element in ocr_result.elements:
        if element.confidence < min_confidence:
            continue

        bbox = element.bbox

        # Use pixel coordinates (already computed in BoundingBox)
        box_coords = [
            bbox.pixel_x_min,
            bbox.pixel_y_min,
            bbox.pixel_x_max,
            bbox.pixel_y_max
        ]

        # Color based on confidence
        color = get_confidence_color(element.confidence)

        # Draw bounding box
        draw.rectangle(box_coords, outline=color, width=box_width)

        if show_text:
            # Draw text label above box
            text = truncate_text(element.text)
            text_x = bbox.pixel_x_min
            text_y = max(0, bbox.pixel_y_min - 20)

            # Text background for readability
            if font:
                text_bbox = draw.textbbox((text_x, text_y), text, font=font)
            else:
                text_bbox = [text_x, text_y, text_x + len(text) * 8, text_y + 15]

            draw.rectangle(text_bbox, fill=(0, 0, 0, 200))
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        if show_confidence:
            # Draw confidence score
            conf_text = f"{element.confidence:.2f}"
            conf_x = bbox.pixel_x_max - 50
            conf_y = bbox.pixel_y_min + 5
            draw.text((conf_x, conf_y), conf_text, fill=color, font=font)

    # Generate output path if not provided
    if output_path is None:
        output_path = generate_output_path(image_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save annotated image
    img.save(output_path)

    return os.path.abspath(output_path)


async def visualize_layout_regions(
    ocr_result: Union[OCRPageResult, OCRDocumentResult],
    image_path: str,
    output_path: Optional[str] = None,
    show_labels: bool = True,
    box_width: int = 3,
    color_map: Optional[dict] = None,
) -> str:
    """
    Visualize layout region boxes from PPStructureV3.

    This function draws the structural layout regions (paragraphs, titles, figures, etc.)
    detected by PPStructureV3's layout detection model, NOT individual OCR text boxes.

    Args:
        ocr_result: OCRPageResult or OCRDocumentResult with layout_info
        image_path: Path to original image file
        output_path: Where to save visualization (auto-generated if None)
        show_labels: Whether to show layout type labels on boxes
        box_width: Width of bounding box lines in pixels
        color_map: Custom colors for layout types (optional)

    Returns:
        Absolute path to saved visualization image

    Raises:
        ValueError: If ocr_result doesn't contain layout_info (use_structure=False)

    Example:
        >>> from cognee.infrastructure.ocr import PaddleOCRAdapter
        >>> adapter = PaddleOCRAdapter(use_structure=True)
        >>> page_result = await adapter.process_image("document.png")
        >>> viz_path = await visualize_layout_regions(
        ...     page_result,
        ...     image_path="document.png",
        ...     show_labels=True
        ... )
        >>> print(f"Layout visualization saved to: {viz_path}")
    """
    # Default color map for layout types
    if color_map is None:
        color_map = {
            'text': 'blue',
            'title': 'red',
            'heading': 'orange',
            'paragraph': 'cyan',
            'figure': 'green',
            'table': 'purple',
            'list': 'yellow',
            'caption': 'pink',
            'header': 'magenta',
            'footer': 'brown',
        }

    # Handle both single page and multi-page
    if hasattr(ocr_result, 'pages'):  # OCRDocumentResult
        page_result = ocr_result.pages[0]  # Use first page for now
    else:  # OCRPageResult
        page_result = ocr_result

    # Check for layout info
    if not page_result.layout_info or not page_result.layout_info.get('layout_boxes'):
        raise ValueError(
            "OCR result does not contain layout_info. "
            "Ensure use_structure=True when running PaddleOCRAdapter."
        )

    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to load font for labels
    try:
        # Try system fonts
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        try:
            # Fallback for Linux
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            # Last resort: default font
            font = ImageFont.load_default()

    # Draw layout boxes
    layout_boxes = page_result.layout_info['layout_boxes']
    for box in layout_boxes:
        coords = box['coordinate']
        label = box.get('label', 'unknown')
        color = color_map.get(label, 'yellow')

        # Draw rectangle
        draw.rectangle(
            [(coords[0], coords[1]), (coords[2], coords[3])],
            outline=color,
            width=box_width
        )

        # Draw label if requested
        if show_labels:
            label_text = f"{label}"
            text_x = coords[0]
            text_y = max(0, coords[1] - 20)

            # Get text bounding box for background
            try:
                text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
            except Exception:
                # Fallback if textbbox not available
                text_bbox = [text_x, text_y, text_x + len(label_text) * 10, text_y + 16]

            # Draw background rectangle for label
            draw.rectangle(text_bbox, fill=color)
            draw.text((text_x, text_y), label_text, fill='white', font=font)

    # Generate output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_layout_regions.png"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save and return
    img.save(output_path)
    return os.path.abspath(output_path)


async def visualize_ocr_document(
    pdf_path: str,
    ocr_result: OCRDocumentResult,
    output_dir: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Visualize multi-page PDF OCR results.

    Args:
        pdf_path: Path to source PDF
        ocr_result: OCR result for all pages
        output_dir: Directory to save images (home dir if None)
        **kwargs: Additional arguments for visualize_ocr_page

    Returns:
        List of paths to saved annotated images (one per page)
    """
    from pdf2image import convert_from_path
    import tempfile

    # Convert PDF to images (pattern from PaddleOCRAdapter.process_pdf)
    images = convert_from_path(pdf_path)

    output_paths = []

    for page_idx, (page_result, page_image) in enumerate(zip(ocr_result.pages, images)):
        # Save page to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = temp_file.name
            page_image.save(temp_path, "PNG")

        # Generate output path for this page
        if output_dir:
            output_path = os.path.join(
                output_dir,
                f"{Path(pdf_path).stem}_page_{page_idx + 1}.png"
            )
        else:
            output_path = generate_output_path(pdf_path, page_idx + 1)

        # Visualize page
        result_path = await visualize_ocr_page(
            temp_path,
            page_result,
            output_path=output_path,
            **kwargs
        )

        output_paths.append(result_path)

        # Clean up temp file
        os.unlink(temp_path)

    return output_paths


async def visualize_ocr(
    source: Union[str, OCRPageResult, OCRDocumentResult],
    output_path: Optional[str] = None,
    min_confidence: float = 0.0,
    show_text: bool = True,
    show_confidence: bool = True,
    box_width: int = 2,
    use_structure: bool = False,
    structure_config: Optional[dict] = None,
    **kwargs
) -> Union[str, List[str]]:
    """
    Visualize OCR results with bounding boxes and text labels on images.

    This is the main public API for OCR visualization. Accepts either:
    - File path (runs OCR automatically)
    - OCRPageResult object (single page)
    - OCRDocumentResult object (multi-page)

    Args:
        source: Image/PDF path OR OCR result object
        output_path: Where to save (auto-generated if None)
        min_confidence: Filter elements below this confidence (0-1)
        show_text: Display extracted text on image
        show_confidence: Display confidence scores
        box_width: Width of bounding box lines in pixels
        use_structure: Whether to use PPStructureV3 for layout-aware OCR (default: False)
        structure_config: Optional configuration dict for PPStructureV3 (default: None)
        **kwargs: Additional visualization parameters

    Returns:
        Path to saved image (str) or list of paths (List[str]) for PDFs

    Raises:
        ImportError: If PaddleOCR not installed
        FileNotFoundError: If source file doesn't exist

    Example:
        >>> # Visualize from file path (automatic OCR)
        >>> result = await visualize_ocr("document.png")
        >>> print(f"Saved to: {result}")

        >>> # Visualize with custom settings
        >>> result = await visualize_ocr(
        ...     "scan.pdf",
        ...     min_confidence=0.7,
        ...     output_path="output.png"
        ... )

        >>> # Reuse existing OCR results
        >>> from cognee.infrastructure.ocr import PaddleOCRAdapter
        >>> adapter = PaddleOCRAdapter()
        >>> ocr_result = await adapter.process_image("image.png")
        >>> viz_path = await visualize_ocr(ocr_result)
    """
    # Check if PaddleOCR is available
    if not is_paddleocr_available():
        raise ImportError(
            "PaddleOCR is required for OCR visualization. "
            "Install with: pip install cognee[ocr]"
        )

    # Handle different input types
    if isinstance(source, str):
        # File path - run OCR first
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        # Run OCR
        adapter = PaddleOCRAdapter(
            min_confidence=min_confidence,
            use_structure=use_structure,
            structure_config=structure_config
        )

        if source_path.suffix.lower() == '.pdf':
            # Multi-page PDF
            ocr_result = await adapter.process_pdf(str(source_path))
            return await visualize_ocr_document(
                str(source_path),
                ocr_result,
                output_dir=output_path,
                min_confidence=min_confidence,
                show_text=show_text,
                show_confidence=show_confidence,
                box_width=box_width,
                **kwargs
            )
        else:
            # Single image
            ocr_result = await adapter.process_image(str(source_path))
            return await visualize_ocr_page(
                str(source_path),
                ocr_result,
                output_path=output_path,
                min_confidence=min_confidence,
                show_text=show_text,
                show_confidence=show_confidence,
                box_width=box_width,
                **kwargs
            )

    elif isinstance(source, OCRPageResult):
        # Single page OCR result - need original image path
        # For now, require passing via kwargs
        image_path = kwargs.get('image_path')
        if not image_path:
            raise ValueError(
                "When passing OCRPageResult, must provide 'image_path' in kwargs"
            )

        return await visualize_ocr_page(
            image_path,
            source,
            output_path=output_path,
            min_confidence=min_confidence,
            show_text=show_text,
            show_confidence=show_confidence,
            box_width=box_width
        )

    elif isinstance(source, OCRDocumentResult):
        # Multi-page document result
        pdf_path = kwargs.get('pdf_path')
        if not pdf_path:
            raise ValueError(
                "When passing OCRDocumentResult, must provide 'pdf_path' in kwargs"
            )

        return await visualize_ocr_document(
            pdf_path,
            source,
            output_dir=output_path,
            min_confidence=min_confidence,
            show_text=show_text,
            show_confidence=show_confidence,
            box_width=box_width
        )

    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. "
            "Expected str (file path), OCRPageResult, or OCRDocumentResult"
        )
