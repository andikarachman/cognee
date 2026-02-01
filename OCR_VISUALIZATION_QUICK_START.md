# OCR Visualization Quick Start Guide

## Installation

```bash
pip install cognee[ocr]
```

## Basic Usage

### 1. Simple Visualization (Auto OCR)

```python
import asyncio
from cognee import visualize_ocr

async def main():
    # Automatically runs OCR and creates annotated image
    result = await visualize_ocr("document.png")
    print(f"Saved to: {result}")

asyncio.run(main())
```

**Output**: `~/ocr_visualization_document.png`

### 2. Custom Settings

```python
result = await visualize_ocr(
    "document.png",
    output_path="annotated.png",    # Custom output path
    min_confidence=0.7,              # Filter low confidence
    show_text=True,                  # Show text labels
    show_confidence=True,            # Show scores
    box_width=3                      # Thicker boxes
)
```

### 3. Multi-Page PDF

```python
# Returns list of paths (one per page)
pages = await visualize_ocr("document.pdf")

for i, path in enumerate(pages, 1):
    print(f"Page {i}: {path}")
```

### 4. Reuse OCR Results

```python
from cognee.infrastructure.ocr import PaddleOCRAdapter

# Run OCR once
adapter = PaddleOCRAdapter(min_confidence=0.5)
ocr_result = await adapter.process_image("document.png")

# Visualize without re-running OCR
result = await visualize_ocr(
    ocr_result,
    image_path="document.png"
)
```

## Color Coding

Bounding boxes are color-coded by confidence:

- 🟢 **Green**: High confidence (>0.8)
- 🟠 **Orange**: Medium confidence (0.5-0.8)
- 🔴 **Red**: Low confidence (<0.5)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str/OCRResult | - | File path or OCR result |
| `output_path` | str/None | None | Where to save (auto if None) |
| `min_confidence` | float | 0.0 | Filter threshold (0-1) |
| `show_text` | bool | True | Display extracted text |
| `show_confidence` | bool | True | Display confidence scores |
| `box_width` | int | 2 | Bounding box line width |

## Example Output

When you run:
```python
await visualize_ocr("example.png", min_confidence=0.6)
```

You get an annotated image with:
- ✓ Color-coded bounding boxes around text
- ✓ Text labels showing extracted content
- ✓ Confidence scores (e.g., "0.95")
- ✓ High confidence = green, medium = orange, low = red

## Common Use Cases

### Debugging OCR
```python
# See what OCR detected with low confidence
result = await visualize_ocr(
    "scan.png",
    min_confidence=0.3,
    show_confidence=True
)
```

### Quality Control
```python
# Only show high-confidence detections
result = await visualize_ocr(
    "document.png",
    min_confidence=0.9
)
```

### Batch Processing
```python
import glob

for image_path in glob.glob("*.png"):
    result = await visualize_ocr(image_path)
    print(f"Processed: {image_path}")
```

## Run Example

```bash
python examples/guides/ocr_visualization_example.py
```

## Running Tests

```bash
# Unit tests
pytest cognee/tests/unit/modules/visualization/test_ocr_visualization.py -v

# Integration tests
pytest cognee/tests/integration/test_ocr_visualization.py -v
```

## Troubleshooting

### PaddleOCR Not Found
```bash
pip install cognee[ocr]
```

### File Not Found
Make sure the file path exists:
```python
from pathlib import Path
assert Path("document.png").exists()
```

### OCRResult Without Image Path
When passing OCR results, include `image_path`:
```python
await visualize_ocr(ocr_result, image_path="original.png")
```

## More Information

- Full documentation: See `OCR_VISUALIZATION_IMPLEMENTATION.md`
- Examples: See `examples/guides/ocr_visualization_example.py`
- Tests: See `cognee/tests/integration/test_ocr_visualization.py`
