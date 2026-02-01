# PPStructureV3 Layout Region Validation Implementation

## Summary

Successfully enhanced the PPStructureV3 test to validate and visualize **layout region boxes** in addition to OCR text boxes. This addresses the missing validation of PPStructureV3's core layout detection feature.

## Implementation Date
2026-02-01

## Changes Made

### 1. New Visualization Function: `visualize_layout_regions()`

**File**: `cognee/modules/visualization/ocr_visualization.py`

Added a new reusable function that visualizes layout region boxes detected by PPStructureV3:

```python
async def visualize_layout_regions(
    ocr_result: Union[OCRPageResult, OCRDocumentResult],
    image_path: str,
    output_path: Optional[str] = None,
    show_labels: bool = True,
    box_width: int = 3,
    color_map: Optional[dict] = None,
) -> str
```

**Features:**
- Draws layout region boxes (paragraphs, titles, figures, tables, etc.)
- Color-coded by layout type (blue=text, red=title, green=figure, etc.)
- Labels showing layout type names
- Supports custom color maps
- Works with both single-page and multi-page OCR results
- Validates that `use_structure=True` was used

**Location**: Lines 140-263

### 2. Export New Function

**File**: `cognee/modules/visualization/__init__.py`

Exported `visualize_layout_regions` for public use:

```python
from cognee.modules.visualization.ocr_visualization import (
    visualize_ocr,
    visualize_layout_regions,  # NEW
)

__all__ = ["cognee_network_visualization", "visualize_ocr", "visualize_layout_regions"]
```

### 3. Enhanced Test Validation

**File**: `cognee/tests/unit/loaders/test_ocr_image_loader_real.py`

**Test**: `TestPPStructureV3RealProcessing::test_ppstructure_basic_processing`

**Added Section 4b**: Layout Region Box Validation (lines 1213-1233)
- Validates `page_result.layout_info` exists
- Validates `layout_boxes` has at least one region
- Validates each box has:
  - `coordinate` field (4-element list)
  - `label` field (layout type)
  - Numeric coordinates (handles numpy types)
- Collects all detected layout types

**Enhanced Section 5**: Print Results (lines 1235-1240)
- Shows count of layout regions detected
- Lists layout types found (e.g., 'doc_title', 'text', 'figure')
- Displays sample layout boxes for inspection
- Shows OCR text element count and types

**Added Section 7**: Layout Region Visualization (lines 1265-1283)
- Uses new `visualize_layout_regions()` function
- Saves visualization to `.test_artifacts/ppstructure_layout_regions.png`
- Shows document structure regions (not individual text lines)
- Clear output messaging to distinguish from OCR text visualization

## Test Results

### Validation Output
```
PPStructureV3 Layout Detection Results:
  Layout regions detected: 1
  Layout types found: {'doc_title'}
  Sample layout boxes: [{'cls_id': 11, 'label': 'doc_title', 'score': 0.84061598777771, 'coordinate': [np.float32(0.16012573), np.float32(19.95082), np.float32(487.54507), np.float32(182.31195)]}]
  OCR text elements: 4
  OCR layout types: {'text'}
```

### Visualizations Generated
1. **ppstructure_visualization.png** - OCR text boxes (existing)
2. **ppstructure_layout_regions.png** - Layout region boxes (NEW)

### Test Status
✅ **PASSED** (280.52 seconds)

## Key Technical Details

### Layout Box Data Structure
```python
{
    'coordinate': [x1, y1, x2, y2],  # numpy.float32 types
    'label': 'text' | 'title' | 'figure' | 'table' | ...,
    'cls_id': int,
    'score': float  # confidence
}
```

### Coordinate Type Handling
Layout box coordinates are **numpy.float32** types, not Python floats. The validation converts them:

```python
coords_numeric = [float(c) for c in coords]
```

### Color Map (Default)
```python
{
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
```

## Benefits

### Testing Improvements
✅ Validates PPStructureV3's **actual layout detection output** (previously untested)
✅ Tests both layout regions AND OCR text boxes
✅ Visual verification of layout detection quality
✅ Proper assertions for layout_info structure and content

### Code Quality
✅ Reusable visualization function for all PPStructureV3 users
✅ Clear documentation and examples
✅ Demonstrates proper usage of layout detection API
✅ Better debugging capabilities

### User Value
✅ Shows how to access and use layout boxes
✅ Provides template for custom layout visualizations
✅ Distinguishes between layout regions and OCR text boxes
✅ Testing pattern for layout-aware features

## Usage Example

```python
from cognee.infrastructure.ocr import PaddleOCRAdapter
from cognee.modules.visualization import visualize_layout_regions

# Run PPStructureV3 with layout detection
adapter = PaddleOCRAdapter(use_structure=True)
page_result = await adapter.process_image("document.png")

# Visualize layout regions
viz_path = await visualize_layout_regions(
    page_result,
    image_path="document.png",
    show_labels=True,
    box_width=3
)

print(f"Layout visualization saved to: {viz_path}")

# Access layout boxes programmatically
layout_boxes = page_result.layout_info['layout_boxes']
for box in layout_boxes:
    print(f"Region: {box['label']} at {box['coordinate']}")
```

## Files Modified

1. `/Users/andikarachman/personal_projects/kg/cognee/cognee/modules/visualization/ocr_visualization.py`
   - Added `visualize_layout_regions()` function (~124 lines)

2. `/Users/andikarachman/personal_projects/kg/cognee/cognee/modules/visualization/__init__.py`
   - Exported new function

3. `/Users/andikarachman/personal_projects/kg/cognee/cognee/tests/unit/loaders/test_ocr_image_loader_real.py`
   - Added layout region validation (~20 lines)
   - Added layout region visualization (~19 lines)
   - Enhanced output messaging

**Total Code Added**: ~163 lines

## Verification

Run the test:
```bash
source .venv-arm64/bin/activate
pytest cognee/tests/unit/loaders/test_ocr_image_loader_real.py::TestPPStructureV3RealProcessing::test_ppstructure_basic_processing -v -s
```

Check visualizations:
```bash
ls -lh cognee/tests/unit/loaders/.test_artifacts/ppstructure*.png
open cognee/tests/unit/loaders/.test_artifacts/ppstructure_layout_regions.png
```

## Next Steps

### Potential Enhancements
1. Add support for multi-page layout visualization
2. Create side-by-side comparison visualization (layout vs OCR)
3. Add statistics output (average region size, distribution of layout types)
4. Support for custom layout type hierarchies
5. Interactive visualization with hover labels

### Documentation
- Add example to main documentation
- Update OCR guide to show layout detection usage
- Create tutorial notebook for layout-aware processing

## Conclusion

The implementation successfully validates PPStructureV3's layout detection output and provides a reusable visualization tool. The test now properly exercises the core feature of PPStructureV3 (layout region detection) rather than just OCR text extraction.
