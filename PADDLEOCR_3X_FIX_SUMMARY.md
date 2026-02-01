# PaddleOCR 3.x Compatibility Fix - Implementation Summary

## Problem

The `PaddleOCRAdapter` was hardcoded to use PaddleOCR 2.x API, which caused multiple errors when using PaddleOCR 3.x:

1. **`TypeError: Unknown argument: use_gpu`** - PaddleOCR 3.x removed the `use_gpu` parameter
2. **`TypeError: Unknown argument: show_log`** - PaddleOCR 3.x removed the `show_log` parameter
3. **`TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'`** - PaddleOCR 3.x changed from `ocr()` to `predict()` method
4. **Different result format** - PaddleOCR 3.x returns `OCRResult` objects instead of nested lists

## Solution

Implemented version detection and conditional API usage in `cognee/infrastructure/ocr/PaddleOCRAdapter.py` to support both PaddleOCR 2.x and 3.x.

### Changes Made

#### 1. Added Version Detection (Lines 100-110)

```python
# Determine device parameter based on PaddleOCR version
# PaddleOCR 3.x uses 'device' parameter, 2.x uses 'use_gpu'
try:
    version = paddleocr.__version__
    major_version = int(version.split('.')[0])
except (AttributeError, ValueError, IndexError):
    # Default to version 3 behavior if version detection fails
    major_version = 3

# Store version for later use
self._paddleocr_version = major_version
```

#### 2. Version-Specific Initialization (Lines 112-127)

**PaddleOCR 3.x:**
- Uses `device` parameter: `"gpu"` or `"cpu"`
- Removed `show_log` parameter (not supported)

```python
if major_version >= 3:
    # PaddleOCR 3.x: Use 'device' parameter, no 'show_log' parameter
    device = "gpu" if self.use_gpu else "cpu"
    logger.info(f"Initializing PaddleOCR v{major_version}.x with lang={self.lang}, device={device}")
    self._ocr_engine = PaddleOCR(
        lang=self.lang,
        device=device,
    )
```

**PaddleOCR 2.x (Legacy):**
- Uses `use_gpu` parameter: `True` or `False`
- Supports `show_log` parameter

```python
else:
    # PaddleOCR 2.x: Use 'use_gpu' parameter (legacy)
    logger.info(f"Initializing PaddleOCR v{major_version}.x with lang={self.lang}, use_gpu={self.use_gpu}")
    self._ocr_engine = PaddleOCR(
        lang=self.lang,
        use_gpu=self.use_gpu,
        show_log=False,
    )
```

#### 3. Version-Specific OCR Processing (process_image method)

**PaddleOCR 3.x API:**
- Uses `predict()` method instead of `ocr()`
- Returns `OCRResult` object with dict-like structure
- Extract results from: `rec_texts`, `rec_scores`, `rec_polys`

```python
if self._paddleocr_version and self._paddleocr_version >= 3:
    # PaddleOCR 3.x API
    result = ocr_engine.predict(image_path)
    if result and len(result) > 0:
        ocr_result = result[0]
        # Extract texts, scores, and bounding boxes
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [])
        rec_polys = ocr_result.get('rec_polys', ocr_result.get('dt_polys', []))

        for i in range(len(rec_texts)):
            text = rec_texts[i]
            confidence = float(rec_scores[i]) if i < len(rec_scores) else 1.0
            bbox_coords = rec_polys[i] if i < len(rec_polys) else None
            # ... process bbox and create OCRTextElement
```

**PaddleOCR 2.x API (Legacy):**
- Uses `ocr()` method with `cls=True` parameter
- Returns nested list structure: `[[bbox, [text, confidence]], ...]`

```python
else:
    # PaddleOCR 2.x API
    result = ocr_engine.ocr(image_path, cls=True)
    if result and result[0]:
        for line in result[0]:
            bbox_coords = line[0]
            text_info = line[1]
            text = text_info[0]
            confidence = text_info[1]
            # ... process bbox and create OCRTextElement
```

#### 4. Removed Test Workaround

Removed the `patched_paddleocr` fixture from `test_ocr_image_loader_real.py` since the production code now handles version compatibility correctly.

### Files Modified

1. **`cognee/infrastructure/ocr/PaddleOCRAdapter.py`**
   - Added `_paddleocr_version` instance variable
   - Implemented version detection in `_get_ocr_engine()`
   - Added version-specific initialization logic
   - Updated `process_image()` to handle both API versions

2. **`cognee/tests/unit/loaders/test_ocr_image_loader_real.py`**
   - Removed `patched_paddleocr` fixture (lines 25-49)
   - Removed all `patched_paddleocr` parameters from 14 test methods

3. **`cognee/tests/unit/loaders/README_OCR_TESTS.md`**
   - Updated "Known Issues & Workarounds" section
   - Added "Supported PaddleOCR Versions" section

## Test Results

All 14 OCR tests now pass with PaddleOCR 3.3.3:

```bash
$ pytest cognee/tests/unit/loaders/test_ocr_image_loader_real.py -v

============ 14 passed, 1 skipped, 8 warnings in 120.54s (0:02:00) =============
```

**Passed Tests:**
- ✅ test_process_example_png_success
- ✅ test_ocr_extracts_actual_text
- ✅ test_bbox_coordinates_normalized
- ✅ test_confidence_scores_present_and_valid
- ✅ test_bbox_dimensions_reasonable
- ✅ test_text_ordering_logical
- ✅ test_low_confidence_threshold
- ✅ test_high_confidence_threshold
- ✅ test_blank_white_image
- ✅ test_very_small_image
- ✅ test_empty_ocr_result_triggers_fallback
- ✅ test_output_format_matches_expected
- ✅ test_page_number_always_one_for_images
- ✅ test_output_format_consistency_with_pdf_loader

## Version Compatibility

### PaddleOCR 3.x (Recommended)
- **Initialization:** `PaddleOCR(lang="en", device="gpu")`
- **Method:** `predict(image_path)`
- **Result Format:** `OCRResult` object with keys: `rec_texts`, `rec_scores`, `rec_polys`

### PaddleOCR 2.x (Legacy Support)
- **Initialization:** `PaddleOCR(lang="en", use_gpu=True, show_log=False)`
- **Method:** `ocr(image_path, cls=True)`
- **Result Format:** Nested list `[[bbox, [text, confidence]], ...]`

## Benefits

1. ✅ **Fixes Production Bug:** OcrImageLoader now works with PaddleOCR 3.x
2. ✅ **Backward Compatible:** Still supports PaddleOCR 2.x for existing users
3. ✅ **No Breaking Changes:** External API (`use_gpu` parameter) remains unchanged
4. ✅ **Cleaner Tests:** No workaround fixtures needed
5. ✅ **Future-Proof:** Version detection handles upgrades gracefully
6. ✅ **Better Logging:** Shows detected version in logs

## API Differences Between Versions

| Feature | PaddleOCR 2.x | PaddleOCR 3.x |
|---------|---------------|---------------|
| **GPU Parameter** | `use_gpu=True/False` | `device="gpu"/"cpu"` |
| **Logging Control** | `show_log=True/False` | ❌ Not supported |
| **OCR Method** | `ocr(path, cls=True)` | `predict(path)` |
| **Result Type** | Nested lists | `OCRResult` object |
| **Text Access** | `result[0][i][1][0]` | `result[0]['rec_texts'][i]` |
| **Confidence Access** | `result[0][i][1][1]` | `result[0]['rec_scores'][i]` |
| **BBox Access** | `result[0][i][0]` | `result[0]['rec_polys'][i]` |

## Verification

To verify the fix works on your system:

```bash
# Check PaddleOCR version
python -c "import paddleocr; print(f'PaddleOCR version: {paddleocr.__version__}')"

# Test basic initialization
python -c "
from cognee.infrastructure.ocr.PaddleOCRAdapter import PaddleOCRAdapter
adapter = PaddleOCRAdapter(use_gpu=False)
ocr_engine = adapter._get_ocr_engine()
print('✓ PaddleOCR initialized successfully')
"

# Run tests
pytest cognee/tests/unit/loaders/test_ocr_image_loader_real.py -v
```

## Edge Cases Handled

1. **Version Detection Failure:** Falls back to version 3 behavior (most common)
2. **Pre-release Versions:** Parses version strings like "3.0.0rc1" correctly
3. **Missing __version__ Attribute:** Gracefully defaults to version 3
4. **Numpy Arrays:** Converts `rec_polys` numpy arrays to lists for bbox processing
5. **Missing Fields:** Uses `.get()` with defaults for optional OCRResult fields

## References

- **PaddleOCR 3.x Migration Issue:** https://github.com/PaddlePaddle/PaddleOCR/issues/10429
- **PaddleOCR Version:** 3.3.3 (tested)
- **Related Files:**
  - `cognee/infrastructure/ocr/PaddleOCRAdapter.py`
  - `cognee/infrastructure/loaders/core/ocr_image_loader.py`
  - `cognee/tests/unit/loaders/test_ocr_image_loader_real.py`
