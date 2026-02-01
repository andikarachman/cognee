# Quick Guide: Running OCR Tests

## Quick Start

### Run All OCR Tests
```bash
# All OCR unit tests
pytest cognee/tests/unit/infrastructure/ocr/ -v

# All layout-related tests
pytest cognee/tests/unit/chunking/test_layout_chunk_model.py \
       cognee/tests/unit/chunking/test_layout_text_chunker.py \
       cognee/tests/unit/search/test_search_result_layout.py -v

# All loader tests
pytest cognee/tests/unit/loaders/test_*ocr*.py \
       cognee/tests/unit/loaders/test_pdfplumber_loader.py \
       cognee/tests/unit/loaders/test_pdf_type_detector.py -v

# Integration tests
pytest cognee/tests/integration/test_ocr_pipeline_end_to_end.py -v
```

### Run Single Test File
```bash
pytest cognee/tests/unit/infrastructure/ocr/test_paddle_ocr_adapter.py -v
```

### Run Specific Test Class
```bash
pytest cognee/tests/unit/infrastructure/ocr/test_paddle_ocr_adapter.py::TestBoundingBoxNormalization -v
```

### Run Single Test
```bash
pytest cognee/tests/unit/chunking/test_layout_chunk_model.py::TestBoundingBoxModel::test_valid_bbox_creation -v
```

## With Coverage

### Generate Coverage Report
```bash
# OCR infrastructure coverage
pytest cognee/tests/unit/infrastructure/ocr/ \
  --cov=cognee.infrastructure.ocr \
  --cov-report=html \
  --cov-report=term

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Comprehensive Coverage
```bash
# All OCR-related modules
pytest cognee/tests/unit/infrastructure/ocr/ \
       cognee/tests/unit/loaders/test_*ocr*.py \
       cognee/tests/unit/chunking/test_layout*.py \
  --cov=cognee.infrastructure.ocr \
  --cov=cognee.infrastructure.loaders \
  --cov=cognee.modules.chunking \
  --cov-report=html
```

## Debugging

### Show Full Error Traceback
```bash
pytest cognee/tests/unit/infrastructure/ocr/test_paddle_ocr_adapter.py -vv
```

### Run Failed Tests Only
```bash
# First run to identify failures
pytest cognee/tests/unit/infrastructure/ocr/ -v

# Re-run only failed tests
pytest --lf -v
```

### Stop on First Failure
```bash
pytest cognee/tests/unit/infrastructure/ocr/ -x
```

### Show Print Statements
```bash
pytest cognee/tests/unit/infrastructure/ocr/ -v -s
```

## Test Selection

### By Marker
```bash
# Run slow tests only
pytest -m slow -v

# Skip slow tests
pytest -m "not slow" -v
```

### By Name Pattern
```bash
# All tests with "bbox" in name
pytest -k "bbox" -v

# All normalization tests
pytest -k "normalize" -v

# All OCR config tests
pytest -k "config" -v
```

## Performance

### Show Slowest Tests
```bash
pytest cognee/tests/unit/infrastructure/ocr/ --durations=10
```

### Run Tests in Parallel (requires pytest-xdist)
```bash
pip install pytest-xdist
pytest cognee/tests/unit/ -n auto
```

## Validation

### Collect Tests (Syntax Check)
```bash
# Verify all tests can be collected
pytest cognee/tests/unit/infrastructure/ocr/ --collect-only
```

### Dry Run
```bash
pytest cognee/tests/unit/infrastructure/ocr/ --collect-only -q
```

## Common Issues

### Issue: Import Errors
```bash
# Ensure you're in the project root
cd /Users/andikarachman/personal_projects/kg/cognee

# Ensure cognee is installed in editable mode
pip install -e .
```

### Issue: Missing Dependencies
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Install OCR dependencies (optional)
pip install "cognee[ocr]"
```

### Issue: Async Tests Fail
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check pytest-asyncio mode
pytest --asyncio-mode=auto -v
```

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run OCR Tests
  run: |
    pytest cognee/tests/unit/infrastructure/ocr/ \
           cognee/tests/unit/loaders/test_*ocr*.py \
      --cov=cognee.infrastructure.ocr \
      --cov-report=xml
```

### Coverage Threshold
```bash
# Fail if coverage < 90%
pytest cognee/tests/unit/infrastructure/ocr/ \
  --cov=cognee.infrastructure.ocr \
  --cov-fail-under=90
```

## Test Files Reference

### Unit Tests
```
cognee/tests/unit/infrastructure/ocr/
├── test_paddle_ocr_adapter.py (20 tests)
├── test_ocr_config.py (5 tests)

cognee/tests/unit/loaders/
├── test_pdf_type_detector.py (12 tests)
├── test_pdfplumber_loader.py (18 tests)
├── test_ocr_pdf_loader.py (15 tests)
├── test_ocr_image_loader.py (12 tests)

cognee/tests/unit/chunking/
├── test_layout_chunk_model.py (25 tests)
├── test_layout_text_chunker.py (18 tests)

cognee/tests/unit/retrieval/
├── test_chunks_retriever_layout.py (10 tests)

cognee/tests/unit/search/
├── test_search_result_layout.py (10 tests)
```

### Integration Tests
```
cognee/tests/integration/
├── test_ocr_pipeline_end_to_end.py (8 tests)
├── test_loader_integration.py (5 tests)
├── test_chunking_integration.py (5 tests)
```

## Expected Execution Times

- **Fast unit tests:** <1 second per test
- **Mock-heavy tests:** 1-3 seconds per test
- **Integration tests:** 5-10 seconds per test
- **Full suite:** <60 seconds for unit tests

## Next Steps

1. **Fix known issues:**
   ```bash
   # Focus on these first
   pytest cognee/tests/unit/search/test_search_result_layout.py -v
   pytest cognee/tests/unit/infrastructure/ocr/test_ocr_config.py -v
   ```

2. **Run comprehensive test:**
   ```bash
   pytest cognee/tests/unit/ -v --tb=short
   ```

3. **Generate coverage report:**
   ```bash
   pytest cognee/tests/unit/ --cov=cognee --cov-report=html
   ```

## Useful Pytest Options

- `-v` : Verbose output
- `-vv` : Very verbose (shows full diff)
- `-s` : Show print statements
- `-x` : Stop on first failure
- `--lf` : Run last failed tests
- `--ff` : Run failed tests first
- `--tb=short` : Shorter traceback
- `--tb=no` : No traceback
- `-k EXPRESSION` : Run tests matching expression
- `-m MARKER` : Run tests with marker
- `--durations=N` : Show N slowest tests
- `--collect-only` : Collect tests without running
- `-n auto` : Run tests in parallel (requires pytest-xdist)

For more help: `pytest --help`
