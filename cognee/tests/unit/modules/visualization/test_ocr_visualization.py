"""Unit tests for OCR visualization module."""

import pytest
import os
from pathlib import Path
from cognee.modules.visualization.ocr_visualization import (
    get_confidence_color,
    truncate_text,
    generate_output_path,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    LOW_CONFIDENCE,
)


def test_get_confidence_color():
    """Test confidence score to color mapping."""
    assert get_confidence_color(0.95) == HIGH_CONFIDENCE
    assert get_confidence_color(0.85) == HIGH_CONFIDENCE
    assert get_confidence_color(0.81) == HIGH_CONFIDENCE

    assert get_confidence_color(0.8) == MEDIUM_CONFIDENCE
    assert get_confidence_color(0.6) == MEDIUM_CONFIDENCE
    assert get_confidence_color(0.5) == MEDIUM_CONFIDENCE

    assert get_confidence_color(0.49) == LOW_CONFIDENCE
    assert get_confidence_color(0.3) == LOW_CONFIDENCE
    assert get_confidence_color(0.1) == LOW_CONFIDENCE


def test_truncate_text():
    """Test text truncation for display."""
    short = "Hello"
    assert truncate_text(short, 10) == "Hello"

    long = "This is a very long text that should be truncated"
    result = truncate_text(long, 20)
    assert len(result) <= 23  # 20 + "..."
    assert result.endswith("...")

    # Test default max length
    very_long = "A" * 50
    result = truncate_text(very_long)
    assert len(result) == 33  # 30 + "..."
    assert result.endswith("...")


def test_generate_output_path():
    """Test output path generation."""
    source = "/path/to/image.png"
    output = generate_output_path(source)

    assert "ocr_visualization_image.png" in output
    assert os.path.expanduser("~") in output

    # Test with page number
    output_page = generate_output_path(source, page_num=2)
    assert "page_2" in output_page
    assert "ocr_visualization_image_page_2.png" in output_page


def test_generate_output_path_different_extensions():
    """Test output path generation with different file extensions."""
    sources = [
        "/path/to/document.pdf",
        "/path/to/scan.jpg",
        "/path/to/photo.jpeg",
    ]

    for source in sources:
        output = generate_output_path(source)
        base_name = Path(source).stem
        assert f"ocr_visualization_{base_name}.png" in output
