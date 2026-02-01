"""Integration tests for OCR visualization."""

import pytest
from pathlib import Path
from cognee import visualize_ocr
from cognee.infrastructure.ocr import PaddleOCRAdapter, is_paddleocr_available

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.mark.skipif(not is_paddleocr_available(), reason="PaddleOCR not installed")
class TestOCRVisualization:

    @pytest.mark.asyncio
    async def test_visualize_from_file_path(self, tmp_path):
        """Test visualization from image file path."""
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont

        # Create a white image with some text
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((50, 50), "Test Text 123", fill='black', font=font)
        draw.text((50, 100), "Sample Document", fill='black', font=font)

        test_image = tmp_path / "test_input.png"
        img.save(test_image)

        # Run visualization
        output_path = str(tmp_path / "output.png")

        result = await visualize_ocr(
            str(test_image),
            output_path=output_path,
            min_confidence=0.5
        )

        assert Path(result).exists()
        assert result == str(Path(output_path).absolute())

        # Verify output is a valid image
        output_img = Image.open(result)
        assert output_img.size == (400, 200)

    @pytest.mark.asyncio
    async def test_visualize_from_ocr_result(self, tmp_path):
        """Test visualization from OCRPageResult object."""
        # Create test image
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((30, 30), "OCR Test", fill='black', font=font)

        test_image = tmp_path / "test_ocr.png"
        img.save(test_image)

        # Run OCR first
        adapter = PaddleOCRAdapter(min_confidence=0.5)
        ocr_result = await adapter.process_image(str(test_image))

        # Visualize
        output_path = str(tmp_path / "from_result.png")
        result = await visualize_ocr(
            ocr_result,
            image_path=str(test_image),
            output_path=output_path
        )

        assert Path(result).exists()
        output_img = Image.open(result)
        assert output_img.size == (300, 150)

    @pytest.mark.asyncio
    async def test_confidence_filtering(self, tmp_path):
        """Test that confidence filtering parameter is accepted."""
        # Create test image
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((30, 30), "Confidence Test", fill='black', font=font)

        test_image = tmp_path / "confidence_test.png"
        img.save(test_image)

        # High confidence threshold
        output_high = str(tmp_path / "high_conf.png")
        result_high = await visualize_ocr(
            str(test_image),
            output_path=output_high,
            min_confidence=0.9
        )

        # Low confidence threshold
        output_low = str(tmp_path / "low_conf.png")
        result_low = await visualize_ocr(
            str(test_image),
            output_path=output_low,
            min_confidence=0.3
        )

        # Both should create files
        assert Path(result_high).exists()
        assert Path(result_low).exists()

    @pytest.mark.asyncio
    async def test_auto_generated_output_path(self, tmp_path):
        """Test that output path is auto-generated when not provided."""
        # Create test image
        from PIL import Image

        img = Image.new('RGB', (200, 100), color='white')
        test_image = tmp_path / "auto_path_test.png"
        img.save(test_image)

        # Run without specifying output path
        result = await visualize_ocr(str(test_image))

        # Should generate path in home directory
        assert Path(result).exists()
        assert "ocr_visualization_auto_path_test.png" in result
        import os
        assert os.path.expanduser("~") in result

        # Clean up
        Path(result).unlink()

    @pytest.mark.asyncio
    async def test_custom_visualization_options(self, tmp_path):
        """Test custom visualization options."""
        # Create test image
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw.text((30, 30), "Options Test", fill='black', font=font)

        test_image = tmp_path / "options_test.png"
        img.save(test_image)

        output_path = str(tmp_path / "custom_options.png")

        # Test with custom options
        result = await visualize_ocr(
            str(test_image),
            output_path=output_path,
            min_confidence=0.5,
            show_text=True,
            show_confidence=True,
            box_width=3
        )

        assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_missing_file_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await visualize_ocr("/nonexistent/file.png")

    @pytest.mark.asyncio
    async def test_ocr_result_without_image_path_error(self, tmp_path):
        """Test that OCRPageResult without image_path raises ValueError."""
        # Create test image and get OCR result
        from PIL import Image

        img = Image.new('RGB', (200, 100), color='white')
        test_image = tmp_path / "test.png"
        img.save(test_image)

        adapter = PaddleOCRAdapter()
        ocr_result = await adapter.process_image(str(test_image))

        # Try to visualize without providing image_path
        with pytest.raises(ValueError, match="must provide 'image_path'"):
            await visualize_ocr(ocr_result)
