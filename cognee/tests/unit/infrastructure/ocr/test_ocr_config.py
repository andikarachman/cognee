"""Unit tests for OCR configuration."""

import pytest
from unittest.mock import patch
from cognee.infrastructure.ocr.config import OCRConfig


class TestOCRConfig:
    """Tests for OCR configuration settings."""

    def test_default_config_values(self):
        """Test default OCR configuration values."""
        config = OCRConfig()

        assert config.ocr_enabled is False
        assert config.ocr_language == "en"
        assert config.ocr_use_gpu is False
        assert config.ocr_min_confidence == 0.5
        assert config.ocr_fallback_on_failure is True

    def test_config_with_custom_values(self):
        """Test OCR config with custom values."""
        config = OCRConfig(
            ocr_enabled=True,
            ocr_language="fr",
            ocr_use_gpu=True,
            ocr_min_confidence=0.8,
            ocr_fallback_on_failure=False,
        )

        assert config.ocr_enabled is True
        assert config.ocr_language == "fr"
        assert config.ocr_use_gpu is True
        assert config.ocr_min_confidence == 0.8
        assert config.ocr_fallback_on_failure is False

    @patch.dict(
        "os.environ",
        {
            "OCR_ENABLED": "true",
            "OCR_LANGUAGE": "es",
            "OCR_USE_GPU": "true",
            "OCR_MIN_CONFIDENCE": "0.9",
            "OCR_FALLBACK_ON_FAILURE": "false",
        },
    )
    def test_config_from_environment(self):
        """Test OCR config loads from environment variables."""
        config = OCRConfig()

        assert config.ocr_enabled is True
        assert config.ocr_language == "es"
        assert config.ocr_use_gpu is True
        assert config.ocr_min_confidence == 0.9
        assert config.ocr_fallback_on_failure is False

    def test_config_confidence_validation(self):
        """Test that confidence is validated to be in range 0-1."""
        # Valid confidence values
        config1 = OCRConfig(ocr_min_confidence=0.0)
        assert config1.ocr_min_confidence == 0.0

        config2 = OCRConfig(ocr_min_confidence=1.0)
        assert config2.ocr_min_confidence == 1.0

        config3 = OCRConfig(ocr_min_confidence=0.75)
        assert config3.ocr_min_confidence == 0.75
