"""OCR configuration settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class OCRConfig(BaseSettings):
    """Configuration for OCR functionality."""

    ocr_enabled: bool = False
    ocr_language: str = "en"
    ocr_use_gpu: bool = False
    ocr_min_confidence: float = 0.5
    ocr_fallback_on_failure: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


def get_ocr_config() -> OCRConfig:
    """Get OCR configuration instance."""
    return OCRConfig()
