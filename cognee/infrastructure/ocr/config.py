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

    # PPStructureV3 options
    ocr_use_structure: bool = False
    ocr_structure_use_table_recognition: bool = True
    ocr_structure_use_formula_recognition: bool = False
    ocr_structure_layout_threshold: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


def get_ocr_config() -> OCRConfig:
    """Get OCR configuration instance."""
    return OCRConfig()


def is_ppstructure_available() -> bool:
    """
    Check if PPStructureV3 and its dependencies (PaddleX[ocr]) are available.

    This function checks if the required imports are available without performing
    full initialization to avoid loading models during pytest collection phase.
    Any runtime dependency errors will be caught during actual usage.

    Returns:
        bool: True if PPStructureV3 can be imported, False otherwise
    """
    try:
        # Check if base PaddleOCR is available first
        import paddleocr  # noqa: F401

        # Import PPStructureV3 (requires PaddleX)
        from paddleocr import PPStructureV3  # noqa: F401

        # If we can import PPStructureV3, it's available
        # Any runtime dependency errors will be caught during actual usage
        # This avoids loading models twice (once during collection, once during execution)
        return True

    except (ImportError, ModuleNotFoundError):
        # Import failed - PPStructureV3 or its dependencies not installed
        return False
    except Exception:
        # Other errors (e.g., initialization errors) - treat as unavailable
        return False
