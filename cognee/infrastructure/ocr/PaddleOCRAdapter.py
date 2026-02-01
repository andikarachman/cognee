"""PaddleOCR adapter for OCR processing."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from cognee.shared.logging_utils import get_logger
from cognee.shared.data_models import BoundingBox

logger = get_logger(__name__)


@dataclass
class OCRTextElement:
    """Single text element extracted from OCR."""

    text: str
    bbox: BoundingBox
    confidence: float
    page_number: int
    layout_type: str = "text"  # Default for backward compatibility


@dataclass
class OCRPageResult:
    """OCR result for a single page."""

    page_number: int
    elements: List[OCRTextElement]
    page_width: int
    page_height: int
    layout_info: Optional[Dict[str, Any]] = None  # PPStructureV3 raw data


@dataclass
class OCRDocumentResult:
    """OCR result for entire document."""

    pages: List[OCRPageResult]
    total_pages: int


class PaddleOCRAdapter:
    """
    Adapter for PaddleOCR functionality.

    Provides OCR processing for images and PDFs with bounding box extraction.
    Lazy initialization ensures PaddleOCR is only loaded when actually used.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
        use_structure: bool = False,
        structure_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PaddleOCR adapter.

        Args:
            lang: Language code for OCR (e.g., 'en', 'ch', 'fr')
            use_gpu: Whether to use GPU acceleration
            min_confidence: Minimum confidence threshold for text detection
            use_structure: Whether to use PPStructureV3 for layout-aware OCR
            structure_config: Optional configuration for PPStructureV3
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.use_structure = use_structure
        self.structure_config = structure_config or {}
        self._ocr_engine = None
        self._structure_engine = None
        self._paddleocr_version = None

    def _get_ocr_engine(self):
        """Lazy initialization of PaddleOCR engine."""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                import paddleocr

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

                if major_version >= 3:
                    # PaddleOCR 3.x: Use 'device' parameter, no 'show_log' parameter
                    device = "gpu" if self.use_gpu else "cpu"
                    logger.info(f"Initializing PaddleOCR v{major_version}.x with lang={self.lang}, device={device}")
                    self._ocr_engine = PaddleOCR(
                        lang=self.lang,
                        device=device,
                    )
                else:
                    # PaddleOCR 2.x: Use 'use_gpu' parameter (legacy)
                    logger.info(f"Initializing PaddleOCR v{major_version}.x with lang={self.lang}, use_gpu={self.use_gpu}")
                    self._ocr_engine = PaddleOCR(
                        lang=self.lang,
                        use_gpu=self.use_gpu,
                        show_log=False,
                    )

            except ImportError as e:
                raise ImportError(
                    "PaddleOCR is required for OCR processing. "
                    "Install with: pip install cognee[ocr]"
                ) from e

        return self._ocr_engine

    def _get_structure_engine(self):
        """Lazy initialization of PPStructureV3 engine."""
        if self._structure_engine is None:
            try:
                from paddleocr import PPStructureV3

                logger.info(f"Initializing PPStructureV3 with lang={self.lang}")

                init_params = {
                    'lang': self.lang,
                    'device': 'gpu' if self.use_gpu else 'cpu',
                }

                if self.structure_config:
                    init_params.update(self.structure_config)

                self._structure_engine = PPStructureV3(**init_params)

            except ImportError as e:
                raise ImportError(
                    "PPStructureV3 requires additional dependencies. "
                    "Install with: pip install paddlex[ocr]"
                ) from e

        return self._structure_engine

    def _normalize_bbox(
        self,
        bbox_coords: List[List[float]],
        page_width: int,
        page_height: int,
    ) -> BoundingBox:
        """
        Normalize bounding box coordinates to 0-1 range.

        Args:
            bbox_coords: PaddleOCR bbox format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            page_width: Page width in pixels
            page_height: Page height in pixels

        Returns:
            BoundingBox with normalized and pixel coordinates
        """
        # Extract min/max from 4-point polygon
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]

        pixel_x_min = int(min(x_coords))
        pixel_y_min = int(min(y_coords))
        pixel_x_max = int(max(x_coords))
        pixel_y_max = int(max(y_coords))

        # Normalize to 0-1 range
        x_min = pixel_x_min / page_width if page_width > 0 else 0
        y_min = pixel_y_min / page_height if page_height > 0 else 0
        x_max = pixel_x_max / page_width if page_width > 0 else 1
        y_max = pixel_y_max / page_height if page_height > 0 else 1

        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            pixel_x_min=pixel_x_min,
            pixel_y_min=pixel_y_min,
            pixel_x_max=pixel_x_max,
            pixel_y_max=pixel_y_max,
        )

    def _map_layout_label_to_type(self, label: str) -> str:
        """Map PPStructureV3 layout label to LayoutType enum value."""
        LAYOUT_MAPPING = {
            'text': 'text',
            'title': 'title',
            'paragraph title': 'heading',
            'paragraph': 'paragraph',
            'table': 'table',
            'table caption': 'caption',
            'figure': 'figure',
            'figure caption': 'caption',
            'image': 'figure',
            'header': 'header',
            'footer': 'footer',
            'page number': 'footer',
            'formula': 'code',
            'formula number': 'code',
            'reference': 'text',
            'footnote': 'text',
            'chart': 'figure',
            'algorithm': 'code',
            'seal': 'figure',
            'list': 'list',
        }
        return LAYOUT_MAPPING.get(label.lower(), 'text')

    def _calculate_bbox_overlap(
        self,
        x1_min: float, y1_min: float, x1_max: float, y1_max: float,
        x2_min: float, y2_min: float, x2_max: float, y2_max: float,
    ) -> float:
        """Calculate intersection over union (IoU) between two bboxes."""
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _find_layout_type_for_bbox(
        self,
        ocr_bbox: BoundingBox,
        layout_boxes: List[Dict],
        page_width: int,
        page_height: int,
    ) -> str:
        """Find layout type by matching OCR bbox with layout regions."""
        max_overlap = 0
        best_label = 'text'

        for layout_box in layout_boxes:
            coords = layout_box.get('coordinate', [])
            if len(coords) != 4:
                continue

            # Normalize layout bbox
            layout_x_min = coords[0] / page_width
            layout_y_min = coords[1] / page_height
            layout_x_max = coords[2] / page_width
            layout_y_max = coords[3] / page_height

            # Calculate overlap
            overlap = self._calculate_bbox_overlap(
                ocr_bbox.x_min, ocr_bbox.y_min, ocr_bbox.x_max, ocr_bbox.y_max,
                layout_x_min, layout_y_min, layout_x_max, layout_y_max,
            )

            if overlap > max_overlap:
                max_overlap = overlap
                best_label = layout_box.get('label', 'text')

        return self._map_layout_label_to_type(best_label)

    def _match_ocr_to_layout(
        self,
        rec_texts: List[str],
        rec_scores: List[float],
        rec_polys: List[List[List[float]]],
        layout_boxes: List[Dict],
        page_width: int,
        page_height: int,
        page_number: int,
    ) -> List[OCRTextElement]:
        """Match OCR text regions to layout classifications."""
        elements = []

        for i, text in enumerate(rec_texts):
            confidence = rec_scores[i] if i < len(rec_scores) else 1.0
            bbox_coords = rec_polys[i] if i < len(rec_polys) else None

            if confidence < self.min_confidence or bbox_coords is None:
                continue

            # Normalize bbox
            bbox = self._normalize_bbox(bbox_coords, page_width, page_height)
            bbox.confidence = confidence

            # Find matching layout type
            layout_type = self._find_layout_type_for_bbox(
                bbox, layout_boxes, page_width, page_height
            )

            elements.append(
                OCRTextElement(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    page_number=page_number,
                    layout_type=layout_type,
                )
            )

        return elements

    async def _process_image_with_structure(
        self,
        image_path: str,
        page_number: int = 1,
    ) -> OCRPageResult:
        """Process image with PPStructureV3 (layout-aware)."""
        from PIL import Image

        structure_engine = self._get_structure_engine()

        # Load image dimensions
        with Image.open(image_path) as img:
            page_width, page_height = img.size

        logger.info(f"Processing image {image_path} with PPStructureV3")

        # Run PPStructureV3 prediction
        results = structure_engine.predict(image_path)

        elements = []
        layout_boxes = []

        for res in results:
            # Access layout detection and OCR results
            layout_res = res.get('layout_det_res', {})
            ocr_res = res.get('overall_ocr_res', {})

            # Extract layout boxes
            layout_boxes = layout_res.get('boxes', [])

            # Extract OCR results
            rec_texts = ocr_res.get('rec_texts', [])
            rec_scores = ocr_res.get('rec_scores', [])
            rec_polys = ocr_res.get('rec_polys', [])

            # Match OCR to layout
            elements.extend(
                self._match_ocr_to_layout(
                    rec_texts, rec_scores, rec_polys,
                    layout_boxes, page_width, page_height, page_number
                )
            )

        logger.info(f"Extracted {len(elements)} elements with layout types")

        return OCRPageResult(
            page_number=page_number,
            elements=elements,
            page_width=page_width,
            page_height=page_height,
            layout_info={'layout_boxes': layout_boxes} if layout_boxes else None,
        )

    async def process_image(
        self,
        image_path: str,
        page_number: int = 1,
    ) -> OCRPageResult:
        """
        Process a single image with OCR.

        Routes to PPStructureV3 if use_structure=True, else uses PaddleOCR.

        Args:
            image_path: Path to image file
            page_number: Page number for tracking (default 1 for images)

        Returns:
            OCRPageResult with extracted text and bounding boxes

        Raises:
            ImportError: If PaddleOCR not installed
            Exception: If OCR processing fails
        """
        if self.use_structure:
            return await self._process_image_with_structure(image_path, page_number)
        else:
            return await self._process_image_with_paddleocr(image_path, page_number)

    async def _process_image_with_paddleocr(
        self,
        image_path: str,
        page_number: int = 1,
    ) -> OCRPageResult:
        """Process image with standard PaddleOCR (existing logic)."""
        try:
            from PIL import Image

            ocr_engine = self._get_ocr_engine()

            # Load image to get dimensions
            with Image.open(image_path) as img:
                page_width, page_height = img.size

            logger.info(f"Processing image {image_path} with PaddleOCR")

            # PaddleOCR 3.x uses predict() method, 2.x uses ocr() method
            elements = []
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

                        # Skip low confidence results
                        if confidence < self.min_confidence:
                            continue

                        if bbox_coords is not None:
                            # Convert numpy array to list if needed
                            if hasattr(bbox_coords, 'tolist'):
                                bbox_coords = bbox_coords.tolist()

                            bbox = self._normalize_bbox(bbox_coords, page_width, page_height)
                            bbox.confidence = confidence

                            elements.append(
                                OCRTextElement(
                                    text=text,
                                    bbox=bbox,
                                    confidence=confidence,
                                    page_number=page_number,
                                )
                            )
            else:
                # PaddleOCR 2.x API
                result = ocr_engine.ocr(image_path, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        bbox_coords = line[0]
                        text_info = line[1]
                        text = text_info[0]
                        confidence = text_info[1]

                        # Skip low confidence results
                        if confidence < self.min_confidence:
                            continue

                        bbox = self._normalize_bbox(bbox_coords, page_width, page_height)
                        bbox.confidence = confidence

                        elements.append(
                            OCRTextElement(
                                text=text,
                                bbox=bbox,
                                confidence=confidence,
                                page_number=page_number,
                            )
                        )

            logger.info(f"Extracted {len(elements)} text elements from image")
            return OCRPageResult(
                page_number=page_number,
                elements=elements,
                page_width=page_width,
                page_height=page_height,
            )

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise

    async def process_pdf(self, pdf_path: str) -> OCRDocumentResult:
        """
        Process PDF with OCR by converting to images first.

        Args:
            pdf_path: Path to PDF file

        Returns:
            OCRDocumentResult with all pages processed

        Raises:
            ImportError: If required libraries not installed
            Exception: If PDF processing fails
        """
        try:
            from pdf2image import convert_from_path

            logger.info(f"Converting PDF {pdf_path} to images for OCR")
            images = convert_from_path(pdf_path)

            pages = []
            for page_num, image in enumerate(images, start=1):
                # Save image temporarily
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path, "PNG")

                # Process with OCR
                page_result = await self.process_image(temp_path, page_number=page_num)
                pages.append(page_result)

                # Cleanup temp file
                import os

                os.unlink(temp_path)

                logger.info(
                    f"Processed page {page_num}/{len(images)} "
                    f"({len(page_result.elements)} elements)"
                )

            return OCRDocumentResult(
                pages=pages,
                total_pages=len(pages),
            )

        except ImportError as e:
            raise ImportError(
                "pdf2image is required for PDF OCR. Install with: pip install cognee[ocr]"
            ) from e
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise
