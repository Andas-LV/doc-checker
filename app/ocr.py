import io
import os
from pathlib import Path

import pymupdf as fitz
import pytesseract
from PIL import Image, UnidentifiedImageError

from .config import settings

os.environ["OMP_THREAD_LIMIT"] = "1"
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

OCR_DPI = 200
MIN_PAGE_TEXT = 30  # меньше символов на странице — считаем её сканом
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class DocumentReadError(Exception):
    """Файл не удалось прочитать: неподдерживаемый формат или битые данные."""


def _ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, lang="rus+eng")


def _page_text(page: fitz.Page) -> str:
    text = page.get_text().strip()
    if len(text) >= MIN_PAGE_TEXT:
        return text
    pixmap = page.get_pixmap(dpi=OCR_DPI)
    return _ocr_image(Image.open(io.BytesIO(pixmap.tobytes("png"))))


def _pdf_text(data: bytes) -> str:
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(_page_text(page) for page in doc)
    except fitz.FileDataError as e:
        raise DocumentReadError(f"Повреждённый PDF: {e}") from e


def _image_text(data: bytes) -> str:
    try:
        return _ocr_image(Image.open(io.BytesIO(data)))
    except UnidentifiedImageError as e:
        raise DocumentReadError("Не удалось распознать изображение") from e


def extract_text(data: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(data)
    if suffix in IMAGE_SUFFIXES:
        return _image_text(data)
    raise DocumentReadError(
        f"Неподдерживаемый формат «{suffix or filename}». Нужен PDF или изображение."
    )
