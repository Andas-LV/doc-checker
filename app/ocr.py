import io
import pymupdf as fitz
import pytesseract
from PIL import Image
from .config import settings
import os

os.environ["OMP_THREAD_LIMIT"] = "1"
pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

def _ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, lang="rus+eng")

def extract_text(data: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        parts = []
        for page in doc:
            text = page.get_text().strip()
            if len(text) < 30:  # почти нет текста → скан, нужен OCR
                pix = page.get_pixmap(dpi=200)
                text = _ocr_image(Image.open(io.BytesIO(pix.tobytes("png"))))
            parts.append(text)
        return "\n".join(parts)
    return _ocr_image(Image.open(io.BytesIO(data)))