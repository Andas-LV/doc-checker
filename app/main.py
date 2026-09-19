from fastapi import FastAPI, HTTPException, UploadFile
from openai import OpenAIError
from pydantic import ValidationError

from .extract import LLMResponseError, extract_fields
from .ocr import DocumentReadError, extract_text
from .schemas import ExtractResponse

MAX_UPLOAD_MB = 20
PREVIEW_CHARS = 500

app = FastAPI(title="doc-checker")


@app.post("/extract", response_model=ExtractResponse)
def extract(file: UploadFile) -> ExtractResponse:
    limit = MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Файл больше {MAX_UPLOAD_MB} МБ")

    try:
        text = extract_text(data, file.filename or "")
    except DocumentReadError as e:
        raise HTTPException(400, str(e)) from e

    try:
        fields = extract_fields(text)
    except OpenAIError as e:
        raise HTTPException(503, f"Модель недоступна: {e}") from e
    except (LLMResponseError, ValidationError) as e:
        raise HTTPException(502, f"Неожиданный ответ модели: {e}") from e

    return ExtractResponse(text_preview=text[:PREVIEW_CHARS], fields=fields)
