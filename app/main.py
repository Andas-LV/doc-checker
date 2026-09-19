from fastapi import FastAPI, UploadFile
from .ocr import extract_text
from .extract import extract_fields

app = FastAPI()

@app.post("/extract")
def extract(file: UploadFile):
    text = extract_text(file.file.read(), file.filename)
    fields = extract_fields(text)
    return {"text_preview": text[:500], "fields": fields}