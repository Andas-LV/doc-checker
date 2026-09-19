from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_api_key: str = "ollama"
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

settings = Settings()