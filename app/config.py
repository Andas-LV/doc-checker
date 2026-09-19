from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

class Settings(BaseSettings):
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    tesseract_cmd: str

settings = Settings()
