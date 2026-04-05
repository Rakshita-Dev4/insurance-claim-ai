import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    database_url: str = "sqlite:///./app.db"
    environment: str = "development"
    # Tesseract binary path — override via TESSERACT_CMD in .env for non-Windows
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
