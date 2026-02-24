"""Configuration management for Enterprise Intelligence Agent."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    database_url: str = Field(
        default="sqlite:///./data/enterprise.db",
        alias="DATABASE_URL",
    )
    model_path: str = Field(
        default="models/model.pkl",
        alias="MODEL_PATH",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        alias="LLM_MODEL",
    )
    mock_llm_mode: bool = Field(
        default=False,
        alias="MOCK_LLM_MODE",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    @property
    def should_use_mock_llm(self) -> bool:
        """Use mock LLM when no API key or mock mode is enabled."""
        return self.mock_llm_mode or not self.has_openai_key


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
