from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):

    app_name: str = "EduGenie"

    gemini_api_key: str = ""

    gemini_model: str = "gemini-2.5-flash"

    use_local_explainer: bool = False

    local_explainer_model: str = (
        "MBZUAI/LaMini-Flan-T5-783M"
    )

    max_input_chars: int = 30000

    request_timeout_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def gemini_configured(self) -> bool:

        return bool(
            self.gemini_api_key.strip()
        )


@lru_cache
def get_settings() -> Settings:

    return Settings()


settings = get_settings()