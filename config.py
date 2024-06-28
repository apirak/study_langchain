from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    OPEN_AI_KEY: str
    OPEN_AI_ORG: str
    LANG_SMIT_KEY: str

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()