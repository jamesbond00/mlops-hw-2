"""Application settings loaded from .env / environment.

The Nebius API key is wrapped in SecretStr — it does not appear in repr,
str, or default JSON. Unwrap only at the OpenAI client boundary via
.get_secret_value().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Push .env values into os.environ so libraries that read the environment
# directly (mlflow client, boto3, etc.) see the same values that
# pydantic-settings reads via env_file. No effect when .env is absent.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nebius Token Factory
    nebius_api_key: SecretStr = Field(...)
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "travel-assistant"

    # Assistant service — dev mode loads a config by id from configs_dir.
    assistant_config: str = "v1"
    configs_dir: Path = Path("configs")
    assistant_host: str = "0.0.0.0"
    assistant_port: int = 8000

    # MLflow Model Registry — production deployment shape.
    # The eval CLI auto-registers full evals under `mlflow_registered_model_name`
    # as new versions. When `assistant_model_alias` is set, the service loads
    # its config from the version that alias currently points at. Unset alias
    # = local-dev mode (loads from configs_dir).
    mlflow_registered_model_name: str = "travel-assistant"
    assistant_model_alias: str | None = None

    # Admin-endpoint auth. When set, POST /admin/reload requires an
    # `X-Admin-Token` header matching this value. Unset = open in dev.
    admin_token: SecretStr | None = None

    # Judge — orthogonal to assistant configs
    judge_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    judge_temperature: float = 0.0
    judge_sample_rate: float = Field(0.05, ge=0.0, le=1.0)
    judge_queue_maxsize: int = 1000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
