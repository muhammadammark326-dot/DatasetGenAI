"""Central settings and path configuration for DatasetGen AI."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env if present
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core data directory (NO hardcoded C:\ or /content/ paths)
    datasetgen_data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("DATASETGEN_DATA_DIR", "./data")).resolve()
    )

    # Provider configs
    default_provider: str = Field(default="mock")
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    gemini_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    # Logging and execution
    log_level: str = Field(default="INFO")
    cache_enabled: bool = Field(default=True)
    batch_size: int = Field(default=20)
    max_retries: int = Field(default=3)

    @property
    def data_dir(self) -> Path:
        """Returns the resolved data directory, ensuring it exists."""
        self.datasetgen_data_dir.mkdir(parents=True, exist_ok=True)
        return self.datasetgen_data_dir

    @property
    def traces_dir(self) -> Path:
        """Directory for execution traces."""
        path = self.data_dir / "traces"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def datasets_dir(self) -> Path:
        """Directory for generated datasets."""
        path = self.data_dir / "datasets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for run checkpoints."""
        path = self.data_dir / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Directory for request/batch cache."""
        path = self.data_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def benchmark_results_dir(self) -> Path:
        """Directory for benchmark results."""
        path = self.data_dir / "benchmark_results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sft_data_dir(self) -> Path:
        """Directory for prepared SFT datasets."""
        path = self.data_dir / "sft_data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Settings:
        """Load settings from a YAML configuration file."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Remap data_dir if present
        if "data_dir" in data:
            data["datasetgen_data_dir"] = Path(data.pop("data_dir"))
        return cls(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application settings."""
    return Settings()
