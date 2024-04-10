from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    text_max_len: int = 1024

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AppSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return cls(**config)


class PosExtractorSettings(BaseSettings):
    ...

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PosExtractorSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return cls(**config)
