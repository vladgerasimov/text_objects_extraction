from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

ROOT_DIR = Path.cwd()
CONFIGS_DIR = ROOT_DIR / "configs"


class AppSettings(BaseSettings):
    text_max_len: int = 1024

    @classmethod
    def from_yaml(cls, config_path: str | Path, common_settings_path: str | Path) -> "AppSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return cls(**config)


class PosExtractorSettings(BaseSettings):
    ...

    @classmethod
    def from_yaml(cls, config_path: str | Path, common_settings_path: str | Path) -> "PosExtractorSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        with open(common_settings_path) as file:
            config.update(yaml.safe_load(file))
        return cls(**config)


class ExtractorClientSettings(AppSettings):
    extractor_url: str = ""
    extract_objects_handler: str = "/extract"

    @property
    def extract_object_endpoint(self) -> str:
        return self.extractor_url + self.extract_objects_handler

    @classmethod
    def from_yaml(cls, config_path: str | Path, common_settings_path: str | Path) -> "ExtractorClientSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        with open(common_settings_path) as file:
            config.update(yaml.safe_load(file))
        return cls(**config)
