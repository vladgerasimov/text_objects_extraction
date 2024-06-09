from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings

from src.core.models import ProcessAttentions

ROOT_DIR = Path.cwd()
CONFIGS_DIR = ROOT_DIR / "configs"


class ExtractorType(StrEnum):
    pos_extractor = auto()
    llm_extractor = auto()
    bert_extractor = auto()


class AppSettings(BaseSettings):
    text_max_len: int = 1024
    api_selected_extractor: ExtractorType = ExtractorType.bert_extractor

    @classmethod
    def from_yaml(cls, config_path: str | Path, common_settings_path: str | Path = "") -> "AppSettings":
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
    def from_yaml(cls, config_path: str | Path, common_settings_path: str | Path = "") -> "ExtractorClientSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        with open(common_settings_path) as file:
            config.update(yaml.safe_load(file))
        return cls(**config)


class LLMExtractorSettings(BaseSettings):
    openai_key: SecretStr
    prompt_task: str = ""
    prompt_example: str = ""
    multi_text_prompt_example: str = ""
    prompt_input: str = ""
    multi_text_input: str = ""

    model: str = "gpt-3.5-turbo"

    openai_url: str = "https://api.openai.com"
    openai_handler: str = "/v1/chat/completions"

    multi_request_batch_size: int = 16
    max_retries: int = 10

    @property
    def openai_endpoint(self) -> str:
        return self.openai_url + self.openai_handler

    def make_prompt(self, input_text: str) -> str:
        return self.prompt_task + self.prompt_example + self.prompt_input.format(input_text=input_text)

    def make_multi_texts_prompt(self, input_texts: Iterable[str]) -> str:
        return (
            self.prompt_task
            + self.multi_text_prompt_example
            + self.multi_text_input.format(input_texts=[f"<{input_text}>" for input_text in input_texts])
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path, **kwargs: Any) -> "LLMExtractorSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return cls(**config, **kwargs)


class BertExtractorSettings(BaseSettings):
    pretrained_model_name: str = "google-bert/bert-base-uncased"
    process_attentions: ProcessAttentions
    n_blocks_to_average: int = 12

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BertExtractorSettings":
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return cls(**config)


if __name__ == "__main__":
    config_path = "/Users/user/hse/text_objects_extraction/configs/llm_extractor_settings.yaml"
    config = LLMExtractorSettings.from_yaml(config_path)
    print(config)
