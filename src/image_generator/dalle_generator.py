import json

import requests
from loguru import logger

from src.core.models import ExtractedObjectsDict
from src.core.settings import DallESettings


class DallEGenerator:
    def __init__(self, config: DallESettings):
        self.config = config
        self.__api_token = config.openai_key.get_secret_value()

    def get_generated_url(self, input_json: ExtractedObjectsDict) -> str:
        headers = {"Authorization": f"Bearer {self.__api_token}", "Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "prompt": self.config.make_prompt(json.dumps(input_json)),
        }
        try:
            response_json = requests.post(self.config.openai_endpoint, headers=headers, json=body).json()
            return response_json["data"][0]["url"]
        except Exception as e:
            logger.warning(f"Exception raised when requesting OpenAI API: {e}")
            return ""


if __name__ == "__main__":
    from src.core.settings import CONFIGS_DIR

    settings = DallESettings.from_yaml(CONFIGS_DIR / "dall_e_settings.yaml", _env_file=".env")
    generator = DallEGenerator(settings)
    input_json = {"objects": {"car": ["white"], "road": ["dirty"]}}  # pyright: ignore
    url = generator.get_generated_url(input_json)  # pyright: ignore
    logger.success(url)
