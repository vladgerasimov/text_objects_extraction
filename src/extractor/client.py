import json

import requests
from loguru import logger

from src.core.models import ExtractedObjectsDict, ExtractObjectsResponseDict
from src.core.settings import ExtractorClientSettings

on_exception_return_value: ExtractedObjectsDict = {"objects": {}}


class ExtractorClient:
    def __init__(self, config: ExtractorClientSettings):
        self.config = config

    def extract_objects(self, text: str) -> ExtractedObjectsDict:
        try:
            response_dict: ExtractObjectsResponseDict = requests.post(
                self.config.extract_object_endpoint, json=text
            ).json()
            return response_dict.get("result", on_exception_return_value)
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(e)
            return on_exception_return_value
