import asyncio
import json
import random
from itertools import batched
from pathlib import Path
from typing import Iterable, Sequence

import aiohttp
import requests
from loguru import logger

from src.core.models import ExtractedObjectsDict, LLMResponseJSON
from src.core.settings import AppSettings, LLMExtractorSettings
from src.extractor.base import BaseObjectsExtractor

configs_dir = Path(__file__).parent.parent.parent / "configs"

app_settings = AppSettings()
extractor_settings = LLMExtractorSettings.from_yaml(configs_dir / "llm_extractor_settings.yaml", _env_file=".env")


class LLMExtractor(BaseObjectsExtractor):
    def __init__(self, config: LLMExtractorSettings):
        self.config = config
        self.__api_token = config.openai_key.get_secret_value()

        self.successful_batches_count = 0
        self.parsed: dict[str, ExtractedObjectsDict] = {}

    def extract(self, text: str) -> ExtractedObjectsDict:
        response_json = self.get_response(input_text=text)
        objects = self.parse_response(response_json=response_json)
        return objects

    def get_response(self, input_text: str) -> LLMResponseJSON:
        headers = {"Authorization": f"Bearer {self.__api_token}", "Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": self.config.make_prompt(input_text=input_text)}],
        }
        try:
            response = requests.post(self.config.openai_endpoint, headers=headers, json=body).json()
            return response
        except Exception as e:
            logger.warning(f"Exception raised when requesting OpenAI API: {e}")
            return {"choices": []}

    @staticmethod
    def parse_response(response_json: LLMResponseJSON) -> ExtractedObjectsDict:
        if (
            (choices := response_json.get("choices"))
            and (choice := choices[0])
            and (message := choice["message"])
            and (content := message["content"])
        ):
            return json.loads(content)

        logger.warning(f"Could not parse response: {response_json}")
        return {"objects": {}}

    async def parse_and_save_response(
        self, response_jsons: Iterable[LLMResponseJSON], save_path: str | Path = ""
    ) -> None:
        parsed = self.parse_multi_text_response(response_jsons=response_jsons)
        if save_path:
            for text, label in parsed.items():
                async with asyncio.Lock():
                    with open(save_path, "a") as f:
                        f.write(f"\n{text}~{json.dumps(label)}")

    def parse_multi_text_response(self, response_jsons: Iterable[LLMResponseJSON]) -> dict[str, ExtractedObjectsDict]:
        self.parsed: dict[str, ExtractedObjectsDict] = {}
        for response_json in response_jsons:
            if (
                (choices := response_json.get("choices"))
                and (choice := choices[0])
                and (message := choice["message"])
                and (content := message["content"])
            ):
                try:
                    parsed_batch = json.loads(content.replace("'", "'"))
                    self.parsed.update(parsed_batch)
                except Exception as e:
                    logger.warning(f"couldn't parse: {content=}\n{e}")
            else:
                logger.warning(f"response could not be parsed: {response_json}")

        return self.parsed

    async def get_async_response(
        self,
        input_texts: Sequence[str],
        session: aiohttp.ClientSession | None = None,
        save: bool = False,
        save_path: str | Path = "",
    ) -> LLMResponseJSON:
        if session is None:
            session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True))

        retries = 0
        headers = {"Authorization": f"Bearer {self.__api_token}", "Content-Type": "application/json"}
        body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": self.config.make_multi_texts_prompt(input_texts=input_texts)}],
        }

        while retries < self.config.max_retries:
            async with asyncio.Semaphore(1):
                async with session.post(self.config.openai_endpoint, headers=headers, json=body) as response:
                    if response.status != 200:
                        logger.warning(f"OpenAI request failed: {await response.json()}")
                        retries += 1
                        await asyncio.sleep(random.randint(30, 120))
                        continue
                    response_json = await response.json()
                    if save and save_path:
                        await self.parse_and_save_response((response_json,), save_path=save_path)
                    self.successful_batches_count += 1

                    return response_json

        logger.warning("Max retries exceeded")
        return {"choices": []}

    async def get_multiple_extraction_responses(
        self, texts: Iterable[str], save: bool = False, save_path: str | Path = ""
    ) -> list[LLMResponseJSON]:
        self.successful_batches_count = 0
        session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True))

        batches = list(batched(texts, self.config.multi_request_batch_size))
        tasks = [
            asyncio.ensure_future(self.get_async_response(batch, session=session, save=save, save_path=save_path))
            for batch in batches
        ]
        results = await asyncio.gather(*tasks)
        logger.debug(f"{len(results)=}")
        return list(results)

    def extract_multiple(
        self, texts: Iterable[str], save: bool = False, save_path: str | Path = ""
    ) -> list[ExtractedObjectsDict]:
        async_result = asyncio.run(self.get_multiple_extraction_responses(texts, save=save, save_path=save_path))
        logger.debug(f"{len(async_result)=}")
        parsed_responses = self.parse_multi_text_response(async_result)
        return list(parsed_responses.values())


if __name__ == "__main__":
    import pandas as pd

    data_dir = Path(__file__).parent.parent.parent / "data"

    val_df = pd.read_csv(data_dir / "val.csv")  # pyright: ignore
    save_path = data_dir / "labels_val.csv"
    texts = val_df["text"].tolist()  # pyright: ignore
    logger.info(f"{len(texts)=}")  # pyright: ignore
    logger.info(f"{len(set(texts))=}")  # pyright: ignore
    logger.info(texts[:5])
    logger.info(texts[-5:])
    # text = "Furry white rabbit"
    # texts = [text, "Yellow car is driving on dirty road"]
    extractor = LLMExtractor(extractor_settings)

    extracted = extractor.extract_multiple(texts=texts, save=True, save_path=save_path)  # pyright: ignore
    logger.info(f"{extractor.successful_batches_count=}")
    logger.info(f"{len(extracted)=}")
