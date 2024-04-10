import json
import re

import spacy
from loguru import logger

from src.core.models import ExtractedObjectsDict, LanguageDependency, PartOfSpeech

nlp = spacy.load("en_core_web_sm")
word_quote_pattern = re.compile(r"([a-zA-Z])'([a-zA-Z])")


def parse_texts(raw_string: str) -> list[str] | None:
    """
    Used to parse JSON from original dataset
    :param raw_string: JSON as string
    :return: list[str] | None - parsed list of sentences or None (if couldn't parse)
    """
    try:
        return json.loads(
            word_quote_pattern.sub(r"\1::\2", raw_string)
            .replace("'", '"')
            .replace("\n", ",")
            .replace("::", "'")
            .replace('" "', '", "')
        )
    except json.decoder.JSONDecodeError:
        return


def text_has_objectives(text: str) -> bool:
    doc = nlp(text)
    return PartOfSpeech.adjective in {token.pos_ for token in doc}


def make_labels(text: str) -> ExtractedObjectsDict:
    doc = nlp(text)

    objects: dict[str, list[str]] = {}

    for token in doc:
        if token.pos_ in [PartOfSpeech.noun, PartOfSpeech.proper_noun]:
            adjectives = [
                child.text for child in token.children if child.dep_ == LanguageDependency.adjectival_modifier
            ]
            objects[token.text] = adjectives

    return {"objects": objects}


if __name__ == "__main__":
    import pandas as pd

    val = pd.read_csv("data/val_clean.csv")  # pyright: ignore
    texts = val["text"]  # pyright: ignore
    for text in texts.iloc[:32]:  # pyright: ignore
        logger.info(text)  # pyright: ignore
        logger.info(make_labels(text))  # pyright: ignore
