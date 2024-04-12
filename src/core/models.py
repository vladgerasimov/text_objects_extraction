from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel


class ExtractedObjectsDict(TypedDict):
    objects: dict[str, list[str]]


class ExtractObjectsResponse(BaseModel):
    result: ExtractedObjectsDict
    input_text_truncated: bool = False


class ExtractObjectsResponseDict(TypedDict):
    result: ExtractedObjectsDict
    input_text_truncated: bool


class PartOfSpeech(StrEnum):
    noun = "NOUN"
    proper_noun = "PROPN"
    adjective = "ADJ"
    verb = "VERB"


class LanguageDependency(StrEnum):
    adjectival_modifier = "amod"
