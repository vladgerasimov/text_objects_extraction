import re
from collections import defaultdict
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc

from src.core.models import ExtractedObjectsDict, LanguageDependency, PartOfSpeech
from src.extractor.base import BaseObjectsExtractor

multiple_spaces_pattern = re.compile(r"\s{2,}")


@dataclass(frozen=True, slots=True)
class Token:
    index: int
    text: str


def preprocess_text(text: str) -> str:
    return re.sub(multiple_spaces_pattern, " ", text).lower().strip()


class PosExtractor(BaseObjectsExtractor):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def get_doc(self, text: str) -> Doc:
        return self.nlp(preprocess_text(text))

    def get_nouns(self, text: str | None = None, doc: Doc | None = None) -> list[Token]:
        if not text and not doc:
            raise ValueError("text or doc must be passed")
        if not doc and text:
            doc = self.get_doc(text)

        return [
            Token(index=index, text=token.text)
            for index, token in enumerate(doc)  # pyright: ignore
            if token.pos_ in {PartOfSpeech.noun, PartOfSpeech.proper_noun}
        ]

    def get_adjectives(self, text: str | None = None, doc: Doc | None = None) -> list[Token]:
        if not text and not doc:
            raise ValueError("text or doc must be passed")
        if not doc and text:
            doc = self.get_doc(text)

        return [
            Token(index=index, text=token.text)
            for index, token in enumerate(doc)  # pyright: ignore
            if token.pos_ == PartOfSpeech.adjective
        ]

    def extract(self, text: str) -> ExtractedObjectsDict:
        objects: dict[str, list[str]] = defaultdict(list)
        doc = self.get_doc(text)
        nouns = self.get_nouns(doc=doc)
        if not nouns:
            return {"objects": {}}

        nouns_without_adjectives = [
            token.text
            for token in doc
            if token.pos_ == PartOfSpeech.noun
            and LanguageDependency.adjectival_modifier not in {child.dep_ for child in token.children}
        ]
        for noun in nouns_without_adjectives:
            objects[noun] = []
        adjectives = self.get_adjectives(doc=doc)
        adj_noun_mapping: dict[str, str] = {}
        for adjective in adjectives:
            min_distance = float("inf")
            corresponding_noun = nouns[0]
            for noun in nouns:
                distance = abs(adjective.index - noun.index)
                if distance < min_distance:
                    min_distance = distance
                    corresponding_noun = noun

            adj_noun_mapping[adjective.text] = corresponding_noun.text

        for adjective, noun in adj_noun_mapping.items():
            objects[noun].append(adjective)

        return {"objects": dict(objects)}
