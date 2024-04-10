from collections import defaultdict
from dataclasses import dataclass

import spacy

from src.core.models import ExtractedObjectsDict, LanguageDependency, PartOfSpeech
from src.extractor.base import BaseObjectsExtractor


@dataclass
class Token:
    index: int
    text: str


class PosExtractor(BaseObjectsExtractor):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str) -> ExtractedObjectsDict:
        objects: dict[str, list[str]] = defaultdict(list)
        doc = self.nlp(text.lower())
        nouns = [
            Token(index=index, text=token.text)
            for index, token in enumerate(doc)
            if token.pos_ in {PartOfSpeech.noun, PartOfSpeech.proper_noun}
        ]
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
        adjectives = [
            Token(index=index, text=token.text)
            for index, token in enumerate(doc)
            if token.pos_ == PartOfSpeech.adjective
        ]
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
