from collections import defaultdict
from typing import Callable, TypeAlias

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

from src.core.models import ExtractedObjectsDict, ProcessAttentions
from src.core.settings import ROOT_DIR, BertExtractorSettings
from src.extractor.base import BaseObjectsExtractor
from src.extractor.pos_extractor import PosExtractor

Attentions_T: TypeAlias = tuple[torch.Tensor, ...]

DATA_DIR = ROOT_DIR / "data"


class BertExtractor(BaseObjectsExtractor):
    def __init__(self, config: BertExtractorSettings):
        self.config = config
        self.pos_extractor = PosExtractor()

        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(  # pyright: ignore
            config.pretrained_model_name, cache_dir=DATA_DIR
        )
        self.model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(  # pyright: ignore
            config.pretrained_model_name, output_attentions=True, cache_dir=DATA_DIR
        )

        self._get_attentions_mapping: dict[str, Callable[[Attentions_T], torch.Tensor]] = {
            ProcessAttentions.get_first: self.get_first_attention_block,
            ProcessAttentions.get_all_mean: self.get_all_attention_blocks,
        }
        self._preprocess_attentions_mapping: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            ProcessAttentions.get_first: self.get_first_attention_tensor,
            ProcessAttentions.get_all_mean: self.get_mean_attention_tensor,
        }

    @staticmethod
    def get_all_attention_blocks(attentions: Attentions_T) -> torch.Tensor:
        return torch.concat(attentions, dim=0)

    @staticmethod
    def get_first_attention_block(attentions: Attentions_T) -> torch.Tensor:
        return attentions[0]

    @staticmethod
    def get_first_attention_tensor(attention: torch.Tensor) -> torch.Tensor:
        return torch.mean(attention, dim=1).squeeze()[1:-1, 1:-1]

    def get_mean_attention_tensor(self, attention: torch.Tensor) -> torch.Tensor:
        return attention[: self.config.n_blocks_to_average].mean(dim=(0, 1))[1:-1, 1:-1]

    def get_net_attentions(self, text: str) -> Attentions_T:
        return self.model(**self.tokenizer(text, return_tensors="pt")).attentions

    def get_tokens_processed_attentions(self, attentions: Attentions_T) -> list[list[float]]: ...

    def extract(self, text: str) -> ExtractedObjectsDict:
        objects: dict[str, list[str]] = defaultdict(list)

        doc = self.pos_extractor.get_doc(text)
        nouns = self.pos_extractor.get_nouns(doc=doc)
        nouns_without_adjectives = {noun.text for noun in nouns}
        nouns_mask = torch.tensor([noun.index for noun in nouns])

        adjectives = self.pos_extractor.get_adjectives(doc=doc)

        tokenized_text = self.tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            attentions = self._get_attentions_mapping[self.config.process_attentions](
                self.model(**tokenized_text).attentions
            )
        avg_attention_weights = self._preprocess_attentions_mapping[self.config.process_attentions](attentions)

        adj_noun_mapping: dict[str, str] = {}
        for adj in adjectives:
            noun_idx: int = avg_attention_weights[adj.index, nouns_mask].argmax().item()  # pyright: ignore
            corresponding_noun = nouns[noun_idx].text
            adj_noun_mapping[adj.text] = corresponding_noun
            nouns_without_adjectives.discard(corresponding_noun)

        for adj, noun in adj_noun_mapping.items():
            objects[noun].append(adj)

        for noun in nouns_without_adjectives:
            objects[noun] = []

        return {"objects": dict(objects)}

    def get_adjectives_attentions(self, text: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = defaultdict(dict[str, float])

        doc = self.pos_extractor.get_doc(text)
        nouns = self.pos_extractor.get_nouns(doc=doc)
        adjectives = self.pos_extractor.get_adjectives(doc=doc)

        tokenized_text = self.tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            attentions = self._get_attentions_mapping[self.config.process_attentions](
                self.model(**tokenized_text).attentions
            )
        avg_attention_weights = self._preprocess_attentions_mapping[self.config.process_attentions](attentions)
        for adj in adjectives:
            for noun in nouns:
                result[adj.text][noun.text] = avg_attention_weights[adj.index, noun.index].item()

        return dict(result)


if __name__ == "__main__":
    from src.core.settings import CONFIGS_DIR

    extractor = BertExtractor(BertExtractorSettings.from_yaml(CONFIGS_DIR / "bert_extractor_settings.yaml"))
    text = "beautiful furry rabbit in fresh snow"
    result = extractor.extract(text)

    print(result)
    attentions = extractor.get_adjectives_attentions(text)
