from src.core.settings import CONFIGS_DIR
from src.extractor.bert_extractor import BertExtractor, BertExtractorSettings
from src.extractor.llm_extractor import LLMExtractor, LLMExtractorSettings
from src.extractor.pos_extractor import PosExtractor

# configs_path = Path(__file__).parent.parent.parent / "configs"


def get_pos_extractor() -> PosExtractor:
    return PosExtractor()


def get_llm_extractor() -> LLMExtractor:
    return LLMExtractor(LLMExtractorSettings.from_yaml(CONFIGS_DIR / "llm_extractor_settings.yaml"))


def get_bert_extractor() -> BertExtractor:
    return BertExtractor(BertExtractorSettings.from_yaml(CONFIGS_DIR / "bert_extractor_settings.yaml"))
