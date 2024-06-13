from fastapi import APIRouter, Body, Depends

from src.core.models import ExtractObjectsResponse
from src.core.settings import CONFIGS_DIR, AppSettings, ExtractorType
from src.dependencies import get_bert_extractor, get_llm_extractor, get_pos_extractor
from src.dependencies.settings import get_app_settings
from src.extractor import BertExtractor, PosExtractor
from src.extractor.llm_extractor import LLMExtractor

router = APIRouter(tags=["extract"])
app_settings = AppSettings.from_yaml(config_path=CONFIGS_DIR / "app_settings.yaml", common_settings_path="")


@router.post("/extract")
def extract_objects(
    text: str = Body(embed=False),
    app_settings: AppSettings = Depends(get_app_settings),
    pos_extractor: PosExtractor = Depends(get_pos_extractor),
    llm_extractor: LLMExtractor = Depends(get_llm_extractor),
    bert_extractor: BertExtractor = Depends(get_bert_extractor),
) -> ExtractObjectsResponse:
    if len(text) > (max_len := app_settings.text_max_len):
        text = text[:max_len]
        input_text_truncated = True
    else:
        input_text_truncated = False

    match app_settings.api_selected_extractor:
        case ExtractorType.pos_extractor:
            extractor = pos_extractor
        case ExtractorType.llm_extractor:
            extractor = llm_extractor
        case ExtractorType.bert_extractor:
            extractor = bert_extractor

    objects = extractor.extract(text=text)
    return ExtractObjectsResponse(result=objects, input_text_truncated=input_text_truncated)
