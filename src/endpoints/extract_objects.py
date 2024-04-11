from fastapi import APIRouter, Body, Depends

from src.core.models import ExtractObjectsResponse
from src.core.settings import CONFIGS_DIR, AppSettings
from src.dependencies import get_pos_extractor
from src.extractor.base import BaseObjectsExtractor

router = APIRouter(tags=["extract"])
app_settings = AppSettings.from_yaml(config_path=CONFIGS_DIR / "app_settings.yaml", common_settings_path="")


@router.post("/extract")
def extract_objects(
    text: str = Body(embed=False), extractor: BaseObjectsExtractor = Depends(get_pos_extractor)
) -> ExtractObjectsResponse:
    if len(text) > (max_len := app_settings.text_max_len):
        text = text[:max_len]
        input_text_truncated = True
    else:
        input_text_truncated = False

    objects = extractor.extract(text=text)
    return ExtractObjectsResponse(result=objects, input_text_truncated=input_text_truncated)
