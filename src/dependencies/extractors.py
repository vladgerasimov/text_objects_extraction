from pathlib import Path

from src.extractor import PosExtractor

configs_path = Path(__file__).parent.parent.parent / "configs"


def get_pos_extractor() -> PosExtractor:
    return PosExtractor()
