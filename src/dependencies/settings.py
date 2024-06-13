from src.core.settings import CONFIGS_DIR, AppSettings


def get_app_settings() -> AppSettings:
    return AppSettings.from_yaml(CONFIGS_DIR / "app_settings.yaml")
