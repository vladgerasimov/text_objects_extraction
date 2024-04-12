import streamlit as st

from src.core.settings import CONFIGS_DIR, ExtractorClientSettings
from src.extractor.client import ExtractorClient

config = ExtractorClientSettings.from_yaml(
    config_path=CONFIGS_DIR / "extractor_client_settings.yaml", common_settings_path=CONFIGS_DIR / "app_settings.yaml"
)
client = ExtractorClient(config=config)


def extract_objects(text: str):
    st.session_state.extracted_objects = client.extract_objects(text)


def receive_user_input():
    input_field = st.text_input(label="Enter text", key="input_field")
    st.button("Extract objects", on_click=extract_objects, args=(input_field,))


def display_extracted_objects():
    if extracted_objects := st.session_state.get("extracted_objects"):
        st.write("Extracted objects:", extracted_objects)  # pyright: ignore [reportUnknownMemberType]


def main():
    st.title("Object extraction")
    receive_user_input()
    display_extracted_objects()


if __name__ == "__main__":
    main()
