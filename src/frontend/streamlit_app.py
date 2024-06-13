import streamlit as st

from src.core.settings import CONFIGS_DIR, ExtractorClientSettings
from src.extractor.client import ExtractorClient
from src.image_generator import DallEGenerator, DallESettings

config = ExtractorClientSettings.from_yaml(
    config_path=CONFIGS_DIR / "extractor_client_settings.yaml", common_settings_path=CONFIGS_DIR / "app_settings.yaml"
)
client = ExtractorClient(config=config)

config_generator = DallESettings.from_yaml(CONFIGS_DIR / "dall_e_settings.yaml")
image_generator = DallEGenerator(config=config_generator)


def extract_objects(text: str):
    with st.spinner("Extracting objects from text..."):
        st.session_state.extracted_objects = client.extract_objects(text)


def receive_user_input():
    input_field = st.text_input(label="Enter text", key="input_field")
    extract_button = st.button("Extract objects")  # , on_click=extract_objects, args=(input_field,))
    if extract_button:
        extract_objects(input_field)


def display_extracted_objects():
    if extracted_objects := st.session_state.get("extracted_objects"):
        st.write("Extracted objects:", extracted_objects)  # pyright: ignore [reportUnknownMemberType]


def display_generated_image():
    if extracted_objects := st.session_state.get("extracted_objects"):
        image_button = st.button("Generate image")
        if image_button:
            with st.spinner("Generating image..."):
                generated_url = image_generator.get_generated_url(input_json=extracted_objects)
            st.image(generated_url, caption="Generated image")


def main():
    st.title("Object extraction")
    receive_user_input()
    display_extracted_objects()
    display_generated_image()


if __name__ == "__main__":
    main()
