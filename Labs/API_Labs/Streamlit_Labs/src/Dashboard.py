# Dashboard.py
import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FASTAPI_IRIS_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'src' / 'iris_model.pkl'
LOGGER = get_logger(__name__)

def run():
    st.set_page_config(page_title="Iris Flower Prediction Demo", page_icon="🪻")

    with st.sidebar:
        st.write("### Backend")
        try:
            backend_request = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/health", timeout=3)
            if backend_request.status_code == 200 and backend_request.json().get("status") == "ok":
                st.success("Backend online ✅")
            else:
                st.warning("Problem connecting 😭")
        except requests.RequestException as ce:
            LOGGER.error(ce)
            st.error("Backend offline 😱")

        st.info("Configure parameters")

        # Toggle input mode
        input_mode = st.radio("Input mode", ["Sliders", "Upload JSON"], horizontal=True)

        client_payload = None

        if input_mode == "Upload JSON":
            test_input_file = st.file_uploader("Upload test prediction file (.json)", type=["json"])
            if test_input_file:
                try:
                    test_input_data = json.load(test_input_file)
                    st.write("Preview:")
                    st.json(test_input_data)
                    # Accept either {"input_test": {...}} or flat {...}
                    if "input_test" in test_input_data:
                        client_payload = test_input_data["input_test"]
                    else:
                        client_payload = test_input_data
                except Exception as e:
                    st.error("Invalid JSON file.")
                    LOGGER.error(e)
        else:
            # Sliders
            sepal_length = st.slider("Sepal Length", 4.3, 7.9, 5.1, 0.1, help="cm", format="%f")
            sepal_width  = st.slider("Sepal Width",  2.0, 4.4, 3.5, 0.1, help="cm", format="%f")
            petal_length = st.slider("Petal Length", 1.0, 6.9, 1.4, 0.1, help="cm", format="%f")
            petal_width  = st.slider("Petal Width",  0.1, 2.5, 0.2, 0.1, help="cm", format="%f")

            client_payload = {
                "sepal_length": float(sepal_length),
                "sepal_width":  float(sepal_width),
                "petal_length": float(petal_length),
                "petal_width":  float(petal_width),
            }

        predict_button = st.button("Predict")

    st.write("# Iris Flower Prediction! 🪻")
    result_container = st.empty()

    if predict_button:
        if not FASTAPI_IRIS_MODEL_LOCATION.is_file():
            LOGGER.warning("iris_model.pkl not found in FastAPI Lab. Run train.py first.")
            st.toast(":red[Model iris_model.pkl not found. Please run train.py in FastAPI Lab]", icon="🔥")
            return

        if not client_payload:
            st.toast(":red[No input provided. Use sliders or upload a JSON file.]", icon="🔴")
            return

        try:
            with st.spinner("Predicting..."):
                # IMPORTANT: send JSON correctly
                resp = requests.post(f"{FASTAPI_BACKEND_ENDPOINT}/predict", json=client_payload, timeout=5)
            if resp.status_code == 200:
                iris_content = resp.json()
                label = iris_content.get("label", "unknown")
                result_container.success(f"The flower predicted is: **{label}**")
            else:
                st.toast(f':red[Server returned {resp.status_code}. Refresh and check backend status]', icon="🔴")
        except Exception as e:
            LOGGER.error(e)
            st.toast(":red[Problem contacting backend. Refresh and check status]", icon="🔴")

if __name__ == "__main__":
    run()