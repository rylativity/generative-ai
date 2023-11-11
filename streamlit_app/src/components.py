import streamlit as st
from torch.cuda import is_available as cuda_is_available
from llm_utils import MODEL_NAMES, AppModel



def default_layout_component():
    with st.sidebar:
        with st.form("Model Settings"):
            st.header("Model Settings")
            st.write(f"CUDA Available: {cuda_is_available()}")
            model_name = st.selectbox(
                "Model", options=MODEL_NAMES, placeholder="Select a model...", index=None
            )

            load_model = st.form_submit_button("Load Model")

            if load_model:
                with st.spinner("Loading model"):
                    st.session_state["model"] = AppModel(model_name=model_name)
                st.write(f"Model {model_name} loaded successfully")