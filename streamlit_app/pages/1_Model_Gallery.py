import streamlit as st
from llm_utils import MODEL_NAMES, AppModel
from torch.cuda import is_available as cuda_is_available

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
                st.session_state.model = AppModel(model_name=model_name)
            st.write(f"Model {model_name} loaded successfully")

if "model" in st.session_state:
    model = st.session_state.model

    with st.container():
        with st.form("Generation"):
            with st.sidebar:
                st.header("Generation Parameters")
                min_new_tokens = st.number_input(
                    "min_new_tokens", min_value=1, max_value=1000, value=1
                )

                max_new_tokens = st.number_input(
                    "max_new_tokens", min_value=min_new_tokens + 1, max_value=1000
                )

                repetition_penalty = st.slider(
                    "repetition_penalty", min_value=1.0, max_value=2.0, value=1.0
                )
            text_input = st.text_area("Text Input", key="text_input")

            generate_submit = st.form_submit_button("Generate")
            with st.spinner("Generating"):
                output = model.run(
                    {"input": text_input},
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )

        if generate_submit:
            st.text(output)
