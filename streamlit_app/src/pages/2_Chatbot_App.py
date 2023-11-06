import streamlit as st
from torch.cuda import empty_cache as cuda_empty_cache, is_available as cuda_is_available, OutOfMemoryError

from llm_utils import MODEL_NAMES, AppModel
from prompt_templates import CHAT_PROMPT_TEMPLATE

with st.sidebar:
    with st.form("Model Settings"):
        st.header("Model Settings")
        
        if cuda_is_available():
            st.success(f"CUDA Available")
        else:
            st.error(f"CUDA Unavailable")

        model_name = st.selectbox(
            "Model", options=MODEL_NAMES, placeholder="Select a model...", index=None
        )

        load_model = st.form_submit_button("Load Model")

        if load_model:
            try:
                if "model" in st.session_state:
                    del st.session_state.model
                    cuda_empty_cache()
                with st.spinner("Loading model"):
                    st.session_state.model = AppModel(model_name=model_name)
                st.write(f"Model {model_name} loaded successfully")
            except OutOfMemoryError as e:
                st.error("CUDA Out of Memory. Try reloading the model or restarting your runtime")

if not "model" in st.session_state:
    st.header("*Load a model to get started*")
else:
    model = st.session_state.model

    st.title("ChatBot")
    st.caption(f"Using model {model._model_name}")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    def clear_messages():
        st.session_state.messages=[]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("..."):
                messages_history_string = "\n\n".join([
                    f"{message['role'].title()}: {message['content']}" for message in st.session_state.messages
                ])
                response = model.run(
                    inputs = {"messages":messages_history_string},
                    prompt_template=CHAT_PROMPT_TEMPLATE,
                    max_new_tokens=300,
                    stop_sequences=["User:"]
                )["text"]
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.button("Clear chat history", on_click=clear_messages)