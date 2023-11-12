import streamlit as st
from torch.cuda import empty_cache as cuda_empty_cache, is_available as cuda_is_available, OutOfMemoryError

from llm_utils import MODEL_NAMES, AppModel
from prompt_templates import CHAT_PROMPT_TEMPLATE
from components import model_settings

model_settings()

if not "model" in st.session_state:
    st.header("*Load a model to get started*")
else:
    model: AppModel = st.session_state.model
    generation_parameters:dict = st.session_state.generation_parameters

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
                    stop_sequences=["User:"],
                    **generation_parameters
                )["text"]
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.button("Clear chat history", on_click=clear_messages)

