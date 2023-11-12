import streamlit as st

from llm_utils import AppModel
from prompt_templates import CHAT_PROMPT_TEMPLATE
from components import model_settings

model_settings(default_generation_kwarg_overrides={"max_new_tokens":1000,"repetition_penalty":1.1})

if not "model" in st.session_state:
    st.header("*Load a model to get started*")
else:
    model: AppModel = st.session_state.model
    generation_parameters: dict = st.session_state.generation_parameters

    st.title("ChatBot")
    st.caption(f"Using model {model._model_name}")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def clear_messages():
        st.session_state.messages = []

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
                messages_history_string = "\n\n".join(
                    [
                        f"{message['role'].title()}: {message['content']}"
                        for message in st.session_state.messages
                    ]
                )
                response = model.run(
                    inputs={"messages": messages_history_string},
                    prompt_template=CHAT_PROMPT_TEMPLATE,
                    stop_sequences=["User:"],
                    **generation_parameters,
                )["text"]
            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.button("Clear chat history", on_click=clear_messages)
