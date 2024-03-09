import streamlit as st


from prompt_templates import CHAT_PROMPT_TEMPLATE
from components import model_settings
from utils.inference import generate, generate_stream, healthcheck

model_settings(
    default_generation_kwarg_overrides={
        "max_new_tokens": 100,
        "repetition_penalty": 1.1,
        "decoding_strategy": "sample",
    }
)

if not healthcheck():
    st.error("Cannot connect to inference endpoint...")
else:
    generation_parameters: dict = st.session_state.generation_parameters

    st.title("ChatBot")
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
            with st.spinner("..."):
                messages_history_string = "\n\n".join(
                    [
                        f"{message['role'].title()}: {message['content']}"
                        for message in st.session_state.messages
                    ]
                )
                input = CHAT_PROMPT_TEMPLATE.format(messages=messages_history_string)
                generation_parameters["stop_sequences"] = ["User:"]
                response = generate_stream(
                    input=input,
                    generation_params=generation_parameters
                )
                response_msg = st.write_stream(response)

        st.session_state.messages.append({"role": "assistant", "content": response_msg})
    st.button("Clear chat history", on_click=clear_messages)
