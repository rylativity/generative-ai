import streamlit as st
from llm_utils import MODEL_NAMES, AppModel
from prompt_templates import LLAMA2_DEFAULT
from langchain.prompts import PromptTemplate
from torch.cuda import is_available as cuda_is_available
from string import Formatter


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
        with st.expander("Advanced Prompting"):
            use_custom_prompt = st.checkbox("Custom Prompt")
            if use_custom_prompt:
                st.session_state.prompt_template = st.text_area("Prompt Template", value=LLAMA2_DEFAULT.template)
                st.caption("Type your prompt template above. Variables should be surrounded by curly braces (e.g. {input_string})")

                st.session_state.prompt_variables = names = [fn for _, fn, _, _ in Formatter().parse(st.session_state.prompt_template) if fn is not None]
                st.write(f"Found variables in prompt template: {st.session_state.prompt_variables}")
                    

    with st.container():
        with st.form("Generation"):
            with st.sidebar:
                st.header("Generation Parameters")
                st.session_state.min_new_tokens = st.number_input(
                    "min_new_tokens", min_value=1, max_value=1000, value=1
                )

                st.session_state.max_new_tokens = st.number_input(
                    "max_new_tokens", min_value=1, max_value=1000, value=200
                )

                st.session_state.repetition_penalty = st.slider(
                    "repetition_penalty", min_value=1.0, max_value=2.0, value=1.0
                )
            
            if use_custom_prompt:

                for variable in st.session_state.prompt_variables:
                    if variable == "context":
                        default_value = "The following is a common proverb about journeys. 'A journey of a thousand miles begins with a single step'"
                    elif variable == "input":
                        default_value = "Explain the meaning of the proverb above to me as if I am 10 years old."
                    else:
                        default_value = None

                    st.session_state[f"prompt_var_{variable}"] = st.text_area(variable, value=default_value)
                inputs = {
                    variable:st.session_state[f"prompt_var_{variable}"]
                      for variable in st.session_state.prompt_variables
                }
                prompt_template = PromptTemplate(template=st.session_state.prompt_template, input_variables=st.session_state.prompt_variables)
            else:
                text_input = st.text_area("Text Input", key="text_input")
                inputs =  {"input": text_input}
                prompt_template = PromptTemplate(template="{input}",input_variables=["input"])

            def form_is_valid(error_banner=False):
                if st.session_state.max_new_tokens < st.session_state.min_new_tokens:
                    if error_banner:
                        st.error("Min tokens is greater than max tokens")
                    return False
                elif st.session_state.min_new_tokens < 1 or st.session_state.max_new_tokens < 1:
                    if error_banner:
                        st.error("Min and max tokens must be greater than 1")
                    return False
                else:
                    return True
            
            if st.checkbox("Show Formatted Prompt"):
                st.header("Formatted Prompt")
                st.write(prompt_template.format(**inputs))

            generate_submit = st.form_submit_button("Generate", disabled=not form_is_valid())
            if generate_submit and form_is_valid(error_banner=True):
                with st.spinner("Generating"):
                    output = model.run(
                       inputs=inputs,
                       prompt_template=prompt_template,
                        min_new_tokens=st.session_state.min_new_tokens,
                        max_new_tokens=st.session_state.max_new_tokens,
                        repetition_penalty=st.session_state.repetition_penalty,
                    )

        if generate_submit and form_is_valid():
            st.write(output["text"])
            st.caption(f"Generated tokens: {output['output_token_length']}")
    
    