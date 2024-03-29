import streamlit as st

from prompt_templates import LLAMA2_DEFAULT
from langchain.prompts import PromptTemplate
from torch.cuda import is_available as cuda_is_available
from string import Formatter
from components import model_settings
from utils.inference import healthcheck, generate, generate_stream

## This will providfe us with model selection and generation parameter components.
model_settings()
# Model is accessible from st.session_state.model, and generation parameters are accessible from st.session_state.generation_parameters

if not healthcheck():
    st.error("Cannot connect to inference endpoint...")
else:
    st.title("Basic Prompting")

    generation_parameters: dict = st.session_state.generation_parameters

    with st.container():
        use_custom_prompt = st.checkbox("Custom Prompt")
        if use_custom_prompt:
            st.caption(
                    """
                    Type your prompt template below. Prompt variables/placeholders should be surrounded by curly braces (e.g. {input_string}).
                    
                    A form field will be created for each prompt variable/placeholder below.
                    """
                )
            with st.container(border=True):
               
                st.session_state.prompt_template = st.text_area(
                    "Prompt Template", value=LLAMA2_DEFAULT.template, height=350
                )
                

                st.session_state.prompt_variables = names = [
                    fn
                    for _, fn, _, _ in Formatter().parse(
                        st.session_state.prompt_template
                    )
                    if fn is not None
                ]
                st.caption(
                    f"Found variables in prompt template: {st.session_state.prompt_variables}"
                )

    with st.container():
        with st.form("Generation"):
            if use_custom_prompt:
                for variable in st.session_state.prompt_variables:
                    if variable == "context":
                        default_value = "The following is a common proverb about journeys. 'A journey of a thousand miles begins with a single step'"
                    elif variable == "input":
                        default_value = "Explain the meaning of the proverb above to me as if I am 10 years old."
                    else:
                        default_value = None

                    st.session_state[f"prompt_var_{variable}"] = st.text_area(
                        variable, value=default_value
                    )
                inputs = {
                    variable: st.session_state[f"prompt_var_{variable}"]
                    for variable in st.session_state.prompt_variables
                }
                prompt_template = PromptTemplate(
                    template=st.session_state.prompt_template,
                    input_variables=st.session_state.prompt_variables,
                )
            else:
                text_input = st.text_area("Text Input", key="text_input")
                inputs = {"input": text_input}
                prompt_template = PromptTemplate(
                    template="{input}", input_variables=["input"]
                )

            def form_is_valid(error_banner=False):
                if (
                    generation_parameters["max_new_tokens"]
                    < generation_parameters["min_new_tokens"]
                ):
                    if error_banner:
                        st.error("Min tokens is greater than max tokens")
                    return False
                elif (
                    generation_parameters["min_new_tokens"] < 1
                    or generation_parameters["max_new_tokens"] < 1
                ):
                    if error_banner:
                        st.error("Min and max tokens must be greater than 1")
                    return False
                else:
                    return True

            if st.checkbox("Show Formatted Prompt"):
                st.header("Formatted Prompt")
                st.write(prompt_template.format(**inputs))

            generate_submit = st.form_submit_button(
                "Generate", disabled=not form_is_valid()
            )
            if generate_submit and form_is_valid(error_banner=True):
                with st.spinner("Generating"):
                    input = prompt_template.format(**inputs)
                    output = generate(
                        input=input,
                        generation_params=generation_parameters
                    )

        if generate_submit and form_is_valid():
            st.header("Generated Text")
            st.write(output["text"])
            st.caption(f"Generated tokens: {output['output_token_length']}")
