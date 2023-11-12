import streamlit as st
from torch.cuda import is_available as cuda_is_available
from llm_utils import CPU_MODEL_NAMES, GPU_MODEL_NAMES, AppModel

def model_settings(include_gen_params=True):
    with st.sidebar:
        if cuda_is_available():
            st.success("CUDA Available")
        else:
            st.warning("CUDA Unavailable")
        disable_cuda = st.checkbox("Disable CUDA (Being implemented)", key="disable_cuda", disabled=True)
        if disable_cuda or not cuda_is_available():
                st.session_state.available_model_names = CPU_MODEL_NAMES
        else:
            st.session_state.available_model_names = CPU_MODEL_NAMES + GPU_MODEL_NAMES
        with st.form("model_selector"):
            st.header("Model Selector")

            # if cuda_is_available():
            #     st.write(":white_check_mark: CUDA Available")
            # else:
            #     st.write(":no_entry_sign: CUDA Unavailable")
            # if disable_cuda:
            #     st.error("CUDA Disabled")
            # else:
            #     st.success("CUDA Enabled")

            model_name = st.selectbox(
                "Model", options=st.session_state.available_model_names, placeholder="Select a model...", index=None, key="model_name"
            )

            load_model = st.form_submit_button("Load Model")

            if load_model:
                with st.spinner("Loading model"):
                    if disable_cuda:
                        st.session_state["model"] = AppModel(model_name=model_name, device_map="cpu")
                    else:
                        st.session_state["model"] = AppModel(model_name=model_name)
                    st.write(f"Model {st.session_state.model._model_name} loaded successfully")
            if "model" in st.session_state:
                st.caption(f"Using Model {st.session_state.model._model_name}")
                st.link_button("Model Card", url=f"https://huggingface.co/{st.session_state.model._model_name}")
        
        
        if include_gen_params:
            do_sample=st.radio("Decoding Strategy", options=[False,True], format_func=lambda x: "Greedy" if x == False else "Sample")
            with st.form("Generation Parameters"):
                st.header("Generation Parameters")

                min_new_tokens=st.number_input("Min New Tokens", min_value=1, max_value=None, value=1)
                max_new_tokens=st.number_input("Max New Tokens", min_value=1, max_value=None, value=25)
                repetition_penalty=st.number_input("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.0)
                
                if do_sample:
                    temperature=st.number_input("Temperature", min_value=0.0, max_value=1.4, value=0.5)
                    num_beams=st.number_input("Number of Beams", min_value=1, max_value=None, value=1)
                    num_return_sequences=st.number_input("Num Return Sequences", min_value=1, max_value=None, value=1)
                
                params = {
                        "min_new_tokens": min_new_tokens,
                        "max_new_tokens": max_new_tokens,
                        "repetition_penalty": repetition_penalty,
                        "do_sample": do_sample,
                    }
                if do_sample:
                    params.update({
                        "temperature":temperature,
                        "num_beams": num_beams,
                        "num_return_sequences": num_return_sequences
                    })

                set_parameters = st.form_submit_button("Set Parameters")
                if set_parameters:
                    
                    st.session_state.generation_parameters = params
                    st.write(f"Parameters set successfully")

                if not "generation_parameters" in st.session_state:
                    st.session_state.generation_parameters = params
                st.header("Active Params")
                st.json(st.session_state.generation_parameters)
