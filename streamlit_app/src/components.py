import streamlit as st
from torch.cuda import is_available as cuda_is_available
from llm_utils import CPU_MODEL_NAMES, GPU_MODEL_NAMES, AppModel


def model_settings(
    include_gen_params=True,
    default_generation_kwarg_overrides={},
    #    default_model_kwarg_overrides = {}
):
    with st.sidebar:
        if cuda_is_available():
            st.success("CUDA Available")
        else:
            st.warning("CUDA Unavailable")

        device_map = st.selectbox(
            "Device Map", options=["auto", "cpu"]  # , disabled=True
        )
        if device_map == "cpu" or not cuda_is_available():
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
                "Model",
                options=st.session_state.available_model_names,
                placeholder="Select a model...",
                index=None,
                key="model_name",
            )

            load_model = st.form_submit_button("Load Model")

            if load_model:
                with st.spinner("Loading model"):
                    st.session_state["model"] = AppModel(
                        model_name=model_name, device_map=device_map
                    )
                    st.write(
                        f"Model {st.session_state.model._model_name} loaded successfully"
                    )
            if "model" in st.session_state:
                st.caption(f"Using Model {st.session_state.model._model_name}")
                st.link_button(
                    "Model Card",
                    url=f"https://huggingface.co/{st.session_state.model._model_name}",
                )

        if include_gen_params:
            if default_generation_kwarg_overrides:
                for k, v in default_generation_kwarg_overrides.items():
                    st.session_state[k] = v
            do_sample = st.radio(
                "Decoding Strategy",
                options=[False, True],
                format_func=lambda x: "Greedy" if x == False else "Sample",
                key="do_sample",
            )
            with st.form("Generation Parameters"):
                st.header("Generation Parameters")

                st.number_input(
                    "Min New Tokens",
                    min_value=1,
                    max_value=None,
                    value=1,
                    key="min_new_tokens",
                )
                st.number_input(
                    "Max New Tokens",
                    min_value=1,
                    max_value=None,
                    value=25,
                    key="max_new_tokens",
                )
                st.number_input(
                    "Repetition Penalty",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.0,
                    key="repetition_penalty",
                )

                if do_sample:
                    st.number_input(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.4,
                        value=0.5,
                        key="temperature",
                    )
                    st.number_input(
                        "Number of Beams",
                        min_value=1,
                        max_value=None,
                        value=1,
                        key="num_beams",
                    )
                    st.number_input(
                        "Num Return Sequences",
                        min_value=1,
                        max_value=None,
                        value=1,
                        key="num_return_sequences",
                    )

                params = {
                    "min_new_tokens": st.session_state.min_new_tokens,
                    "max_new_tokens": st.session_state.max_new_tokens,
                    "repetition_penalty": st.session_state.repetition_penalty,
                    "do_sample": st.session_state.do_sample,
                }
                if do_sample:
                    params.update(
                        {
                            "temperature": st.session_state.temperature,
                            "num_beams": st.session_state.num_beams,
                            "num_return_sequences": st.session_state.num_return_sequences,
                        }
                    )

                set_parameters = st.form_submit_button("Set Parameters")
                if set_parameters:
                    st.session_state.generation_parameters = params
                    st.write(f"Parameters set successfully")

                if not "generation_parameters" in st.session_state:
                    st.session_state.generation_parameters = params

                st.header("Active Params")
                st.json(st.session_state.generation_parameters)
