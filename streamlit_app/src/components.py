import streamlit as st
from torch.cuda import is_available as cuda_is_available
from models import CPU_MODELS, GPU_MODELS, AppModel

MODEL_CONFIGS = CPU_MODELS + GPU_MODELS

def model_settings(include_gen_params=True, 
                   default_generation_kwarg_overrides={}, 
                #    default_model_kwarg_overrides = {}
                   ):
    with st.sidebar:
        if cuda_is_available():
            st.success("CUDA Available")
        else:
            st.warning("CUDA Unavailable")

        device_map = st.selectbox(
            "Device Map", 
            options=["auto", "cpu"], 
            help="'auto' will use GPU if available, while 'cpu' will always use CPU for inference, even if a GPU is available"
            # disabled=True
        )
        if device_map == "cpu" or not cuda_is_available():
            st.session_state.available_model_names = [m["model_name"] for m in CPU_MODELS]
        else:
            st.session_state.available_model_names = [m["model_name"] for m in CPU_MODELS + GPU_MODELS]
        with st.form("model_selector"):
            st.header("Model Selector",
                      help="Select a Large Language Model to use. (Available options depend on device map. GPTQ models can only run on GPU)")

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
                model_config = [m for m in MODEL_CONFIGS if m["model_name"] == model_name][0]
                with st.spinner("Loading model"):
                    st.session_state["model"] = AppModel(
                        **model_config,device_map=device_map
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
                        if not k in st.session_state:
                            st.session_state[k] = v
            do_sample = st.radio(
                "Decoding Strategy",
                options=[False, True],
                format_func=lambda x: "Greedy" if x == False else "Sample",
                key="do_sample",
                help="Token decoding strategy. 'Greedy' will select the most likely token, while 'sample' will sample from a set of 'k' most likely tokens"
            )
            with st.form("Generation Parameters"):
                st.header("Generation Parameters",
                          help="Parameters that control text generation. For detailed parameter information, see https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/text_generation#transformers.GenerationConfig")

                st.number_input(
                    "Min New Tokens", min_value=1, max_value=None, value=1, key="min_new_tokens",
                    help="Minimum number of output tokens in generated response. (Applies to both greedy and sample decoding)"
                )
                st.number_input(
                    "Max New Tokens", min_value=1, max_value=None, value=25, key="max_new_tokens",
                    help="Maximum number of output tokens in generated response. (Applies to both greedy and sample decoding)"
                )
                st.number_input(
                    "Repetition Penalty", min_value=1.0, max_value=2.0, value=1.0, key="repetition_penalty",
                    help="Penalization factor for repeated tokens. (Applies to both greedy and sample decoding)"
                )

                if do_sample:
                    st.number_input(
                        "Temperature", min_value=0.0, max_value=1.4, value=0.5, key="temperature",
                        help="Normalization factor for token probabilities. Higher numbers = greater normalization = greater likelihood of selecting less-likely tokens = greater 'creativity' factor. (Applies to only sample decoding)"
                    )
                    st.number_input(
                        "Number of Beams", min_value=1, max_value=None, value=1, key="num_beams",
                        help="Number of token generation pathways to take at each generation step. (Applies to only sample decoding)"
                    )
                    st.number_input(
                        "Num Return Sequences", min_value=1, max_value=None, value=1, key="num_return_sequences",
                        help="Number of candidate output sequences to return in response to the prompt. (Applies to only sample decoding)"
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
