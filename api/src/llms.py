from enum import Enum
from datetime import datetime
import json

from logging import getLogger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM
from torch.cuda import is_available as cuda_is_available
from auto_gptq import exllama_set_max_input_length

from src.utils.elastic import index_document

INFERENCE_LOGGING_INDEX_NAME = "api-inference-log"

log = getLogger(__name__)


class ModelType(Enum):
    OTHER = "OTHER"
    GPTQ = "GPTQ"
    GGUF = "GGUF"
    AWQ = "AWQ"

FAVORITE_MODELS = [
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "TheBloke/Llama-2-13B-chat-GPTQ",
    "TheBloke/vicuna-13B-v1.5-16K-GGUF",
    "TheBloke/vicuna-13B-v1.5-16K-GPTQ",
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
]

CPU_MODELS = [
    {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
        "model_type": ModelType.OTHER
    },
    {
        "model_name": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "model_file": "mistral-7b-instruct-v0.2.Q8_0.gguf",
        "model_type": ModelType.GGUF,
        "tokenizer_model_name":"TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    },
    {
        "model_name": "TheBloke/Llama-2-7B-chat-GGUF",
        "model_file": "llama-2-7b-chat.Q4_K_M.gguf",
        "model_type": ModelType.GGUF,
        "tokenizer_model_name":"TheBloke/Llama-2-7B-GPTQ"
    },
    {
        "model_name": "TheBloke/Mistral-7B-OpenOrca-GGUF",
        "model_file": "mistral-7b-openorca.Q4_K_M.gguf",
        "model_type": ModelType.GGUF,
        "tokenizer_model_name":"TheBloke/Mistral-7B-OpenOrca-GPTQ"
    },
    {
        "model_name": "TheBloke/vicuna-13B-v1.5-16K-GGUF",
        "model_file": "vicuna-13b-v1.5-16k.Q4_K_M.gguf",
        "model_type": ModelType.GGUF,
        "tokenizer_model_name":"TheBloke/vicuna-13B-v1.5-16K-GPTQ"
    },
    {
        "model_name": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "model_file": "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
        "model_type": ModelType.GGUF,
        "tokenizer_model_name":"TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
    }
]

GPU_MODELS = [
    {
        "model_name": "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
        "model_type": ModelType.AWQ
    },
    {
        "model_name": "TheBloke/Mistral-7B-OpenOrca-GPTQ",
         "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
         "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/Llama-2-7B-chat-GPTQ",
         "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/openchat_3.5-16k-GPTQ",
         "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/orca_mini_13B-GPTQ",
         "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/Llama-2-13B-chat-GPTQ",
         "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/CollectiveCognition-v1.1-Mistral-7B-GPTQ",
        "model_type": ModelType.GPTQ,
    },
    {
        "model_name": "TheBloke/Airoboros-L2-13B-3.1.1-GPTQ",
        "model_type": ModelType.GPTQ,
    },
    {
        "model_name": "TheBloke/Mythalion-13B-GPTQ",
        "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/vicuna-13B-v1.5-16K-GPTQ",
        "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/Athena-v3-GPTQ",
        "model_type": ModelType.GPTQ
    },
    {
        "model_name": "TheBloke/MXLewd-L2-20B-GPTQ",
        "model_type": ModelType.GPTQ
    },
]


class AppModel:
    def __init__(
        self,
        model_name,
        model_type: str,
        model_file: str = None,
        tokenizer_model_name=None,
        device_map="auto",
        n_gpu_layers=30,
        context_length=4000,
        llama_cpp_threads=8
    ):
        self._model_name = model_name
        self._device_map = device_map
        self._model_type = model_type

        model_type_value = ModelType[self._model_type.upper()]
        if model_type_value in [ModelType.GPTQ, ModelType.AWQ, ModelType.OTHER]:
            # if model_file:
            #     self._model_file = model_file
            # else:
            #     pass

            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                device_map=self._device_map,
            )
        
        elif model_type_value in [ModelType.GGUF]:
            if not model_file:
                raise ValueError("Must provide model_file if using GGUF model")
            self._model_file = model_file

            model_path = hf_hub_download(self._model_name, filename=self._model_file)

            self._model = Llama(model_path=model_path, 
                                    n_ctx=context_length,  # The max sequence length to use - note that longer sequence lengths require much more resources
                                    n_threads=llama_cpp_threads,            # The number of CPU threads to use, tailor to your system and the resulting performance
                                    n_gpu_layers=n_gpu_layers        # The number of layers to offload to GPU, if you have GPU acceleration available
                                )
        else:
            raise Exception(f"No AppModel interface implemented for model type {self._model_type}")
        
        try:
            exllama_set_max_input_length(self._model, context_length)
        except (AttributeError, ValueError):
            pass
        
        if not tokenizer_model_name:
            tokenizer_model_name = self._model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model_name,
                device_map=device_map,
        )

        ## The below breaks with GGUF Model
        # self._pipeline = pipeline(
        #     task="text-generation", model=self._model, tokenizer=self._tokenizer
        # )

    def run(
        self,
        input: str,
        min_new_tokens=1,
        max_new_tokens=20,
        repetition_penalty=1.0,
        do_sample=False,
        top_p=1.0,
        top_k=50,
        typical_p=1.0,
        temperature=0.5,
        # num_beams=1,
        # num_return_sequences=1,
        remove_tokens=["<s>", "</s>"],
        stop_sequences=[],
        log_inferences=True,
        # stream=False
    ):
        original_prompt = input

        if ModelType[self._model_type.upper()] == ModelType.GGUF:
            generation_config = {
                "max_tokens":max_new_tokens,
                "temperature":temperature,
                "repeat_penalty":repetition_penalty,
                "top_k":top_k,
                "top_p":top_p,
                "typical_p":typical_p,
            }
            # generation_config["stop"] = generation_config["stop_sequences"]
            generation_config["echo"] = False

            if do_sample:
                pass
            else:
                generation_config["top_k"] = 1
            
            output_token_length = 0
            prompt = original_prompt
            while output_token_length < min_new_tokens:
                response = self._model(
                    prompt, 
                    **generation_config
                    )
                
                generated_text = response["choices"][0]["text"]
                
                output_token_length += response["usage"]["completion_tokens"]
                prompt += generated_text # In case we have to generate more text to meet minimum token count
        else:
            generation_config = {
            "min_new_tokens":min_new_tokens,
            "max_new_tokens":max_new_tokens,
            "do_sample":do_sample,
            "temperature":temperature,
            "repetition_penalty":repetition_penalty,
            "top_k":top_k,
            "top_p":top_p,
            "typical_p":typical_p,
                # "num_beams":num_beams,
                # "num_return_sequences":num_return_sequences,
                # "stop_sequences":stop_sequences
            }
            if self._device_map == "auto" and cuda_is_available():
                input_tensor = self._tokenizer.encode(original_prompt, return_tensors="pt").to(
                    "cuda"
                )
            else:
                input_tensor = self._tokenizer.encode(original_prompt, return_tensors="pt")
            
            output_tensor = self._model.generate(
                input_tensor,
                **generation_config
            )

            generated_tensor = output_tensor[:, input_tensor.shape[1] :]
            output_token_length = len(generated_tensor[0])
            ## The batch_decode call below removes the input tokens
            generated_text = self._tokenizer.batch_decode(generated_tensor)[0]

        for sequence in stop_sequences:
            generated_text = generated_text.split(sequence)[0]
        for token in remove_tokens:
            generated_text = generated_text.replace(token, "")
        generated_text = generated_text.strip('" \n')

        output = {
            "text": generated_text,
            "output_token_length": output_token_length,
        }

        if log_inferences:
            try:
                payload = {
                    "prompt":original_prompt,
                    "output":generated_text,
                    "timestamp":datetime.now().isoformat(),
                    "generation_params":generation_config
                }
                index_document(
                    index_name=INFERENCE_LOGGING_INDEX_NAME,
                    doc=payload,
                    create_missing_index=True
                )
            except Exception as e:
                log.error(e)
            
        return output


    async def stream(
        self,
        input: str,
        min_new_tokens=1,
        max_new_tokens=20,
        repetition_penalty=1.0,
        do_sample=False,
        top_p=1.0,
        top_k=50,
        typical_p=1.0,
        temperature=0.5,
        # num_beams=1,
        # num_return_sequences=1,
        remove_tokens=["<s>", "</s>"],
        stop_sequences=[],
        log_inferences=True,
    ):
        
        if ModelType[self._model_type.upper()] == ModelType.GGUF:
            generation_config = {
                "max_tokens":max_new_tokens,
                "temperature":temperature,
                "repeat_penalty":repetition_penalty,
                "top_k":top_k,
                "top_p":top_p,
                "typical_p":typical_p,
            }
            # generation_config["stop"] = generation_config["stop_sequences"]
            generation_config["echo"] = False

            if do_sample:
                pass
            else:
                generation_config["top_k"] = 1
            
            res = self._model(
                input, 
                stream=True,
                **generation_config
            )
            
            full_text = ""
            for chunk in res:
                token_obj = chunk["choices"][0]
                text = token_obj["text"]
                
                # Remove remove_tokens
                if text in remove_tokens:
                    text = ""
                
                full_text += text
                yield json.dumps(token_obj)+"\n"

                # Break on stop_sequences
                if len([seq for seq in stop_sequences if seq in full_text]) > 0:
                    break   

            if log_inferences:
                try:
                    payload = {
                        "prompt":input,
                        "output":full_text,
                        "timestamp":datetime.now().isoformat(),
                        "generation_kwargs":generation_config,
                        "generation_kwargs.stream":True
                    }
                    index_document(
                        index_name=INFERENCE_LOGGING_INDEX_NAME,
                        doc=payload,
                        create_missing_index=True
                    )
                except Exception as e:
                    log.error(e)
                    
        else:
            raise Exception("Only supports GGUF models")

            