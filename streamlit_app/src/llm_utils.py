from logging import getLogger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM
from langchain.prompts import PromptTemplate
from torch.cuda import is_available as cuda_is_available
from prompt_templates import TINYLLAMA_DEFAULT
from auto_gptq import exllama_set_max_input_length

log = getLogger(__name__)

CPU_MODEL_NAMES = [
    "PY007/TinyLlama-1.1B-Chat-v0.3",
    # "TheBloke/Llama-2-13B-chat-GGML"
]

GPU_MODEL_NAMES = [
    "TheBloke/samantha-mistral-7B-GPTQ",
    "TheBloke/Llama-2-7B-chat-GPTQ",
    "TheBloke/Llama-2-13B-chat-GPTQ",
    # "TheBloke/CollectiveCognition-v1.1-Mistral-7B-GPTQ",
    "TheBloke/Airoboros-L2-13B-3.1.1-GPTQ",
    "TheBloke/Mythalion-13B-GPTQ",
    "TheBloke/Athena-v3-GPTQ",
    "TheBloke/MXLewd-L2-20B-GPTQ",
]


class AppModel:
    def __init__(self, model_name, device_map="auto"):
        self._model_name = model_name
        self._device_map = device_map
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map=self._device_map,
        )
        try:
            exllama_set_max_input_length(self._model, 4096)
        except (AttributeError, ValueError):
            pass
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device_map,
        )

        self._pipeline = pipeline(
            task="text-generation", model=self._model, tokenizer=self._tokenizer
        )

    def run(
        self,
        inputs: dict,
        prompt_template: PromptTemplate = None,
        min_new_tokens=1,
        max_new_tokens=20,
        repetition_penalty=1.0,
        do_sample=False,
        temperature=0.5,
        num_beams=1,
        num_return_sequences=1,
        remove_tokens=["<s>", "</s>"],
        stop_sequences=[],
    ):
        if prompt_template is None:
            prompt_template = TINYLLAMA_DEFAULT

        prompt = prompt_template.format(**inputs)

        if self._device_map == "auto" and cuda_is_available():
            input_tensor = self._tokenizer.encode(prompt, return_tensors="pt").to(
                "cuda"
            )
        else:
            input_tensor = self._tokenizer.encode(prompt, return_tensors="pt")

        output_tensor = self._model.generate(
            input_tensor,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )

        generated_tensor = output_tensor[:, input_tensor.shape[1] :]

        ## The batch_decode call below removes the input tokens
        generated_text = self._tokenizer.batch_decode(generated_tensor)[0]

        for sequence in stop_sequences:
            generated_text = generated_text.split(sequence)[0]
        for token in remove_tokens:
            generated_text = generated_text.replace(token, "")
        generated_text = generated_text.strip('" \n')

        output = {
            "text": generated_text,
            "output_token_length": len(generated_tensor[0]),
        }
        return output
