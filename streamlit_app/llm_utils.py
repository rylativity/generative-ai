from logging import getLogger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from torch.cuda import is_available as cuda_is_available
from prompt_templates import TINYLLAMA_DEFAULT

log = getLogger(__name__)

MODEL_NAMES = [
    "PY007/TinyLlama-1.1B-Chat-v0.3",
]

if cuda_is_available():
    MODEL_NAMES.extend([
        # "TheBloke/CollectiveCognition-v1.1-Mistral-7B-GPTQ",
        "TheBloke/Llama-2-7B-chat-GPTQ",
        "TheBloke/Llama-2-13B-chat-GPTQ"
    ])


class AppModel:
    def __init__(self, model_name, task="text-generation"):
        self._model_name = model_name
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map="auto",
        )

        self._pipe = pipeline(task=task, model=self._model, tokenizer=self._tokenizer)

    def run(
        self,
        inputs: dict,
        prompt_template: PromptTemplate = None,
        min_new_tokens=1,
        max_new_tokens=20,
        repetition_penalty=1.0,
        remove_tokens=["<s>", "</s>"],
    ):
        if prompt_template is None:
            prompt_template = TINYLLAMA_DEFAULT

        prompt = prompt_template.format(**inputs)

        if cuda_is_available():
            input_tensor = self._tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        else:
            input_tensor = self._tokenizer.encode(prompt, return_tensors="pt")

        output_tensor = self._model.generate(
            input_tensor,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

        generated_tensor = output_tensor[:, input_tensor.shape[1] :]

        ## The batch_decode call below removes the input tokens
        generated_text = self._tokenizer.batch_decode(generated_tensor)[0]

        for token in remove_tokens:
            generated_text = generated_text.replace(token, "")
        generated_text = generated_text.strip('" \n')

        output = {
            "text": generated_text,
            "output_token_length": len(generated_tensor[0]),
        }
        return output
