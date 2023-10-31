from langchain.prompts import PromptTemplate

TINYLLAMA_DEFAULT = PromptTemplate(
    template="<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    input_variables=["input"],
)
