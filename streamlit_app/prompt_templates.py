from langchain.prompts import PromptTemplate

TINYLLAMA_DEFAULT = PromptTemplate(
    template="<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    input_variables=["input"],
)

LLAMA2_DEFAULT =  PromptTemplate(
template="""<s>[INST]
<<SYS>>
You are a helpful assistant. Your answers are always based on the included context. Answer honestly and if you do not have enough information to answer, say so. 
<</SYS>>

Context: {context}

Input: {input}[/INST]

Assistant:""", 
input_variables=["input", "context"]
)

CHAT_PROMPT_TEMPLATE = PromptTemplate(
    template="""<s>[INST]
<<SYS>>
You are a conversational assistant. Answer honestly and if you do not have enough information to answer, say so. 
<</SYS>>

{messages}

Assistant:""",
input_variables=["messages"]
)
