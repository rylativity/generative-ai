from langchain.prompts import PromptTemplate

TINYLLAMA_DEFAULT = PromptTemplate(
    template="<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    input_variables=["input"],
)

LLAMA2_DEFAULT = PromptTemplate(
    template="""<s>[INST]
<<SYS>>
You are a helpful assistant. Your answers are always based on the included context. Answer honestly and if you do not have enough information to answer, say so. 
<</SYS>>

Context: {context}

Input: {input}[/INST]

Assistant:""",
    input_variables=["input", "context"],
)

CHAT_PROMPT_TEMPLATE = PromptTemplate(
    template="""<s>[INST]
<<SYS>>
You are a conversational assistant. Answer honestly and if you do not have enough information to answer, say so. 
<</SYS>>

{messages}

Assistant:""",
    input_variables=["messages"],
)

SUMMARIZE_PROMPT_TEMPLATE = PromptTemplate(
    template="""<s>[INST]
<<SYS>>
You are a detail-oriented assistant. The answers you provide are always grounded in the source text. Be as accurate as possible, and do not make up information.
<</SYS>>

Summarize the following text.

Text:{text}[/INST]

Summary:""",
    input_variables=["text"],
)

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""Respond accurately to the user based only on the following context. If the context does not contain the answer, you can ignore it. Always follow up with the user at the end of your response. Be informative, but avoid being overly verbose:
{context}

User: {input}[/INST]

Assistant:""",
    input_variables=["input", "context"],
)

CONDENSE_QUESTION_PROMPT_TEMPLATE = PromptTemplate(
    template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:
""",
    input_variables=["chat_history", "input"],
)
