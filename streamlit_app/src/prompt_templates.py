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


Summarize the following text.

Text:{text}[/INST]

Summary:""",
    input_variables=["text"],
)

RAG_PROMPT_TEMPLATE = PromptTemplate(
    template="""<s>[INST]
<<SYS>>
You are a helpful and honest assistant who provides accurate responses grounded in context. Always follow up with the user to see how else you can help them.
<</SYS>>

Respond accurately to the user based only on the following context. If the context does not contain the information needed to answer the question, say so.

Context: {context}

User: {input}[/INST]

Assistant:""",
    input_variables=["input", "context"],
)

CONDENSE_QUESTION_PROMPT_TEMPLATE = PromptTemplate(
    template="""<s>[INST]
<<SYS>>
You are a helpful assistant.
<</SYS>>

Given the following chat history and a new input from the user, rephrase the input to be a standalone question.

Chat History:
{chat_history}


New Input: {input}[/INST]

Standalone Question:""",
    input_variables=["chat_history", "input"],
)
