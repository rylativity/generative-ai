from langchain.prompts import PromptTemplate

DEFAULT = PromptTemplate(
    template="{input}",
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

{messages}[/INST]

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

EXTRACTION_PROMPT_TEMPLATE = PromptTemplate(
template="""Extract the specified properties for each unique entity in the passage that posesses those properties. Only extract the properties mentioned. DO NOT refer to previous passages or extracted entites. Format each entity as a JSON object where the property is the key.

Passage:
The brown cat found twelve berries and shared them with the blue mouse.

Properties:
animal,color

Extracted Entites (JSON):
[{{"animal":"cat","color":"brown"}},{{"animal":"mouse","color":"blue"}}]

Passage:
{passage}

Properties:
{properties}

Extracted Entities (JSON):""",
input_variables=["passage","properties"]
)