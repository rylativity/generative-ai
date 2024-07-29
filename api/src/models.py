from pydantic import BaseModel
from llama_cpp import llama_model_params

class GenParams(BaseModel):

    max_tokens:int=10
    repeat_penalty:float=1.0
    top_p:float=1.0
    top_k:int=50
    typical_p:float=1.0
    temperature:float=0.5

class GenerationParameters(BaseModel):

    input: str
    # min_new_tokens:int=1
    max_new_tokens:int=10
    repetition_penalty:float=1.0
    do_sample:bool=True
    top_p:float=1.0
    top_k:int=50
    typical_p:float=1.0
    temperature:float=0.5
    # num_beams=1
    # num_return_sequences=1
    remove_tokens:list[str]=[]
    stop_sequences:list[str]=[]
    log_inferences:bool=True

class ChatMessage(BaseModel):
    role: str = "user"
    content: str
