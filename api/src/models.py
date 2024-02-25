from pydantic import BaseModel

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
