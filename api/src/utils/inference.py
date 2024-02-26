import json
import os
import requests

INFERENCE_API_ENDPOINT_URL = "http://inference-api:5000/api/generate"
HEADERS = {
        "Content-Type":"application/json",
        "accept":"application/json"
    }

DEFAULT_GENERATION_PARAMS = {
        "max_new_tokens": 10,
        "repetition_penalty": 1,
        "do_sample": True,
        "top_p": 1,
        "top_k": 50,
        "typical_p": 1,
        "temperature": 0.5,
        "remove_tokens": [],
        "stop_sequences": [],
        "log_inferences": True
    }

def generate_stream(input:str, endpoint=f"{INFERENCE_API_ENDPOINT_URL}?stream=true"):
    
    data = dict(DEFAULT_GENERATION_PARAMS)
    data["input"] = input

    with requests.post(url=endpoint, headers=HEADERS, data=json.dumps(data), stream=True) as r:
            for chunk in r.iter_lines(decode_unicode=True):
                yield chunk


def generate(input:str, endpoint=f"{INFERENCE_API_ENDPOINT_URL}?stream=false"):

    data = dict(DEFAULT_GENERATION_PARAMS)
    data["input"] = input

    res = requests.post(url=endpoint, headers=HEADERS, data=json.dumps(data), stream=False)
    return res.json()["text"]
