import os

MODEL_ENV_VARS = [
    {
    "key":"MODEL_NAME",
    "type": str
    },
    {
    "key":"MODEL_FILE",
    "type": str
    },
    {
    "key":"MODEL_TYPE",
    "type": str
    },
    {
    "key":"TOKENIZER_MODEL_NAME",
    "type": str
    },
    {
    "key":"N_GPU_LAYERS",
    "type": int
    },
    {
    "key":"CONTEXT_LENGTH",
    "type": int
    },
    {
    "key":"LLAMA_CPP_THREADS",
    "type": int
    },
]

OPTIONAL_VARS = ["MODEL_FILE","TOKENIZER_MODEL_NAME"]

MODEL_KWARGS = {}
for var_dict in MODEL_ENV_VARS:
    val = os.environ.get(var_dict["key"])
    if val:
        MODEL_KWARGS[var_dict["key"].lower()] = var_dict["type"](val)
    elif var_dict["key"] not in OPTIONAL_VARS:
        raise EnvironmentError(f"Missing required environment variable: {var_dict['key']}")