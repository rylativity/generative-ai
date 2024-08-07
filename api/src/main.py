from fastapi import FastAPI, Response, Request, Body
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from fastapi.logger import logger
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

import logging

from model_config import MODEL_KWARGS
from llms import AppModel
from models import GenerationParameters, GenParams, ChatMessage

### https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker/issues/19
gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)
# DEBUG > INFO > WARN > ERROR > CRITICAL > FATAL

model = AppModel(**MODEL_KWARGS)

app = FastAPI(
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
    ],
    debug=True
    )

@app.get("/")
async def root():
    logger.warn("LOGGING WORKS!!")

    return RedirectResponse("/docs")

@app.get('/api/health')
def hello_world(request: Request):
    # headers = str(request.headers)
    # hostname = socket.gethostname()
    # resp = f'''<h1>Hello, from the Flask App in Docker!</h1>\n
    # This App is Running on Host: {hostname}
    # #<p>{headers}</p>'''
    return JSONResponse({"status":"success"}, status_code=200)

@app.get('/api/headers')
def return_headers(request:Request):
    headers = dict(request.headers)
    return headers

@app.post("/api/generate")
async def generate(
        params:GenerationParameters,
    ):

    return model.run(
            **dict(params)
        )

@app.post("/api/generate_stream")
async def generate_stream(
        params:GenerationParameters,
    ):

    return StreamingResponse(
        model.stream(
            **dict(params)
        ),
        media_type="application/x-ndjson"
    )

@app.post("/api/generate_completion")
async def generate_completion(
        input:str = Body(...),
        params:GenParams = GenParams().model_dump(),
    ):
    return model.generate_completion(prompt=input,**dict(params))

@app.post("/api/chat/generate")
async def generate_chat(
        messages:list[ChatMessage],
        params:GenParams,
    ):
    return model.generate_chat_completion(messages=messages,**dict(params))

@app.post("/api/chat/generate_stream")
async def generate_chat_stream(
        messages:list[ChatMessage],
        params:GenParams,
    ):
    params["stream"] = True
    return StreamingResponse(
        model.generate_chat_completion(
            messages=messages,
            **dict(params)
        )
    )