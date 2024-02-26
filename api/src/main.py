from fastapi import FastAPI, Response, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.logger import logger

from datetime import datetime
from io import BytesIO
import logging
import socket

from src.model_config import MODEL_KWARGS
from src.llms import AppModel
from src.models import GenerationParameters

### https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker/issues/19
gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)
# DEBUG > INFO > WARN > ERROR > CRITICAL > FATAL

model = AppModel(**MODEL_KWARGS)

app = FastAPI(debug=True)

@app.get("/")
async def root():
    logger.warn("LOGGING WORKS!!")

    return RedirectResponse("/docs")

@app.get('/api/health')
def hello_world(request: Request):
    headers = str(request.headers)
    hostname = socket.gethostname()
    resp = f'''<h1>Hello, from the Flask App in Docker!</h1>\n
    This App is Running on Host: {hostname}
    #<p>{headers}</p>'''
    return Response(resp, status_code=200)

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
    