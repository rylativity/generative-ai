from fastapi import FastAPI, Response, Request
from fastapi.responses import RedirectResponse
from fastapi.logger import logger

from datetime import datetime
import logging
import socket
# import jwt
# from jwt import PyJWKClient

### https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker/issues/19
# gunicorn_logger = logging.getLogger('gunicorn.error')
# logger.addHandler(gunicorn_logger.handlers)

stream_handler=logging.StreamHandler()
stream_formatter=logging.Formatter('{"time":"%(asctime)s", "name": "%(name)s", \
    "level": "%(levelname)s", "message": "%(message)s"}'
)
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)
# DEBUG > INFO > WARN > ERROR > CRITICAL > FATAL

app = FastAPI(debug=True)

@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.post("/log")
async def log(log_message:dict, level="INFO"):
    logger.log(level=logging.getLevelName(level), msg=log_message)

@app.get('/api/hello')
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

#  