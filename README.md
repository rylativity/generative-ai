# Generative AI Examples

This repo contains examples showing how to use containerized, locally hosted, free and open large-language models (e.g. llama2, mixtral, etc...) for various use-cases with either CPU or GPU (or both). This project includes an easy-to-use Streamlit user-interface for LLM inference for various use-cases (e.g. chatbot, RAG, text processing, and more...) and a jupyterlab environment (along with some example notebooks) for interacting with LLMs programatically in python (along with some auxiliary services; e.g. vector-store). 

Follow the setup steps below to get started.

**YOU MUST HAVE DOCKER & DOCKER-COMPOSE INSTALLED TO RUN THIS PROJECT. If you do not have Docker installed, follow the installation instructions for your operating system here - https://docs.docker.com/engine/install/**

**FOR GPU USERS ONLY: In order to share your host machine's CUDA-capable GPU with containers, you must install the Nvidia container toolkit. For installation instructions, see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.**

## Setup Steps

- Copy or rename ".env.template" to ".env" and add your HuggingFace Hub API token
- The "docker-compose.yml" (or "docker-compose-gpu.yml") contains a set of model configuration environment variables under the "inference-api" service. Update those variables to use a different model or to change model settings. **YOU MUST CHOOSE A MODEL THAT CAN FIT ON YOUR CPU/GPU OR THE inference-api SERVICE WILL FAIL TO START (check logs with `docker compose logs -f inference-api` to identify any errors loading model)**
- Run `docker compose up -d` (or `docker compose -f docker-compose-gpu.yml up -d` if using an Nvidia GPU) in the root level of the project (the folder containing the "docker-compose.yml" file)
- The inference-api service will take a while to start up the first time you use a new model, because it the model must first be downloaded to your computer. You can run `docker compose logs -f inference-api` to track the logs and see when the model is ready.
    The logs will appear to hang at the lines below while downloading the model:
    ```
    inference-api-1  | INFO:     Will watch for changes in these directories: ['/app']
    inference-api-1  | INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)
    inference-api-1  | INFO:     Started reloader process [1] using StatReload
    ```

    You should see log lines like the below once the model has loaded successfully:
    ```
    inference-api-1  | INFO:     Started server process [28]
    inference-api-1  | INFO:     Waiting for application startup.
    inference-api-1  | INFO:     Application startup complete.
    inference-api-1  | INFO:     127.0.0.1:52106 - "GET /api/health HTTP/1.1" 200 OK
    ```
- Navigate to http://localhost:8501 in your web browser to play with models and generative AI use-cases
- Navigate to http://localhost:8888 in your web browser to run the python example notebooks.

## To-Do / Roadmap
- [] Update README
- [x] Update available generation parameters in streamlit app
- [x] Move model inference functionality to FastAPI
- [] Implement SQLAlchemy (or other) in streamlit app to save prompt configurations and store documents
- [] Update layout of streamlit app to improve user experience/flow
- [] Add multi-prompting to Basic Prompting in streamlit app
- [] Add natural-language to SQL use-case to streamlit app
- [] Add LLM Agent example (either as separate use-case or as part of chatbot) to streamlit app
- [] Add additional model metadata and model-specific prompts, and automatically update default prompts and kwargs on a per-model basis 
- [] Address bug where llama-cpp-python gguf models don't release GPU VRAM when using n_gpu_layers
