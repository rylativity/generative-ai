# Generative AI Examples

This repo contains examples showing how to use containerized, locally hosted, free and open large-language models (e.g. llama2, mixtral, etc...) for various use-cases with either CPU or GPU (or both). This project includes an easy-to-use Streamlit user-interface for LLM inference for various use-cases (e.g. chatbot, RAG, text processing, and more...) and a jupyterlab environment (along with some example notebooks) for interacting with LLMs programatically in python (along with some auxiliary services; e.g. vector-store). 

Follow the setup steps below to get started.

**YOU MUST HAVE DOCKER & DOCKER-COMPOSE INSTALLED TO RUN THIS PROJECT. If you do not have Docker installed, follow the installation instructions for your operating system here - https://docs.docker.com/engine/install/**

**If not using GPU, you MUST comment out the entire "deploy" section and everything contained within in the docker-compose.yml under the "jupyter" and "streamlit" services.**

**For instructions on installing dependencies required to make GPU available inside Docker containers, see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.**

## Setup Steps

- Copy or rename ".env.template" to ".env" and add your HuggingFace Hub API token
- Run `docker compose up -d` in the root level of the project (the folder containing the "docker-compose.yml" file)
- Navigate to http://localhost:8501 in your web browser to download and play with models and generative AI use-cases
- Navigate to http://localhost:8888 in your web browser and run python example notebooks.

## To-Do / Roadmap
- [] Update README
- [] Move model inference functionality to FastAPI
- [] Implement SQLAlchemy (or other) in streamlit app to save prompt configurations and store documents
- [] Update layout of streamlit app
- [] Add natural-language to SQL use-case to streamlit app
- [] Add LLM Agent example (either as separate use-case or as part of chatbot) to streamlit app
