FROM python:3.10-bookworm

SHELL ["/bin/bash", "-c"]

WORKDIR /app



RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    curl \
    libmagic1 \
    libgl1

ARG PROCESSOR=cpu
# ARG PROCESSOR=gpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt 

### dependency-setup.sh looks at the PROCESSOR arg. If it is set to gpu, it installs CUDA Toolkit and llama-cpp-python with CUDA Support
### set PROCESSOR=gpu to build with CUDA support
COPY dependency-setup.sh .
COPY requirements-cuda.txt .
RUN --mount=type=cache,target=/root/.cache/pip ./dependency-setup.sh

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set PROCESSOR env var to use in application
ENV PROCESSOR=${PROCESSOR}

ENTRYPOINT ["streamlit", "run", "src/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
