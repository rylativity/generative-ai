FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=true

WORKDIR /app

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update && apt-get -y install \
        python3-dev python3-pip \
        curl \
        build-essential \
        software-properties-common \
        # libmagic1 \
        # libgl1&& \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

ARG PROCESSOR=cpu
ENV PROCESSOR=${PROCESSOR}
COPY ./dependency-setup.sh .
RUN ./dependency-setup.sh
     
COPY src/ /app/src/

CMD ["uvicorn","src.main:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "debug", "--reload"]

LABEL org.opencontainers.image.source=https://github.com/rylativity/generative-ai
