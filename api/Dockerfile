FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=true

WORKDIR /app

SHELL ["/bin/bash", "-c"]

RUN useradd --create-home --shell /bin/bash fastapi && chown -R fastapi /app && \
    apt-get -y update && apt-get -y install \
        python3-dev python3-pip \
        curl \
        build-essential \
        software-properties-common \
        # libmagic1 \
        # libgl1&& \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && pip cache purge

ARG PROCESSOR=gpu
ENV PROCESSOR=${PROCESSOR}
COPY ./dependency-setup.sh .
RUN ./dependency-setup.sh
     
COPY --chown=fastapi:root src/ /app/src/

USER fastapi

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-12.3/compat"

CMD ["uvicorn","src.main:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "debug", "--reload"]
