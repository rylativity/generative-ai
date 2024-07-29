FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV PYTHONDONTWRITEBYTECODE=true \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \ 
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get -y update && apt-get -y install \
        python3-dev python3-pip \
        curl \
        build-essential \
        software-properties-common \
        # libmagic1 \
        # libgl1&& \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 -



WORKDIR /app
COPY poetry.lock pyproject.toml ./

# COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    # pip install -r requirements.txt
    poetry install

ARG PROCESSOR=cpu
ENV PROCESSOR=${PROCESSOR}
COPY ./dependency-setup.sh .
RUN ./dependency-setup.sh

COPY src/ /app/src/

CMD ["poetry","run", "python", "src/run.py"]

LABEL org.opencontainers.image.source=https://github.com/rylativity/generative-ai
