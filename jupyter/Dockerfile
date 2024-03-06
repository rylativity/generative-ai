FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
        python3-dev python3-pip \
        curl \
        build-essential \
        software-properties-common \
        # libmagic1 \
        # libgl1&& \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

ARG PROCESSOR=cpu
ENV PROCESSOR=${PROCESSOR}
COPY ./dependency-setup.sh .
RUN ./dependency-setup.sh

WORKDIR /workspace

CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser", "--allow-root"]
