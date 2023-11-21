FROM python:3.10

RUN apt-get update && apt-get install graphviz-dev && \
    pip install jupyterlab

COPY ./requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /workspace

CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser", "--allow-root"]
