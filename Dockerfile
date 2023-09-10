FROM python:3.10

RUN pip install jupyterlab

COPY ./requirements2.txt .

RUN pip install -r requirements2.txt

WORKDIR /workspace

CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser", "--allow-root"]
