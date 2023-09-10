# Generative AI Examples

## Setup

### Python

- `python3 -m venv venv`
- `source venv/bin/activate` (or `source venv/Scripts/activate` on Windows)
- `pip install -r requirements.txt`
- Copy or rename ".env.template" to ".env" and add your HuggingFace Hub API token
- Run notebooks (run `pip install jupyterlab` and then `jupyter lab` if you want to run the notebooks in JupyterLab from the virtual environment)

### Docker

- Copy or rename ".env.template" to ".env" and add your HuggingFace Hub API token
- `docker compose up -d`
- Navigate to http://localhost:8888 in your web browser and run notebooks.
