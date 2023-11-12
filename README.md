# Generative AI Examples

## Setup

**Important Note - Using GPUs in Containers**

If not using GPU, you MUST comment out the entire "deploy" section and everything contained within in the docker-compose.yml under the "jupyter" service.

For instructions on installing dependencies required to make GPU available inside container, see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.

**I highly recommend using docker, because it includes containers for auxiliary services that some of the notebooks and UI Apps rely on. If you use the Python setup instructions, you will need to configure your own auxiliary services (e.g. Chroma vector database) where relevant.**


### Option 1) Docker (Recommended)

- Copy or rename ".env.template" to ".env" and add your HuggingFace Hub API token
- Some notebooks require Milvus or OpenSearch. You can provide your own instances or use the ones provided in the docker-compose.yml.
    - To use Opensearch, uncomment the "opensearch" (and optionally "opensearch-dashboards") service in "docker-compose.yml"
    - To use Milvus, uncomment the "etcd", "minio", and "milvus" services in "docker-compose.yml"
- Run `docker compose up -d` in the root level of the project (the folder containing the "docker-compose.yml" file)
- Navigate to http://localhost:8888 in your web browser and run notebooks.

### Option 2) Python

- `python3 -m venv venv`
- `source venv/bin/activate` (or `source venv/Scripts/activate` on Windows)
- `pip install -r requirements.txt`
- Copy or rename ".env.template" to ".env" and add your HuggingFace Hub API token
- Run notebooks (run `pip install jupyterlab` and then `jupyter lab` if you want to run the notebooks in JupyterLab from the virtual environment)
- Note: In order to run the notebooks that require Milvus or OpenSearch, you will either need to deploy the ones provided as described above under the Docker setup instructions or provide your own instances.
