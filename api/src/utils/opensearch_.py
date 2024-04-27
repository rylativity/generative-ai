from opensearchpy import OpenSearch
from logging import getLogger

OPENSEARCH_HOSTS = "http://opensearch:9200"


log = getLogger(__name__)
log.setLevel("INFO")

def create_index(index_name:str, mappings:dict=None):
    client = OpenSearch(
         hosts=OPENSEARCH_HOSTS.split(","),
    )

    if not client.indices.exists(index=index_name):

        client.indices.create(
            index=index_name,
            body=mappings
        )
    else:
        log.warn(f"Index '{index_name}' already exists. Skipping index creation")

def index_document(index_name:str, doc:dict, id=None, create_missing_index=True):
    client = OpenSearch(
         hosts=OPENSEARCH_HOSTS.split(","),
    )

    if create_missing_index and not client.indices.exists(index=index_name):
        create_index(index_name=index_name)

    client.index(
        index=index_name,
        body=doc,
        id=id
    )
