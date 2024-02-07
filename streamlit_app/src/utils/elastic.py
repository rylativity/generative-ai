from elasticsearch import Elasticsearch
from logging import getLogger

ELASTICSEARCH_HOSTS = "http://elasticsearch:9200"


log = getLogger(__name__)
log.setLevel("INFO")

def create_index(index_name:str, mappings:dict=None):
    client = Elasticsearch(
         hosts=ELASTICSEARCH_HOSTS.split(","),
    )

    if not client.indices.exists(index=index_name):

        client.indices.create(
            index=index_name,
            mappings=mappings
        )
    else:
        log.warn(f"Index '{index_name}' already exists. Skipping index creation")

def index_document(index_name:str, doc:dict, id=None, create_missing_index=True):
    client = Elasticsearch(
         hosts=ELASTICSEARCH_HOSTS.split(","),
    )

    if create_missing_index and not client.indices.exists(index=index_name):
        create_index(index_name=index_name)

    client.index(
        index=index_name,
        document=doc,
        id=None
    )
