from opensearchpy import OpenSearch
from typing import Union

def get_client(hosts:Union[list[str],str], username:str, password:str, use_ssl=False, verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False):
    return OpenSearch(
        hosts=hosts,
        http_auth=(username, password),
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        ssl_assert_hostname=ssl_assert_hostname,
        ssl_show_warn=ssl_show_warn
        )

def create_index(client:OpenSearch, index:str, mappings={}, replace_existing=False, number_of_shards=1):
    
    if replace_existing:
        client.indices.delete(index, ignore_unavailable=True)
        print(f"Deleted existing index: {index}")

    index_body = {
        'settings': {
            'index': {
                'knn': True,
                "knn.algo_param.ef_search": 256,
                'number_of_shards':number_of_shards
            }
        },
        "mappings" : mappings
    }
    response = client.indices.create(index=index,body=index_body)
    return response

# def vector_field_search(query:str, field_name:str, k=3, size=3, index=index_name, opensearch_client=client, embedding_model=embeddings):
#     """k is the number of neighbors the search of each graph will return. You must also include the size option, which indicates how many results the query actually returns. 
#     The plugin returns k amount of results for each shard (and each segment) and size amount of results for the entire query. The plugin supports a maximum k value of 10,000.
#     """
#     query_body = {
#         "size": size,
#         "query": {
#             "knn": {
#                 field_name : {
#                     "vector": generate_embeddings(text=query, embedding_model=embedding_model),
#                     "k": k
#                 }
#             }
#         }
#     }