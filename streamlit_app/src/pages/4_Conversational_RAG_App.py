from glob import glob
import os
from shutil import rmtree
import json
from hashlib import sha1

import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from huggingface_hub import hf_hub_download
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch.cuda import is_available as cuda_is_available
from opensearchpy import OpenSearch
from prompt_templates import (
    RAG_PROMPT_TEMPLATE,
    CONDENSE_QUESTION_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE,
)
from components import model_settings
from utils.inference import generate, generate_stream, healthcheck, create_completion, create_chat_completion
from utils.opensearch import get_client, count_docs_in_index, delete_index, bulk_upsert

FILES_BASE_DIR = "./uploaded_files/"
SOURCE_DOCS_DIR = f"{FILES_BASE_DIR}source/"
PROCESSED_DOCS_DIR = f"{FILES_BASE_DIR}processed/"

os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DOCS_DIR, exist_ok=True)

OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST") or "http://opensearch:9200"
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX") or "streamlit-docs"
OPENSEARCH_USERNAME = os.environ.get("OPENSEARCH_USERNAME") or "admin"
OPENSEARCH_PASSWORD = os.environ.get("OPENSEARCH_PASSWORD") or "admin"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

opensearch_client:OpenSearch = get_client(
    hosts=OPENSEARCH_HOST,
    username=OPENSEARCH_USERNAME,
    password=OPENSEARCH_PASSWORD
)

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

docsearch = OpenSearchVectorSearch(
    opensearch_url=OPENSEARCH_HOST,
    index_name=OPENSEARCH_INDEX,
    embedding_function=embeddings
)

def create_docsearch_index():
    return docsearch.create_index(index_name = OPENSEARCH_INDEX, dimension=embeddings.dict()["client"].get_sentence_embedding_dimension())

if not docsearch.index_exists():
    create_docsearch_index()
    
num_docs = count_docs_in_index(client=opensearch_client, index=OPENSEARCH_INDEX)

error = st.empty()

def delete_all_data():
    rmtree(FILES_BASE_DIR)
    docsearch.delete_index()
    create_docsearch_index()

    # delete_index(opensearch_client,OPENSEARCH_INDEX)

def write_to_fs(file_content: bytes | str, filename: str, upload_dir=SOURCE_DOCS_DIR):
    if type(file_content) == bytes:
        mode = "wb+"
    elif type(file_content) == str:
        mode = "w+"
    else:
        raise TypeError("file_content must be BytesIO or str")
    with open(f"{upload_dir}{filename}", mode) as f:
        f.write(file_content)


if not "sources" in st.session_state:
    st.session_state["sources"] = []


def fetch_available_sources():
    st.session_state["sources"] = glob(f"{SOURCE_DOCS_DIR}*")
    return st.session_state["sources"]

def upsert_opensearch_docs(documents:list[Document], overwrite_existing_source_docs=False):
    if overwrite_existing_source_docs:
        sources = set([doc.metadata["source"] for doc in documents])
        
        opensearch_client.delete_by_query(
            index=OPENSEARCH_INDEX,
            body={
                "query":{
                    "bool":{
                        "should":[
                            {
                                "match":{
                                    "source":source
                                }
                            } for source in sources
                        ]
                    }
                }
            }
        )
    
    docsearch.add_documents(
        documents=documents
    )


def search(query: str, query_filter: dict = None, n_results=5, min_score=0.0):
    results = docsearch.similarity_search_with_score(query, k=n_results)
    
    results = filter(lambda x: x[1] >= min_score, results)

    results = [doc[0].dict() for doc in results]
    
    return results


def store_and_index_uploaded_file(
    uploaded_file,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    overwrite_existing_source_docs=False,
):
    filename = uploaded_file.name
    file_content = uploaded_file.getvalue()

    write_to_fs(
        file_content=file_content, filename=filename, upload_dir=SOURCE_DOCS_DIR
    )

    pdf_path = f"{SOURCE_DOCS_DIR}{filename}"
    loader = UnstructuredPDFLoader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    docs = loader.load_and_split(text_splitter=splitter)

    upsert_opensearch_docs(
        docs, overwrite_existing_source_docs=overwrite_existing_source_docs
    )

    del st.session_state["uploaded_file"]

    st.rerun()


def store_and_index_html(
    url: str,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    overwrite_existing_source_docs=False,
):
    loader = UnstructuredURLLoader(urls=[url])

    try:
        html_doc = loader.load()[0]
    except IndexError as e:
        error.error("URL Likely Invalid")
        raise e

    write_to_fs(json.dumps(html_doc.dict()), f"{url.split('/')[-1]}.json")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    upsert_opensearch_docs(
        docs, overwrite_existing_source_docs=overwrite_existing_source_docs
    )

    del st.session_state["entered_url"]

    st.rerun()


if "generation_parameters" not in st.session_state:
    model_settings(
        default_generation_kwarg_overrides={
            "max_new_tokens": 1000,
            "repetition_penalty": 1.2,
        }
    )
else:
    model_settings()

with st.sidebar:
    st.divider()

    with st.expander(f"Manage Content (*{num_docs} docs available*)"):
        content_upload_options = ["url", "file"]

        upload_option = st.radio(
            "Type", options=content_upload_options, horizontal=True
        )

        with st.form("upload_form"):
            text_chunk_size = st.number_input("Chunk Size", value=DEFAULT_CHUNK_SIZE)
            text_chunk_overlap = st.number_input(
                "Chunk Overlap", value=DEFAULT_CHUNK_OVERLAP
            )
            overwrite_existing_source_docs = st.checkbox(
                "Overwrite Existing Documents Matching Source?", value=True
            )

            if upload_option == "url":
                entered_url = st.text_input(
                    "Enter a URL",
                    value="https://python.langchain.com/docs/get_started/introduction",
                    key="entered_url",
                )
                url_submitted = st.form_submit_button()

                if url_submitted:
                    with st.spinner():
                        store_and_index_html(url=entered_url, chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap, overwrite_existing_source_docs=overwrite_existing_source_docs)

            elif upload_option == "file":
                uploaded_file = st.file_uploader(
                    "Upload a File *(PDF Only)*", key="uploaded_file"
                )
                file_submitted = st.form_submit_button()
                if file_submitted:
                    with st.spinner():
                        store_and_index_uploaded_file(
                            uploaded_file=uploaded_file,
                            chunk_size=text_chunk_size,
                            chunk_overlap=text_chunk_overlap,
                        )

        st.button(
            "Delete All Content", on_click=delete_all_data, use_container_width=True
        )

st.title("Conversational Retrieval Augmented Generation")
if not healthcheck():
    st.error("Cannot connect to inference endpoint...")
else:
    generation_parameters: dict = st.session_state.generation_parameters
    st.caption(f"Found {num_docs} split documents")

    # st.caption(f"Using document {st.session_state.selected_document}")

    st.divider()

    return_sources = st.checkbox("Return Sources?")
    return_intermediate_question = st.checkbox(
        "Return Intermediate (Condensed) Questions"
    )
    max_context_document_chunks = st.number_input(
        "Max RAG Document Chunks",
        min_value=0,
        max_value=None,
        value=5,
        help="Max number of top-matching chunks from the available, chunked documents to add to the prompt context block",
    )
    min_score = st.number_input(
        "Min Relevancy Score",
        min_value=0.00,
        max_value=None,
        value=1.0,
        help="Minimum relevancy score between query and matching document to consider adding the document to the context (limited by Max Context Document Chunks value). Higher scores indicate a better match.",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def clear_messages():
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            condensed_input_placeholder = st.empty()
            message_placeholder = st.empty()
            sources_placeholder = st.empty()

            with st.spinner("..."):
                if not len(st.session_state.messages) > 1:
                    condensed_input = prompt
                    messages_history_string = condensed_input
                else:
                    ## CONDENSE THE CHAT INPUT
                    messages_history_string = "\n\n".join(
                        [
                            f"{message['role'].title()}: {message['content']}"
                            for message in st.session_state.messages
                        ]
                    )
                    input = CONDENSE_QUESTION_PROMPT_TEMPLATE.format(chat_history=messages_history_string, input=prompt)
                    # generation_parameters["stop_sequences"] = stop_sequences
                    response = create_completion(
                        input=input,
                        generation_params=generation_parameters,
                    )
                    response_text = response["choices"][0]["text"]
                    condensed_input = response_text
                    if return_intermediate_question:
                        condensed_input_placeholder.caption(
                            f"Question rephrased to: {condensed_input}"
                        )

                ## FETCH CONTEXT
                with st.spinner("Searching knowledge base..."):
                    search_res = search(
                        query=condensed_input,
                        n_results=(5 * max_context_document_chunks),
                        min_score=min_score,
                    )
                    
                    search_res = search_res[:max_context_document_chunks]
                    # st.write(res)
                    relevant_documents = [f'{res["metadata"]["source"]} - {res["page_content"]}' for res in search_res]
                context_string = "\n\n".join(relevant_documents)

                stop_sequences = ["User:"]
                ## RAG PRROMPT CHAT MODEL
                if relevant_documents:
                    user_message = f"Context:\n{context_string}\n\n{prompt}"
                    messages_list = st.session_state.messages[:-1] + [{"role":"user","content":user_message}] # Replace the last message with the user message and context

                    # input = RAG_PROMPT_TEMPLATE.format(context=context_string, input=condensed_input)
                    
                else:
                    messages_list = st.session_state.messages
                    # input = CHAT_PROMPT_TEMPLATE.format(messages=messages_history_string)
                
                generation_parameters["stop_sequences"] = stop_sequences
                
                response = create_chat_completion(
                    messages=messages_list,
                    generation_params=None,
                )
                
                # st.write(response)
                response_text = response["choices"][0]["message"]["content"]
                # response = generate_stream(
                #         input=input,
                #         generation_params=generation_parameters,
                #     )
                
                # response_msg = message_placeholder.write_stream(response)
                response_msg = message_placeholder.write(response_text)

                if return_sources and len(relevant_documents) > 0:
                    try:
                        st.divider()
                        st.header("Sources")

                        for i, doc_id in enumerate(search_res["ids"]):
                            with st.expander(label=str(i + 1)):
                                for k in search_res:
                                    st.header(k.rstrip("s").title())
                                    st.write(search_res[k][i])
                    except NameError:
                        pass

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            # st.write(messages_list)
    st.button("Clear chat history", on_click=clear_messages)
