from glob import glob
import os
from shutil import rmtree
import json
from hashlib import sha1

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch.cuda import is_available as cuda_is_available

from llm_utils import AppModel
from prompt_templates import (
    RAG_PROMPT_TEMPLATE,
    CONDENSE_QUESTION_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE,
)
from components import model_settings

FILES_BASE_DIR = "./uploaded_files/"
SOURCE_DOCS_DIR = f"{FILES_BASE_DIR}source/"
PROCESSED_DOCS_DIR = f"{FILES_BASE_DIR}processed/"

os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DOCS_DIR, exist_ok=True)

CHROMADB_HOST = os.environ.get("CHROMADB_HOST") or "localhost"
CHROMADB_PORT = os.environ.get("CHROMADB_PORT") or "8000"
CHROMADB_COLLECTION = os.environ.get("CHROMADB_COLLECTION") or "default"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

embedding_model_name = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_name, device="cuda" if cuda_is_available() else "cpu"
)
chroma_httpclient = chromadb.HttpClient(
    host=CHROMADB_HOST,
    port=CHROMADB_PORT,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_httpclient.get_or_create_collection(
    CHROMADB_COLLECTION, embedding_function=embedding_function
)

num_docs = collection.count()

error = st.empty()


def delete_all_data():
    rmtree(FILES_BASE_DIR)
    try:
        chroma_httpclient.delete_collection(CHROMADB_COLLECTION)
    except ValueError:
        pass


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


def upsert_chroma_docs(documents: list[Document], overwrite_existing_source_docs=False):
    if overwrite_existing_source_docs:
        sources = set([doc.metadata["source"] for doc in documents])
        for source in sources:
            collection.delete(where={"source": source})

    texts = [doc.page_content for doc in documents]
    ids = [
        sha1(
            f"{doc.page_content}{doc.metadata['start_index']}".encode("utf8")
        ).hexdigest()
        for doc in documents
    ]
    embeddings = embedding_function(texts)
    metadatas = [doc.metadata for doc in documents]

    collection.upsert(
        ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
    )


def search(query: str, query_filter: dict = None, n_results=5, max_distance=0.8):
    query_embedding = embedding_function([query])
    results = collection.query(
        query_embeddings=query_embedding, where=query_filter, n_results=n_results
    )
    results = {k: v[0] for k, v in results.items() if v}

    irrelevant_indices = sorted(
        [i for i, dist in enumerate(results["distances"]) if dist >= max_distance],
        reverse=True,
    )
    for idx in irrelevant_indices:
        for k in results:
            del results[k][idx]
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

    upsert_chroma_docs(
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
    upsert_chroma_docs(
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
                        store_and_index_html(url=entered_url)

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
if not "model" in st.session_state:
    st.header("*Load a model to get started...*")
else:
    model: AppModel = st.session_state.model
    generation_parameters: dict = st.session_state.generation_parameters
    st.caption(f"Found {num_docs} split documents")

    st.caption(f"Using model {model._model_name}")

    # st.caption(f"Using document {st.session_state.selected_document}")

    st.divider()

    return_sources = st.checkbox("Return Sources?")
    max_context_document_chunks = st.number_input(
        "Max RAG Document Chunks",
        min_value=0,
        max_value=None,
        value=5,
        help="Max number of top-matching chunks from the available, chunked documents to add to the prompt context block",
    )
    max_similarity_distance = st.number_input(
        "Max Similarity Distance",
        min_value=0.01,
        max_value=None,
        value=0.8,
        help="Maximum L2 distance between query and matching document to consider adding the document to the context (limited by Max Context Document Chunks value). Lower scores indicate a better match.",
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
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            with st.spinner("..."):
                ## CONDENSE THE CHAT INPUT
                messages_history_string = "\n\n".join(
                    [
                        f"{message['role'].title()}: {message['content']}"
                        for message in st.session_state.messages
                    ]
                )
                response = model.run(
                    inputs={"chat_history": messages_history_string, "input": prompt},
                    prompt_template=CONDENSE_QUESTION_PROMPT_TEMPLATE,
                    stop_sequences=["User:"],
                    **generation_parameters,
                )
                condensed_input = response["text"]

                ## FETCH CONTEXT
                with st.spinner("Searching knowledge base..."):
                    search_res = search(
                        query=condensed_input,
                        n_results=(5 * max_context_document_chunks),
                        max_distance=max_similarity_distance,
                    )
                    search_res = {
                        k: v[:max_context_document_chunks]
                        for k, v in search_res.items()
                    }
                    # st.write(res)
                    relevant_documents = search_res["documents"]
                context_string = "\n\n".join(relevant_documents)

                stop_sequences = ["User:"]
                ## RAG PRROMPT CHAT MODEL
                if relevant_documents:
                    response = model.run(
                        inputs={"context": context_string, "input": condensed_input},
                        prompt_template=RAG_PROMPT_TEMPLATE,
                        stop_sequences=stop_sequences,
                        **generation_parameters,
                    )["text"]
                else:
                    response = model.run(
                        inputs={"messages": messages_history_string},
                        prompt_template=CHAT_PROMPT_TEMPLATE,
                        stop_sequences=stop_sequences,
                        **generation_parameters,
                    )["text"]
                message_placeholder.markdown(response)
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

            st.session_state.messages.append({"role": "assistant", "content": response})
    st.button("Clear chat history", on_click=clear_messages)
