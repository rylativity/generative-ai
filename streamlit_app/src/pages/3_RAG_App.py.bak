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


from prompt_templates import RAG_PROMPT_TEMPLATE
from components import model_settings
from utils.inference import generate, healthcheck

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


def search(query: str, query_filter: dict = None, n_results=5):
    query_embedding = embedding_function([query])
    results = collection.query(
        query_embeddings=query_embedding, where=query_filter, n_results=n_results
    )
    results = {k: v[0] for k, v in results.items() if v}
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


model_settings(
    default_generation_kwarg_overrides={
        "max_new_tokens": 1000,
        "repetition_penalty": 1.2,
    }
)

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

st.title("Retrieval Augmented Generation")
if not healthcheck():
    st.error("Cannot connect to inference endpoint...")
else:
    generation_parameters: dict = st.session_state.generation_parameters
    st.caption(f"Found {num_docs} split documents")


    # st.caption(f"Using document {st.session_state.selected_document}")

    st.divider()

    with st.form("qa_form"):
        return_sources = st.checkbox("Return Sources?")
        num_context_document_chunks = st.number_input(
            "Num RAG Document Chunks",
            min_value=0,
            max_value=None,
            value=5,
            help="Number of top-matching chunks from the available, chunked documents to add to the prompt context block",
        )
        question = st.text_area(
            "Enter your question...", value="What is langchain?", key="question"
        )

        submit = st.form_submit_button()

        if submit:
            with st.spinner("Searching knowledge base..."):
                res = search(query=question, n_results=num_context_document_chunks)
                # st.write(res)
                relevant_documents = res["documents"]
                st.caption(f"Retrieved {len(relevant_documents)} chunks")
                if not return_sources:
                    st.caption("(return sources to see chunk content and metadata)")
            context_string = "\n\n".join(relevant_documents)
            with st.spinner("Generating response..."):
                input = RAG_PROMPT_TEMPLATE.format(input=question, context=context_string)
                response = generate(
                    input=input,
                    generation_params=generation_parameters,
                )
            st.write(generation_parameters)
            st.header("Answer")
            st.write(response["text"])

            if return_sources:
                st.divider()
                st.header("Sources")

                for i, doc_id in enumerate(res["ids"]):
                    with st.expander(label=str(i + 1)):
                        for k in res:
                            st.header(k.rstrip("s").title())
                            st.write(res[k][i])
