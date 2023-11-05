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

from llm_utils import MODEL_NAMES, AppModel

FILES_BASE_DIR = "./uploaded_files/"
SOURCE_DOCS_DIR = f"{FILES_BASE_DIR}source/"
PROCESSED_DOCS_DIR = f"{FILES_BASE_DIR}processed/"

os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DOCS_DIR, exist_ok=True)

CHROMADB_HOST = os.environ.get("CHROMADB_HOST") or "localhost"
CHROMADB_PORT = os.environ.get("CHROMADB_PORT") or "8000"
CHROMADB_COLLECTION = os.environ.get("CHROMADB_COLLECTION") or "default"
embedding_function = SentenceTransformerEmbeddingFunction(device="cuda" if cuda_is_available() else "cpu")
chroma = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT, settings=Settings(anonymized_telemetry=False))
collection = chroma.get_or_create_collection(CHROMADB_COLLECTION, embedding_function=embedding_function)


error = st.empty()

def delete_all_data():
    rmtree(FILES_BASE_DIR)
    try:
        chroma.delete_collection(CHROMADB_COLLECTION)
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
            collection.delete(where={"source":source})
    
    texts = [doc.page_content for doc in documents]
    ids = [sha1(text.encode("utf8")).hexdigest() for text in texts]
    embeddings = embedding_function(texts)
    metadatas = [doc.metadata for doc in documents]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    
def store_and_index_uploaded_file(uploaded_file, chunk_size=1000, chunk_overlap=200, overwrite_existing_source_docs=False):
    
    filename = uploaded_file.name
    file_content = uploaded_file.getvalue()
        
    write_to_fs(file_content=file_content, filename=filename, upload_dir=SOURCE_DOCS_DIR)

    pdf_path = f"{SOURCE_DOCS_DIR}{filename}"
    loader = UnstructuredPDFLoader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    docs = loader.load_and_split(text_splitter=splitter)
    
    upsert_chroma_docs(docs, overwrite_existing_source_docs=overwrite_existing_source_docs)
    
    del st.session_state["uploaded_file"]
    
    st.rerun()

def store_and_index_html(url:str, chunk_size=1000, chunk_overlap=200, overwrite_existing_source_docs=False):
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
        add_start_index=True
    )
    docs = loader.load_and_split(text_splitter=splitter)
    upsert_chroma_docs(docs, overwrite_existing_source_docs=overwrite_existing_source_docs)

    del st.session_state["entered_url"]
    
    st.rerun()
    

with st.sidebar:

    available_sources = fetch_available_sources()
    selected_document = st.selectbox("Source Document", 
                                     options=available_sources,
                                     format_func=lambda x: x.split("source/")[-1],
                                     key="selected_document")    
    
    with st.form("Model Settings"):
        st.header("Model Settings")
        st.write(f"CUDA Available: {cuda_is_available()}")
        model_name = st.selectbox(
            "Model", options=MODEL_NAMES, placeholder="Select a model...", index=None
        )

        load_model = st.form_submit_button("Load Model")

        if load_model:
            with st.spinner("Loading model"):
                st.session_state.model = AppModel(model_name=model_name)
            st.write(f"Model {model_name} loaded successfully")
    
    st.divider()

    with st.expander("Manage Content"):

        content_upload_options = ["url", "file"]

        upload_option = st.radio("Type", options=content_upload_options, horizontal=True)

        with st.form("upload_form"):
            
            text_chunk_size = st.number_input("Chunk Size", value=1000)
            text_chunk_overlap = st.number_input("Chunk Overlap", value=200)
            overwrite_existing_source_docs = st.checkbox("Overwrite Existing Documents Matching Source?", value=True)
            
            if upload_option == "url":
                    entered_url = st.text_input("Enter a URL", key="entered_url")
                    url_submitted =  st.form_submit_button()
                    
                    if url_submitted:
                        with st.spinner(): 
                            store_and_index_html(url=entered_url)
            
            elif upload_option == "file":
                    uploaded_file = st.file_uploader("Upload a File *(PDF Only)*", key="uploaded_file")
                    file_submitted = st.form_submit_button()
                    if file_submitted:
                        with st.spinner():
                            store_and_index_uploaded_file(uploaded_file=uploaded_file, chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap)
        
        st.button("Delete All Content", on_click=delete_all_data, use_container_width=True)

st.title("Retrieval Augmented Generation")
st.write(f"Found {collection.count()} split documents")
st.divider()
if not "model" in st.session_state or not "selected_document" in st.session_state:
    st.header("*Select a source document and load a model to get started...*")
else:
    model = st.session_state.model

    st.caption(f"Using model {model._model_name}")
    st.caption(f"Using document {st.session_state.selected_document}")
    st.divider()

    