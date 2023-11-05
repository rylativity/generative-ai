from io import BytesIO
from glob import glob
import os
import json

import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from torch.cuda import is_available as cuda_is_available

from llm_utils import MODEL_NAMES, AppModel

SOURCE_DOCS_DIR = "./uploaded_files/source/"
PROCESSED_DOCS_DIR = "./uploaded_files/processed/"

os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DOCS_DIR, exist_ok=True)

error = st.empty()

def write_to_fs(file_content: BytesIO | str, filename: str, upload_dir=SOURCE_DOCS_DIR):
    
    if type(file_content) == BytesIO:
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

def index_chroma_docs(docs, collection_name:str):
    ...
def store_and_index_pdf():
    ...

def store_and_index_html(url:str):
    loader = UnstructuredURLLoader(urls=[url])
    
    try:
        html_doc = loader.load()[0]
    except IndexError as e:
        error.error("URL Likely Invalid")
        raise e
    write_to_fs(json.dumps(html_doc.dict()), f"{url.split('/')[-1]}.json")
    

with st.sidebar:

    available_sources = fetch_available_sources()
    selected_document = st.selectbox("Source Document", 
                                     options=available_sources,
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

    st.header("Add Content")
    
    # Handle File Uploads
    # uploaded_file = st.file_uploader("Upload a File *(PDF Only)*")
    # if uploaded_file is not None:
    #     filename = uploaded_file.name
    #     # To read file as bytes:
    #     bytes_data = BytesIO(uploaded_file.getvalue())

    #     write_to_fs(file_content=bytes_data, filename=filename)
        
        

    entered_url = st.text_input("Enter a URL", key="entered_url")
    if entered_url:
        store_and_index_html(entered_url)
        del st.session_state["entered_url"]
        st.rerun()

if not "model" in st.session_state or not "selected_document" in st.session_state:
    st.header("*Select a source document and load a model to get started*")
else:
    model = st.session_state.model

    st.title("Retrieval Augmented Generation")
    st.caption(f"Using model {model._model_name}")
    st.caption(f"Using document {st.session_state.selected_document}")
    st.divider()

    