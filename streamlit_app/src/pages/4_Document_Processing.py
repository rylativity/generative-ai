import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from torch.cuda import is_available as cuda_is_available
from llm_utils import MODEL_NAMES, AppModel
from prompt_templates import SUMMARIZE_PROMPT_TEMPLATE
from uuid import uuid4

if not "document" in st.session_state:
    st.session_state["document"] = None

with st.sidebar:

    with st.form("Model Settings"):
        st.header("Model Settings")
        if cuda_is_available():
            st.success(f"CUDA Available")
        else:
            st.error(f"CUDA Unavailable")
        model_name = st.selectbox(
            "Model", options=MODEL_NAMES, placeholder="Select a model...", index=None
        )

        load_model = st.form_submit_button("Load Model")

        if load_model:
            with st.spinner("Loading model"):
                st.session_state.model = AppModel(model_name=model_name)
            st.write(f"Model {model_name} loaded successfully")
    
    st.divider()

    st.header(f"Add document")

    content_upload_options = ["url", "file"]

    upload_option = st.radio("Type", options=content_upload_options, horizontal=True)

    with st.form("upload_form"):
        
        if upload_option == "url":
            entered_url = st.text_input("Enter a URL", value="https://python.langchain.com/docs/get_started/introduction", key="entered_url")
            url_submitted =  st.form_submit_button()
            
            if url_submitted:
                with st.spinner():
                    loader = UnstructuredURLLoader(urls=[entered_url]) 
                    doc = loader.load()[0]
                    st.session_state.document = doc
        
        elif upload_option == "file":
            uploaded_file = st.file_uploader("Upload a File *(PDF Only)*", key="uploaded_file")
            file_submitted = st.form_submit_button()
            if file_submitted:
                with st.spinner():
                    tmp_path = f"/tmp/upload_file_{uuid4().hex}"
                    with open(tmp_path, "wb+") as f:
                        f.write(uploaded_file.getvalue())
                    loader = UnstructuredPDFLoader(tmp_path)
                    doc = loader.load()[0]
                    doc.metadata["source"] = uploaded_file.name
                    st.session_state.document = doc

st.title("Document Processing")
if not st.session_state["document"]:
    ...
else:
    st.divider()
    st.header(f"Document Source: {st.session_state['document'].metadata['source']}")
    with st.expander("Document Content", expanded=False):
        st.write(st.session_state["document"].page_content)
    
    processing_method_options = ["Summarization", "Extraction", "Q&A"]
    processing_option = st.radio("Processing Method", options=processing_method_options, index=None, horizontal=True)
    
    if not "model" in st.session_state:
        st.write("Select a model to proceed")
    else:
        model = st.session_state.model
    if processing_option == "Summarization":
        with st.form("Document Processing"):
            submitted = st.form_submit_button("Summarize", use_container_width=True)
        if submitted:
            prompt = SUMMARIZE_PROMPT_TEMPLATE
            inputs = {"text":st.session_state["document"].page_content}
            llm = HuggingFacePipeline(pipeline=model._pipeline)
            token_length = len(model._tokenizer.encode(st.session_state.document.page_content))
            if token_length > 1500:
                chain_type = "map_reduce"
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
                docs = splitter.split_documents([st.session_state.document])
            else:
                chain_type = "stuff"
                docs = [st.session_state.document]
            chain = load_summarize_chain(llm, chain_type=chain_type)
            
            summary = chain.run(docs)
            st.write(summary)

    elif processing_option == "Extraction":
        ...
    elif processing_option == "Q&A":
        ...

    