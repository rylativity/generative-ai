import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from models import AppModel
from prompt_templates import EXTRACTION_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE
from uuid import uuid4
from components import model_settings

from utils.inference import generate, healthcheck

model_settings(include_gen_params=False)
st.caption("Generation parameters disabled for this app")

if not "document" in st.session_state:
    st.session_state["document"] = None

with st.sidebar:
    st.divider()

    st.header(f"Add document")

    content_upload_options = ["url", "file"]

    upload_option = st.radio("Type", options=content_upload_options, horizontal=True)

    with st.form("upload_form"):
        if upload_option == "url":
            entered_url = st.text_input(
                "Enter a URL",
                value="https://python.langchain.com/docs/get_started/introduction",
                key="entered_url",
            )
            url_submitted = st.form_submit_button()

            if url_submitted:
                with st.spinner():
                    loader = UnstructuredURLLoader(urls=[entered_url])
                    doc = loader.load()[0]
                    st.session_state.document = doc

        elif upload_option == "file":
            uploaded_file = st.file_uploader(
                "Upload a File *(PDF Only)*", key="uploaded_file", type=["pdf"]
            )
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
if not healthcheck():
    st.error("Cannot connect to inference endpoint...")
elif not st.session_state["document"]:
    st.write("Add a document to get started...")
else:
    st.divider()
    st.header(f"Document Source: {st.session_state['document'].metadata['source']}")
    with st.expander("Document Content", expanded=False):
        st.write(st.session_state["document"].page_content)

    processing_method_options = ["Summarization", "Entity Extraction", "Fact Extraction"]
    processing_option = st.radio(
        "Processing Method",
        options=processing_method_options,
        index=None,
        horizontal=True,
    )

    text = {"text": st.session_state["document"].page_content}
            
    if processing_option == "Summarization":
        with st.form("Document Processing"):
            submitted = st.form_submit_button("Summarize", use_container_width=True)
        if submitted:
            with st.spinner("Summarizing..."):

                input = SUMMARIZE_PROMPT_TEMPLATE.format(text = st.session_state.document.page_content)
                summary = generate(
                    input=input,
                    generation_params={"max_new_tokens":500}
                )["text"]

                st.write(summary)

    elif processing_option == "Entity Extraction":
        with st.form("Document Processing"):
            properties = st.text_input("Properties to extract", placeholder="Comma separated list of properties (e.g. name,age,height)")
            submitted = st.form_submit_button("Extract", use_container_width=True)
            
            if submitted:
                if not properties:
                    st.error("Must specify properties to extract")
                else:
                    with st.spinner("Extracting..."):
                        properties = properties.split(",")
                        input = EXTRACTION_PROMPT_TEMPLATE.format(passage=st.session_state.document.page_content, properties=properties)
                        # chain = create_extraction_chain(schema=schema, llm=llm, verbose=True)
                        extraction_result = generate(input=input, generation_params={"max_new_tokens":500})
                        st.write(extraction_result)
        
    elif processing_option == "Fact Extraction":
        st.write("TODO...")
