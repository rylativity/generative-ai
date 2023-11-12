import streamlit as st
from diffusers import DiffusionPipeline
from torch.cuda import is_available as cuda_is_available
from torch import float16

if "diffuser_model" not in st.session_state:
    st.session_state.diffuser_model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=float16,
        use_safetensors=True,
        variant="fp16",
    )
    if cuda_is_available:
        st.session_state.diffuser_model.to("cuda")

with st.form("image_generation_form"):
    query = st.text_input(
        "Image instruction",
        value="A glowing, high-tech alien spaceship leaving a barren earth having taken all of its resources",
    )
    num_images = st.number_input("Number of Images", value=1, min_value=1, max_value=10)
    submitted = st.form_submit_button()
    if submitted:
        if query:
            with st.spinner("Generating..."):
                st.session_state.images = [
                    st.session_state.diffuser_model(prompt=query).images[0]
                    for i in range(num_images)
                ]

if "images" in st.session_state:
    for image in st.session_state.images:
        st.image(image)
        st.divider()
