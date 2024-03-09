import streamlit as st
from streamlit_drawable_canvas import st_canvas
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from torch import float16
from components import unload_model
from models import use_cuda

import PIL
import requests

if use_cuda():
    if "diffuser_model" not in st.session_state:
        unload_model()
        with st.spinner("Loading Diffusion Model..."):
            st.session_state.diffuser_model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                pretrained_model_name_or_path="timbrooks/instruct-pix2pix",
                torch_dtype=float16,
                use_safetensors=True,
            )
            st.session_state.diffuser_model.to("cuda")
            st.session_state.diffuser_model.scheduler = EulerAncestralDiscreteScheduler.from_config(st.session_state.diffuser_model.scheduler.config)
        st.caption("Diffusion model loaded successfully")


    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    with st.form("image_generation_form"):
        ## DRAWING PAD
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=PIL.Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=600,
            width=600,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )

        modifier_prompt = st.text_input("Modifier Prompt")
        num_inference_steps = st.number_input("Num Inference Steps", min_value=1, max_value=None, value=50)
        image_guidance_scale = st.number_input("Image Guidance Scale", min_value=0.1, max_value=None, value=1.75)

        submitted = st.form_submit_button()
        if submitted:
            if canvas_result.image_data is None:
                st.error("Draw something")
            else:
                input_img = PIL.Image.fromarray(canvas_result.image_data)
                input_img = input_img.convert("RGB")
                prompt = f"Convert the canvas drawing into a realistic, natural colored picture."
                if modifier_prompt:
                    prompt += f"Original image info: {modifier_prompt}"
                st.session_state.images = st.session_state.diffuser_model(prompt, image=input_img, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale).images
            

    if "images" in st.session_state:
        for image in st.session_state.images:
            st.image(image)
            st.divider()
else:
    st.error("This app is unavailable without CUDA.")