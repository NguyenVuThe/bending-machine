import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from cocoon_method import run_one_image as run_cocoon_image
from cocoon_shade_method import run_one_image as run_cocoon_shade_image

st.set_page_config(page_title="Cocoon Dewarping", layout="wide")

MODE_RUNNERS = {
    "Cocoon Shade": run_cocoon_shade_image,
    "Cocoon": run_cocoon_image,
}

PREVIEW_WIDTH = 420


@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def save_uploaded_image(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def display_phase_images(original_image, mask, image_corners, image_mesh, warped_img, conf):
    st.subheader("Processing Phases")

    phases = [
        ("1. Original", original_image),
        (f"2. Mask (Conf: {conf:.2f})", (mask * 255).astype(np.uint8)),
        ("3. Corners", image_corners),
        ("4. Mesh", image_mesh),
        ("5. Output", warped_img),
    ]

    first_row = st.columns(3)
    second_row = st.columns(2)

    for column, (caption, image) in zip(first_row + second_row, phases):
        with column:
            st.image(image, caption=caption, width="stretch")


st.title("Bending Machine.v2")

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        list(MODE_RUNNERS.keys()),
        index=0,
        help="Cocoon Shade uses shading gradients to relax the mesh before dewarping.",
    )
    model_path = st.text_input("YOLO model path", value="best.pt")
    epsilon_factor = st.slider(
        "Epsilon factor",
        min_value=0.001,
        max_value=0.100,
        value=0.050,
        step=0.001,
        format="%.3f",
        help="Controls corner extraction. Smaller values follow the mask contour more closely.",
    )
    alpha_shading = st.slider(
        "Alpha shading",
        min_value=0.0,
        max_value=80.0,
        value=30.0,
        step=1.0,
        disabled=mode != "Cocoon Shade",
        help="Controls how strongly shading gradients relax the mesh in Cocoon Shade mode.",
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image to run the selected cocoon method.")
else:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Uploaded Image", width=PREVIEW_WIDTH)

    if st.button("Run", type="primary"):
        temp_image_path = save_uploaded_image(uploaded_file)

        try:
            with st.spinner(f"Running {mode}..."):
                model = load_model(model_path)
                runner = MODE_RUNNERS[mode]
                if mode == "Cocoon Shade":
                    mask, image_corners, image_mesh, warped_img, conf = runner(
                        temp_image_path,
                        model,
                        epsilon_factor=epsilon_factor,
                        alpha_shading=alpha_shading,
                    )
                else:
                    mask, image_corners, image_mesh, warped_img, conf = runner(
                        temp_image_path,
                        model,
                        epsilon_factor=epsilon_factor,
                    )

            display_phase_images(
                original_image,
                mask,
                image_corners,
                image_mesh,
                warped_img,
                conf,
            )
        except Exception as exc:
            st.error(f"{mode} failed: {exc}")
        finally:
            temp_image_path.unlink(missing_ok=True)
