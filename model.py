import streamlit as st
from ultralytics import YOLO

@st.cache_resource
def load_yolo_model(model_path):
    """
    Tải model YOLO từ đường dẫn tùy chọn.
    Sử dụng cache để tối ưu tốc độ nếu đường dẫn không thay đổi.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Lỗi khi tải model từ '{model_path}': {e}")
        return None