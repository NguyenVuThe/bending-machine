import streamlit as st
from ultralytics import YOLO
import cv2

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
    
def run_segmentation(image_np, model):
    """Chạy model YOLO để lấy mask và bounding box"""
    results = model(image_np)
    
    if len(results) > 0 and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
        
        box = results[0].boxes[0]
        conf = float(box.conf)
        
        return mask, conf
    return None, None