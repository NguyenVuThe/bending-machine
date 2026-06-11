import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import từ các module đã chia nhỏ
from model import load_yolo_model
from pipeline import run_segmentation, extract_corners, apply_perspective

st.set_page_config(layout="wide", page_title="Bending Machine")

# ==========================================
# SIDEBAR: CẤU HÌNH PIPELINE TỪ UI
# ==========================================
st.sidebar.title("⚙️ Tùy chỉnh Pipeline")

# 1. Tùy chỉnh Model
model_path = st.sidebar.text_input("Đường dẫn Model YOLO", value="best.pt", help="Nhập file .pt của bạn (VD: best.pt, yolov8n-seg.pt)")

# 2. Tùy chỉnh Thuật toán tìm góc
corner_method = st.sidebar.selectbox(
    "Thuật toán trích xuất góc", 
    ["ApproxPolyDP", "MinAreaRect"],
    help="ApproxPolyDP bám sát đường viền mask. MinAreaRect tạo ra một khung chữ nhật bao ngoài vùng mask."
)

# 3. Tùy chỉnh Epsilon (Chỉ hiện nếu chọn ApproxPolyDP)
if corner_method == "ApproxPolyDP":
    epsilon_val = st.sidebar.slider(
        "Hệ số Epsilon", 
        min_value=0.001, max_value=0.100, value=0.020, step=0.001,
        help="Càng nhỏ thì đa giác càng nhiều cạnh, càng lớn thì đa giác càng ít cạnh (mục tiêu là tìm ra đúng 4 cạnh)."
    )
else:
    epsilon_val = 0.02 # Giá trị rác, không dùng đến khi chạy MinAreaRect

st.sidebar.markdown("---")
st.sidebar.info("💡 Mẹo: Chỉnh thông số bên trên và nhấn chạy lại để xem thuật toán nào hoạt động tốt nhất với ảnh của bạn.")

# ==========================================
# MAIN UI: HIỂN THỊ VÀ XỬ LÝ
# ==========================================
st.title("Bending Machine")

# Khởi tạo model dựa trên cấu hình (sẽ tự động reload nếu đổi đường dẫn)
model = load_yolo_model(model_path)

uploaded_file = st.file_uploader("Tải lên ảnh cần test", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    st.markdown("### Kết quả Pipeline")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image, caption="1. Ảnh gốc", use_container_width=True)
    
    if st.button("🚀 Chạy Pipeline với thông số hiện tại", type="primary"):
        if model is None:
            st.error("Model chưa sẵn sàng. Hãy kiểm tra lại đường dẫn ở thanh công cụ bên trái.")
        else:
            with st.spinner('Đang xử lý...'):
                
                # BƯỚC 1: SEGMENTATION
                mask, conf = run_segmentation(image_np, model)
                
                if mask is not None:
                    with col2:
                        st.image((mask * 255).astype(np.uint8), caption=f"2. Mask (Conf: {conf:.2f})", use_container_width=True)
                    
                    # BƯỚC 2: TÌM GÓC (Áp dụng thuật toán từ cấu hình UI)
                    corners = extract_corners(mask, method=corner_method, epsilon_factor=epsilon_val)
                    
                    if corners is not None:
                        img_corners = image_np.copy()
                        for point in corners:
                            cv2.circle(img_corners, (int(point[0]), int(point[1])), 15, (0, 255, 0), -1)
                        with col3:
                            st.image(img_corners, caption=f"3. Corners ({corner_method})", use_container_width=True)
                            
                        # BƯỚC 3: CĂNG PHẲNG
                        warped_img = apply_perspective(image_np, corners)
                        with col4:
                            st.image(warped_img, caption="4. Output", use_container_width=True)
                            
                    else:
                        st.error(f"Thuật toán **{corner_method}** với Epsilon = **{epsilon_val}** không thể tìm ra đúng 4 góc. Hãy thử điều chỉnh thanh Epsilon trên Sidebar.")
                else:
                    st.error("Model không nhận diện được mask trên ảnh này.")