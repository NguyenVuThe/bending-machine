import csv
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from skimage.transform import PiecewiseAffineTransform
from skimage.transform import warp
from model import run_segmentation
from utils.io import save_rgb, save_mask, find_images
from utils.math import nearest_idx, extract_corners, extract_largest_contour, resample_curve, extract_edge_safe
from utils.visualize import draw_corners, draw_contours, draw_mesh, draw_edges, draw_resampled_edges

MODEL_PATH = "best.pt"
OUTPUT_DIR = "outputs"
CORNER_METHOD = "ApproxPolyDP"
EPSILON_FACTOR = 0.05

def run_one_image(image_path, model, epsilon_factor=EPSILON_FACTOR, alpha_shading=30.0):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot find image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Step 1: Segmentation
    mask, conf = run_segmentation(image_np, model)
    if mask is None:
        raise RuntimeError("Model cannot detect a mask on this image.")

    # Step 2: Corner extraction
    corners = extract_corners(mask, method=CORNER_METHOD, epsilon_factor=epsilon_factor)
    print(f"Extracted corners: {corners}")
    if corners is None:
        raise RuntimeError(
            f"{CORNER_METHOD} cannot find exactly 4 corners. "
            f"Try changing epsilon_factor, current value: {epsilon_factor}"
        )

    image_corners = draw_corners(image_np, corners)
    TL, TR, BR, BL = [np.array(pt) for pt in corners] # Chuyển về numpy array để tính toán an toàn
    
    largest_contour = extract_largest_contour(mask)
    boundary = largest_contour.squeeze()

    idx_tl = nearest_idx(boundary, TL)
    idx_tr = nearest_idx(boundary, TR)
    idx_br = nearest_idx(boundary, BR)
    idx_bl = nearest_idx(boundary, BL)


    # Trích xuất 4 cạnh một cách an toàn bằng hàm trợ giúp mới
    top_edge = extract_edge_safe(boundary, idx_tl, idx_tr)
    bottom_edge = extract_edge_safe(boundary, idx_bl, idx_br)
    left_edge = extract_edge_safe(boundary, idx_tl, idx_bl)
    right_edge = extract_edge_safe(boundary, idx_tr, idx_br)

    cols = 100
    rows = 60

    # Nội suy đều các điểm trên 4 cạnh
    top = resample_curve(top_edge, cols)
    bottom = resample_curve(bottom_edge, cols)
    left = resample_curve(left_edge, rows)
    right = resample_curve(right_edge, rows)

    # Tạo Lưới Coons Patch (Source Mesh)
    mesh = []
    for c in range(cols):
        u = c / (cols - 1)
        column = []
        for r in range(rows):
            v = r / (rows - 1)

            # Nội suy từ viền
            C = (1 - v) * top[c] + v * bottom[c]
            D = (1 - u) * left[r] + u * right[r]

            # Hiệu chỉnh 4 góc (Bilinear corner correction)
            B = (1 - u) * (1 - v) * TL + u * (1 - v) * TR + (1 - u) * v * BL + u * v * BR

            # Điểm thực tế của lưới (Coons Patch)
            pt = C + D - B
            column.append(pt)
        mesh.append(column)

    mesh = np.array(mesh)  # Hình dạng (cols, rows, 2)
    
    # 1. Chuyển ảnh sang Grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # 2. Xóa chữ bằng Morphological Dilation
    # Kích thước kernel phải lớn hơn độ dày nét chữ lớn nhất trong ảnh
    kernel_size = 15 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(gray, kernel, iterations=1)

    # 3. Làm mượt để tạo Shading Map (Illumination)
    shading_map = cv2.GaussianBlur(dilated, (51, 51), sigmaX=15, sigmaY=15)
    
    # Chuẩn hóa về dải [0.0, 1.0] để tính toán
    shading_norm = shading_map.astype(np.float32) / 255.0

    # 4. Tính Gradient (đạo hàm bậc 1) theo trục X và Y bằng toán tử Sobel
    # Gradient giúp xác định hướng của "nếp gấp" trên giấy
    grad_x = cv2.Sobel(shading_norm, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(shading_norm, cv2.CV_32F, 0, 1, ksize=5)

    for c in range(1, cols - 1):
        for r in range(1, rows - 1):
            pt = mesh[c, r]
            x, y = int(pt[0]), int(pt[1])
            
            # Đảm bảo tọa độ an toàn không vượt quá viền ảnh
            x = np.clip(x, 0, image_np.shape[1] - 1)
            y = np.clip(y, 0, image_np.shape[0] - 1)
            
            # Lấy vector gradient tại điểm ảnh này
            dx = grad_x[y, x]
            dy = grad_y[y, x]
            
            # Tính trọng số dựa trên độ tối (Shading Intensity)
            # intensity = 1.0 (sáng/phẳng) -> weight = 0 -> Không dịch chuyển
            # intensity = 0.0 (tối/sâu) -> weight = alpha -> Dịch chuyển tối đa
            intensity = shading_norm[y, x]
            weight = (1.0 - intensity) * alpha_shading
            shift_x, shift_y = dx * weight, dy * weight
            
            # Cập nhật tọa độ điểm lưới
            mesh[c, r, 0] += np.clip(shift_x, -10.0, 10.0)
            mesh[c, r, 1] += np.clip(shift_y, -10.0, 10.0)

    image_mesh = draw_mesh(image_np, mesh)

    # Tính toán kích thước output động theo tỷ lệ thực tế
    W = int(max(np.linalg.norm(TR - TL), np.linalg.norm(BR - BL)))
    H = int(max(np.linalg.norm(BL - TL), np.linalg.norm(BR - TR)))
    print(f"Dynamic Output Size: {W}x{H}")

    # Tạo Lưới phẳng (Target Mesh)
    target_mesh = []
    for c in range(cols):
        x = c * W / (cols - 1)
        column = []
        for r in range(rows):
            y = r * H / (rows - 1)
            column.append([x, y])
        target_mesh.append(column)

    target_mesh = np.array(target_mesh)  # Hình dạng (cols, rows, 2)

    # Định dạng điểm cho thuật toán Piecewise Affine
    src_pts = mesh.reshape(-1, 2)
    dst_pts = target_mesh.reshape(-1, 2)

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_pts, src_pts)

    # Áp dụng Dewarping lên ảnh
    warped = warp(image_np, tform, output_shape=(H, W))
    warped_img = (warped * 255).astype(np.uint8)

    return mask, image_corners, image_mesh, warped_img, conf


def main(image_path):
    output_dir = Path(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)
    mask, image_corners, image_mesh, warped_img, conf = run_one_image(image_path, model)

    save_mask(output_dir / "01_mask.png", mask)
    save_rgb(output_dir / "02_corners.png", image_corners)
    save_rgb(output_dir / "03_mesh.png", image_mesh)
    save_rgb(output_dir / "04_output.png", warped_img)

    print(f"Done: {image_path}")
    print(f"Confidence: {conf:.2f}")
    print(f"Saved results to: {output_dir}")


def process_images(input_folder_path):
    input_folder_path = Path(input_folder_path)
    all_steps_dir = Path(OUTPUT_DIR) / "all_steps"
    status_csv_path = all_steps_dir / "status.csv"

    if not input_folder_path.is_dir():
        raise FileNotFoundError(f"Cannot find input folder: {input_folder_path}")

    image_paths = find_images(input_folder_path)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_folder_path}")

    model = YOLO(MODEL_PATH)
    success_count = 0
    rows = []

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_folder_path)
        output_name = relative_path.with_suffix(".png")

        print(f"[{index}/{len(image_paths)}] {relative_path}")

        try:
            mask, image_corners, image_mesh, warped_img, conf = run_one_image(image_path, model)

            save_mask(all_steps_dir / "masks" / output_name, mask)
            save_rgb(all_steps_dir / "corners" / output_name, image_corners)
            save_rgb(all_steps_dir / "meshes" / output_name, image_mesh)
            save_rgb(all_steps_dir / "outputs" / output_name, warped_img)

            success_count += 1
            rows.append(
                {
                    "image": str(relative_path),
                    "status": "processed",
                    "confidence": f"{conf:.4f}",
                }
            )
            print(f"  OK, confidence: {conf:.2f}")
        except Exception as exc:
            rows.append(
                {
                    "image": str(relative_path),
                    "status": "not_processed",
                    "confidence": "",
                }
            )
            print(f"  Failed: {exc}")

    status_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with status_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "status", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {success_count}/{len(image_paths)} images processed")
    print(f"All steps saved to: {all_steps_dir}")
    print(f"Status CSV saved to: {status_csv_path}")


if __name__ == "__main__":
    # Single image:
    main("images/0141_jpg.rf.4a20e0357c228d9c352b1628867d9d52.jpg")

    # Folder of images:
    # process_images("images")
